"""
asl_citizen_extract.py — Extract per-frame holistic landmarks from ASL Citizen videos.

Reads data/ASL_Citizen/splits/{train,val,test}.csv, runs MediaPipe HolisticLandmarker
on every video and saves normalised landmark sequences as .npz files.

Output layout:
    data/asl_citizen_landmarks/<gloss>/<video_stem>.npz

Gloss normalisation: trailing digits stripped, lowercased.
  e.g.  "DOG1" → "dog",  "SOCCER2" → "soccer"

Feature format is identical to wlasl_extract.py (225 floats / frame, SEQ_LEN frames)
so both datasets can be loaded together by the training script.

    pose       33 × 3 = 99   (hip-centred, shoulder-scale normalised)
    left_hand  21 × 3 = 63   (wrist-centred, unit-scale)
    right_hand 21 × 3 = 63   (wrist-centred, unit-scale)

Usage:
    python src/asl_citizen_extract.py [--seq-len N] [--workers W] [--threads T] [--throttle S] [--splits SPLITS] [--delegate cpu|gpu]

Options:
    --seq-len   Fixed sequence length in frames (default: 64)
    --workers   Parallel worker processes (default: conservative auto, max 2)
    --threads   MediaPipe CPU threads per worker (default: 2)
    --throttle  Seconds to sleep after each video, reduces CPU spikes (default: 0)
    --splits    Comma-separated splits to process (default: train,val,test)
    --delegate  MediaPipe delegate: cpu or gpu (default: cpu). gpu offloads inference
                to Jetson's integrated GPU; falls back to cpu if init fails.
"""

import argparse
import csv
import multiprocessing as mp_proc
import os
import re
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

ROOT       = Path(__file__).resolve().parent.parent
VIDEO_DIR  = ROOT / "data" / "ASL_Citizen" / "videos"
SPLITS_DIR = ROOT / "data" / "ASL_Citizen" / "splits"
OUT_DIR    = ROOT / "data" / "asl_citizen_landmarks"
MODEL_DIR  = ROOT / "models"
POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker.task"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

SEQ_LEN_DEFAULT = 64
POSE_N   = 33
HAND_N   = 21
COORDS   = 3
FEAT_DIM = (POSE_N + HAND_N + HAND_N) * COORDS   # 225


# ── Helpers ────────────────────────────────────────────────────────────────────

def normalise_gloss(raw: str) -> str:
    """Strip trailing digits, lowercase.  'DOG1' → 'dog', 'SOCCER2' → 'soccer'."""
    return re.sub(r"\d+$", "", raw.strip()).lower()


def _available_ram_gb() -> float | None:
    try:
        return (os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / (1024 ** 3)
    except (AttributeError, OSError, ValueError):
        return None


def _default_workers() -> int:
    cpu_cap = max(1, mp_proc.cpu_count() - 2)  # leave 2 CPUs free
    ram_gb  = _available_ram_gb()
    ram_cap = max(1, int(ram_gb // 4)) if ram_gb else 1  # 4 GB per worker
    return max(1, min(cpu_cap, ram_cap, 2))  # cap at 2 by default


def ensure_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for path, url, name in [
        (POSE_MODEL_PATH, POSE_MODEL_URL, "pose"),
        (HAND_MODEL_PATH, HAND_MODEL_URL, "hand"),
    ]:
        if not path.exists():
            print(f"Downloading {name} landmarker model → {path} …")
            urllib.request.urlretrieve(url, path)
            print("  Done.")


# ── Normalisation (identical to wlasl_extract.py) ─────────────────────────────

def _lm_to_array(landmarks) -> np.ndarray:
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def normalise_hand(pts: np.ndarray) -> np.ndarray:
    pts = pts - pts[0]
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()


def normalise_pose(pts: np.ndarray) -> np.ndarray:
    centre = (pts[23] + pts[24]) / 2.0
    pts = pts - centre
    shoulder_dist = np.linalg.norm(pts[11] - pts[12])
    if shoulder_dist > 1e-6:
        pts /= shoulder_dist
    return pts.flatten()


# ── Per-video processing ───────────────────────────────────────────────────────

def _make_options(delegate: str) -> tuple:
    d = BaseOptions.Delegate.GPU if delegate == "gpu" else BaseOptions.Delegate.CPU
    pose_opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH), delegate=d),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    hand_opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH), delegate=d),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return pose_opts, hand_opts


def process_video(video_path: Path, out_path: Path, seq_len: int,
                  throttle_s: float = 0.0, delegate: str = "cpu") -> tuple[bool, str]:
    if out_path.exists():
        return True, "already exists"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "cannot open video"

    pose_options, hand_options = _make_options(delegate)

    frames: list[np.ndarray] = []
    try:
        with (PoseLandmarker.create_from_options(pose_options) as pose_detector,
              HandLandmarker.create_from_options(hand_options) as hand_detector):
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_idx = 0
            while frame_idx < seq_len:
                ret, bgr = cap.read()
                if not ret:
                    break
                rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms  = int(frame_idx * 1000 / fps)

                pose_result = pose_detector.detect_for_video(mp_img, ts_ms)
                hand_result = hand_detector.detect_for_video(mp_img, ts_ms)

                # Extract pose landmarks
                pose_vec = normalise_pose(_lm_to_array(pose_result.pose_landmarks[0])) \
                           if pose_result.pose_landmarks else np.zeros(POSE_N * COORDS, dtype=np.float32)

                # Extract hand landmarks (left and right)
                # Hand detector returns both in one result, sorted by handedness
                left_hand = None
                right_hand = None
                if hand_result.handedness:
                    for i, hand in enumerate(hand_result.handedness):
                        if hand[0].category_name == "Left":
                            left_hand = hand_result.hand_landmarks[i]
                        elif hand[0].category_name == "Right":
                            right_hand = hand_result.hand_landmarks[i]

                lh_vec = normalise_hand(_lm_to_array(left_hand)) \
                         if left_hand else np.zeros(HAND_N * COORDS, dtype=np.float32)
                rh_vec = normalise_hand(_lm_to_array(right_hand)) \
                         if right_hand else np.zeros(HAND_N * COORDS, dtype=np.float32)

                frames.append(np.concatenate([pose_vec, lh_vec, rh_vec]).astype(np.float32))
                frame_idx += 1
    finally:
        cap.release()

    if not frames:
        return False, "no frames extracted"

    seq = np.stack(frames, axis=0)
    T   = seq.shape[0]
    if T < seq_len:
        pad = np.zeros((seq_len - T, FEAT_DIM), dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=0)
    else:
        seq = seq[:seq_len]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, features=seq,
                        original_length=np.array(min(T, seq_len)))

    if throttle_s > 0:
        time.sleep(throttle_s)

    return True, f"OK ({T} frames)"


def process_video_star(args):
    return process_video(*args)


# ── Task collection ────────────────────────────────────────────────────────────

def collect_tasks(splits: list[str], seq_len: int, throttle_s: float,
                  delegate: str) -> list[tuple]:
    seen = set()
    tasks = []
    for split in splits:
        csv_path = SPLITS_DIR / f"{split}.csv"
        if not csv_path.exists():
            print(f"  Warning: {csv_path} not found, skipping.")
            continue
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                video_file = row["Video file"].strip()
                gloss      = normalise_gloss(row["Gloss"])
                video_path = VIDEO_DIR / video_file
                if not video_path.exists():
                    continue
                key = video_file
                if key in seen:
                    continue
                seen.add(key)
                stem     = Path(video_file).stem
                out_path = OUT_DIR / gloss / f"{stem}.npz"
                tasks.append((video_path, out_path, seq_len, throttle_s, delegate))
    return tasks


# ── Main ──────────────────────────────────────────────────────────────────────

def _worker_init(num_threads: int):
    """Lower process priority and cap CPU threads so the host stays responsive."""
    try:
        os.nice(10)
    except OSError:
        pass
    t = str(num_threads)
    os.environ["OMP_NUM_THREADS"]      = t
    os.environ["OPENBLAS_NUM_THREADS"] = t
    os.environ["MKL_NUM_THREADS"]      = t


def main():
    parser = argparse.ArgumentParser(description="Extract ASL Citizen holistic landmarks")
    parser.add_argument("--seq-len",  type=int,   default=SEQ_LEN_DEFAULT)
    parser.add_argument("--workers",  type=int,   default=_default_workers(),
                        help="Parallel worker processes (default: conservative auto)")
    parser.add_argument("--threads",  type=int,   default=2,
                        help="MediaPipe CPU threads per worker (default: 2)")
    parser.add_argument("--throttle", type=float, default=0.0,
                        help="Seconds to sleep after each video (default: 0)")
    parser.add_argument("--splits",   type=str,   default="train,val,test",
                        help="Comma-separated splits to process (default: train,val,test)")
    parser.add_argument("--delegate", type=str,   default="cpu",
                        choices=["cpu", "gpu"],
                        help="MediaPipe delegate: cpu or gpu (default: cpu)")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",")]

    ensure_model()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Collecting tasks from splits: {splits} …")
    tasks = collect_tasks(splits, args.seq_len, args.throttle, args.delegate)
    if not tasks:
        print("No videos found. Check that data/ASL_Citizen/videos/ is populated.")
        return

    already = sum(1 for t in tasks if t[1].exists())
    ram_gb  = _available_ram_gb()
    print(f"Videos found     : {len(tasks)}")
    print(f"Already extracted: {already}")
    print(f"To process       : {len(tasks) - already}")
    print(f"Sequence length  : {args.seq_len} frames")
    print(f"Workers          : {args.workers}")
    print(f"Threads/worker   : {args.threads}")
    print(f"Throttle         : {args.throttle}s")
    print(f"Delegate         : {args.delegate}")
    print(f"RAM              : {f'{ram_gb:.1f} GB' if ram_gb else 'unknown'}\n")

    pending = [t for t in tasks if not t[1].exists()]
    total   = len(pending)
    if not total:
        print("Nothing to do — all videos already extracted.")
        return

    ok = failed = 0
    t_start = time.time()
    with mp_proc.Pool(processes=args.workers, initializer=_worker_init,
                      initargs=(args.threads,)) as pool:
        for i, (success, msg) in enumerate(
            pool.imap_unordered(process_video_star, pending, chunksize=1), 1
        ):
            if success:
                ok += 1
            else:
                failed += 1
                if failed <= 20:
                    print(f"  FAIL [{i}]: {msg}")
            if i % 500 == 0 or i == total:
                elapsed = time.time() - t_start
                rate    = i / elapsed if elapsed > 0 else 0
                eta_s   = (total - i) / rate if rate > 0 else 0
                eta_str = f"{int(eta_s // 3600)}h{int((eta_s % 3600) // 60)}m"
                print(f"  [{i}/{total}]  ok={ok}  failed={failed}"
                      f"  rate={rate:.1f} vid/s  ETA={eta_str}")

    print(f"\nDone.  Extracted={ok}  Failed={failed}  Skipped={already}")
    print(f"Landmarks saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
