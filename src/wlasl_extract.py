"""
wlasl_extract.py — Extract per-frame holistic landmarks from downloaded WLASL videos.

For each video in data/wlasl_videos/<gloss>/<video_id>.mp4, runs MediaPipe
HolisticLandmarker on every frame and saves a normalised landmark sequence as
a NumPy .npz file in data/wlasl_landmarks/<gloss>/<video_id>.npz.

Feature vector per frame (225 floats):
    pose       33 × 3 = 99   (translation + scale normalised)
    left_hand  21 × 3 = 63   (wrist-centred, unit-scale)
    right_hand 21 × 3 = 63   (wrist-centred, unit-scale)

Sequences are padded (zeros) or truncated to SEQ_LEN frames.

Usage:
    python src/wlasl_extract.py [--seq-len N] [--workers W]
"""

import argparse
import urllib.request
from pathlib import Path
import multiprocessing as mp_proc
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision import (
    HolisticLandmarker,
    HolisticLandmarkerOptions,
    RunningMode,
)

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "wlasl_videos"
OUT_DIR   = ROOT / "data" / "wlasl_landmarks"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "holistic_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
)

SEQ_LEN_DEFAULT = 64   # frames (covers ~median 2.2 s clip at 25 fps)
POSE_N  = 33
HAND_N  = 21
COORDS  = 3
FEAT_DIM = (POSE_N + HAND_N + HAND_N) * COORDS   # 225


def _available_ram_gb() -> float | None:
    """Best-effort available system RAM in GB on Linux/Unix."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return (pages * page_size) / (1024 ** 3)


def _default_workers() -> int:
    """
    Conservative worker default for MediaPipe extraction.
    This avoids OOM crashes on CPU-only machines with limited RAM.
    """
    cpu_cap = max(1, mp_proc.cpu_count() - 1)
    ram_gb = _available_ram_gb()
    if ram_gb is None:
        # Safe fallback when RAM is unknown.
        return min(2, cpu_cap)
    # Roughly budget ~3.5 GB RAM per worker process (detector + video decode overhead).
    ram_cap = max(1, int(ram_gb // 3.5))
    return max(1, min(cpu_cap, ram_cap))


def ensure_model():
    if MODEL_PATH.exists():
        return
    print(f"Downloading holistic landmarker model → {MODEL_PATH} …")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("  Done.")


# ── Normalisation helpers ──────────────────────────────────────────────────────

def _lm_to_array(landmarks) -> np.ndarray:
    """Convert a list of NormalizedLandmark to (N, 3) float32."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def normalise_hand(pts: np.ndarray) -> np.ndarray:
    """Translate wrist to origin, scale to unit sphere. Returns (N*3,)."""
    pts = pts - pts[0]
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()


def normalise_pose(pts: np.ndarray) -> np.ndarray:
    """
    Centre at midpoint of hips (landmarks 23 & 24).
    Scale by distance between shoulders (landmarks 11 & 12).
    Returns (N*3,).
    """
    centre = (pts[23] + pts[24]) / 2.0
    pts = pts - centre
    shoulder_dist = np.linalg.norm(pts[11] - pts[12])
    if shoulder_dist > 1e-6:
        pts /= shoulder_dist
    return pts.flatten()


def extract_frame_features(result) -> np.ndarray:
    """Build a 225-float feature vector from a HolisticLandmarkerResult."""
    # Pose
    if result.pose_landmarks:
        pose_vec = normalise_pose(_lm_to_array(result.pose_landmarks))
    else:
        pose_vec = np.zeros(POSE_N * COORDS, dtype=np.float32)

    # Left hand
    if result.left_hand_landmarks:
        lh_vec = normalise_hand(_lm_to_array(result.left_hand_landmarks))
    else:
        lh_vec = np.zeros(HAND_N * COORDS, dtype=np.float32)

    # Right hand
    if result.right_hand_landmarks:
        rh_vec = normalise_hand(_lm_to_array(result.right_hand_landmarks))
    else:
        rh_vec = np.zeros(HAND_N * COORDS, dtype=np.float32)

    return np.concatenate([pose_vec, lh_vec, rh_vec]).astype(np.float32)


# ── Per-video processing ───────────────────────────────────────────────────────

def process_video(video_path: Path, out_path: Path, seq_len: int) -> tuple[bool, str]:
    """Extract landmark sequence from one video. Returns (success, message)."""
    if out_path.exists():
        return True, "already exists"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "cannot open video"

    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.3,
        min_pose_landmarks_confidence=0.3,
        min_hand_landmarks_confidence=0.3,
    )

    frames: list[np.ndarray] = []
    try:
        with HolisticLandmarker.create_from_options(options) as detector:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_idx = 0
            while True:
                # We only need seq_len features; don't process extra frames.
                if frame_idx >= seq_len:
                    break
                ret, bgr = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int(frame_idx * 1000 / fps)
                result = detector.detect_for_video(mp_img, ts_ms)
                frames.append(extract_frame_features(result))
                frame_idx += 1
    finally:
        cap.release()

    if not frames:
        return False, "no frames extracted"

    seq = np.stack(frames, axis=0)   # (T, 225)
    T = seq.shape[0]

    # Pad or truncate to seq_len
    if T >= seq_len:
        seq = seq[:seq_len]
    else:
        pad = np.zeros((seq_len - T, FEAT_DIM), dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, features=seq, original_length=np.array(min(T, seq_len)))
    return True, f"OK ({T} frames)"


def process_video_star(args):
    return process_video(*args)


# ── Main ──────────────────────────────────────────────────────────────────────

def collect_tasks(seq_len: int) -> list[tuple]:
    tasks = []
    for gloss_dir in sorted(VIDEO_DIR.iterdir()):
        if not gloss_dir.is_dir():
            continue
        gloss = gloss_dir.name
        for mp4 in sorted(gloss_dir.glob("*.mp4")):
            out = OUT_DIR / gloss / mp4.with_suffix(".npz").name
            tasks.append((mp4, out, seq_len))
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Extract WLASL holistic landmarks")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN_DEFAULT,
                        help=f"Fixed sequence length in frames (default: {SEQ_LEN_DEFAULT})")
    parser.add_argument("--workers", type=int, default=_default_workers(),
                        help="Parallel worker processes (default: RAM-aware safe value)")
    args = parser.parse_args()

    if args.seq_len <= 0:
        raise SystemExit("--seq-len must be > 0")
    if args.workers <= 0:
        raise SystemExit("--workers must be > 0")

    ensure_model()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = collect_tasks(args.seq_len)
    if not tasks:
        print("No videos found in", VIDEO_DIR)
        print("Run wlasl_download.py first.")
        return

    already = sum(1 for (_, out, _) in tasks if out.exists())
    print(f"Videos found: {len(tasks)}  |  Already extracted: {already}  "
          f"|  To process: {len(tasks) - already}")
    ram_gb = _available_ram_gb()
    ram_txt = f"{ram_gb:.1f} GB" if ram_gb is not None else "unknown"
    print(f"Sequence length: {args.seq_len} frames  |  Workers: {args.workers}  |  RAM: {ram_txt}\n")

    ok = failed = skipped = 0
    # Use multiprocessing — each worker creates its own MediaPipe detector
    with mp_proc.Pool(processes=args.workers) as pool:
        for i, (success, msg) in enumerate(
            pool.imap_unordered(process_video_star, tasks), 1
        ):
            if "already" in msg:
                skipped += 1
            elif success:
                ok += 1
            else:
                failed += 1
                if failed <= 20:
                    print(f"  FAIL [{i}]: {msg}")
            if i % 200 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}]  ok={ok}  failed={failed}  skipped={skipped}")

    print(f"\nDone.  Extracted={ok}  Failed={failed}  Skipped={skipped}")
    print(f"Landmarks saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
