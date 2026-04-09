"""
msasl_extract.py — Extract MediaPipe landmarks from downloaded MS-ASL clips.

Scans data/msasl_videos/<gloss>/<video_id>.mp4 and runs the same MediaPipe
HolisticLandmarker pipeline used for WLASL, saving .npz files to:

    data/wlasl_landmarks/<gloss>/<video_id>.npz

Because the output lands in the same directory as WLASL landmarks, the
existing wlasl_train.py picks up all data from both datasets automatically.

Run after msasl_download.py (or after unpacking a pre-downloaded archive).

Usage:
    python src/msasl_extract.py [--seq-len N] [--workers W]
"""

import argparse
import sys
import multiprocessing as mp_proc
from pathlib import Path

# Reuse all normalisation / extraction logic from the WLASL pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wlasl_extract import (   # noqa: E402
    ensure_model,
    process_video,
    _default_workers,
    SEQ_LEN_DEFAULT,
)

ROOT      = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "msasl_videos"
OUT_DIR   = ROOT / "data" / "wlasl_landmarks"   # shared with WLASL


def collect_tasks(seq_len: int) -> list[tuple]:
    tasks = []
    for gloss_dir in sorted(VIDEO_DIR.iterdir()):
        if not gloss_dir.is_dir():
            continue
        for mp4 in sorted(gloss_dir.glob("*.mp4")):
            out = OUT_DIR / gloss_dir.name / mp4.with_suffix(".npz").name
            tasks.append((mp4, out, seq_len))
    return tasks


def process_star(args):
    return process_video(*args)


def main():
    parser = argparse.ArgumentParser(description="Extract MS-ASL landmarks")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN_DEFAULT,
                        help=f"Sequence length in frames (default: {SEQ_LEN_DEFAULT})")
    parser.add_argument("--workers", type=int, default=_default_workers(),
                        help="Parallel worker processes")
    args = parser.parse_args()

    if not VIDEO_DIR.exists() or not any(VIDEO_DIR.iterdir()):
        print(f"No videos found in {VIDEO_DIR}")
        print("Run msasl_download.py first, or unpack the MS-ASL archive there.")
        return

    ensure_model()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = collect_tasks(args.seq_len)
    if not tasks:
        print("No .mp4 files found.")
        return

    already = sum(1 for (_, out, _) in tasks if out.exists())
    to_do   = len(tasks) - already
    print(f"MS-ASL clips found  : {len(tasks)}")
    print(f"Already extracted   : {already}")
    print(f"To process          : {to_do}")
    print(f"Sequence length     : {args.seq_len} frames")
    print(f"Workers             : {args.workers}")
    print(f"Output → {OUT_DIR}\n")

    if to_do == 0:
        print("Nothing to do.")
        return

    ok = failed = skipped = 0
    with mp_proc.Pool(processes=args.workers) as pool:
        for i, (success, msg) in enumerate(
            pool.imap_unordered(process_star, tasks), 1
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
    print(f"Landmarks in {OUT_DIR}")


if __name__ == "__main__":
    main()
