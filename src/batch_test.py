"""
Batch inference test: runs 20 random WLASL + 20 random ASL Citizen videos
through infer.py (wordsign, headless) and prints a results table.

Usage:
    python src/batch_test.py [--conf 0.4] [--seed 42]
"""

import argparse
import csv
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
WLASL_DIR  = ROOT / "data" / "wlasl_videos"
CITIZEN_DIR = ROOT / "data" / "ASL_Citizen" / "videos"
INFER_PY   = ROOT / "src" / "infer.py"
CSV_OUT    = ROOT / "headless_out"   # infer.py writes CSVs here by default


def sample_wlasl(n: int, rng: random.Random) -> list[tuple[Path, str]]:
    """Return n random (video_path, true_label) pairs from wlasl_videos/."""
    pairs = []
    for label_dir in WLASL_DIR.iterdir():
        if not label_dir.is_dir():
            continue
        for mp4 in label_dir.glob("*.mp4"):
            pairs.append((mp4, label_dir.name.lower()))
    return rng.sample(pairs, min(n, len(pairs)))


def sample_citizen(n: int, rng: random.Random) -> list[tuple[Path, str]]:
    """Return n random (video_path, true_label) pairs from ASL_Citizen/videos/.
    Filename format: <id>-<LABEL>.mp4  or  <id>-<LABEL> <N>.mp4
    """
    pairs = []
    for mp4 in CITIZEN_DIR.glob("*.mp4"):
        # Extract label: everything after the first '-', strip trailing ' N'
        stem = mp4.stem
        dash = stem.find("-")
        if dash == -1:
            continue
        raw_label = stem[dash + 1:].strip()
        # Remove trailing digit suffix like " 1", " 2"
        parts = raw_label.rsplit(" ", 1)
        if len(parts) == 2 and parts[1].isdigit():
            raw_label = parts[0]
        pairs.append((mp4, raw_label.lower()))
    return rng.sample(pairs, min(n, len(pairs)))


def run_video(video_path: Path, conf: float) -> list[str]:
    """Run infer.py on one video; return list of predicted words."""
    cmd = [
        sys.executable, str(INFER_PY),
        "--mode", "wordsign",
        "--video", str(video_path),
        "--headless",
        "--conf", str(conf),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    # Parse predictions from stdout lines like:
    #   frame  NNNNN  NNNNNNN ms  word  (0.XX)
    preds = []
    for line in result.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0] == "frame" and parts[1].isdigit() and parts[3] == "ms":
            preds.append(parts[4].lower())
    return preds


def majority(preds: list[str]) -> str | None:
    if not preds:
        return None
    counts: dict[str, int] = {}
    for p in preds:
        counts[p] = counts.get(p, 0) + 1
    return max(counts, key=counts.__getitem__)


def print_table(rows: list[dict], title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  {'TRUE LABEL':<22} {'PREDICTION':<22} {'MATCH':<6} {'# PREDS'}")
    print(f"  {'-' * 60}")
    correct = 0
    for r in rows:
        match = r["true"] == r["pred"] if r["pred"] else False
        if match:
            correct += 1
        marker = "YES" if match else ("---" if r["pred"] else "NONE")
        print(f"  {r['true']:<22} {(r['pred'] or '(no prediction)'):<22} {marker:<6} {r['n_preds']}")
    n = len(rows)
    n_pred = sum(1 for r in rows if r["pred"])
    print(f"  {'-' * 60}")
    print(f"  Correct: {correct}/{n}   Predicted: {n_pred}/{n}   Acc: {correct/n*100:.0f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Confidence threshold passed to infer.py (default: 0.4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of videos per source (default: 20)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    wlasl_samples   = sample_wlasl(args.n, rng)
    citizen_samples = sample_citizen(args.n, rng)

    total = len(wlasl_samples) + len(citizen_samples)
    print(f"\nRunning {total} videos (conf={args.conf}, seed={args.seed})\n")

    wlasl_rows   = []
    citizen_rows = []

    for i, (path, true_label) in enumerate(wlasl_samples, 1):
        print(f"[WLASL {i:2d}/{len(wlasl_samples)}] {true_label:20s}  {path.name}")
        preds = run_video(path, args.conf)
        pred  = majority(preds)
        wlasl_rows.append({"true": true_label, "pred": pred, "n_preds": len(preds)})
        print(f"          -> {pred or '(none)'}  ({len(preds)} raw predictions)")

    for i, (path, true_label) in enumerate(citizen_samples, 1):
        print(f"[CITIZEN {i:2d}/{len(citizen_samples)}] {true_label:20s}  {path.name}")
        preds = run_video(path, args.conf)
        pred  = majority(preds)
        citizen_rows.append({"true": true_label, "pred": pred, "n_preds": len(preds)})
        print(f"          -> {pred or '(none)'}  ({len(preds)} raw predictions)")

    print_table(wlasl_rows,   "WLASL Videos")
    print_table(citizen_rows, "ASL Citizen Videos")

    # Combined summary
    all_rows = wlasl_rows + citizen_rows
    correct  = sum(1 for r in all_rows if r["pred"] == r["true"])
    n_pred   = sum(1 for r in all_rows if r["pred"])
    print(f"\n{'=' * 70}")
    print(f"  OVERALL  Correct: {correct}/{len(all_rows)}   "
          f"Predicted: {n_pred}/{len(all_rows)}   "
          f"Acc: {correct/len(all_rows)*100:.0f}%")
    print(f"{'=' * 70}\n")

    # Write results CSV
    out_csv = ROOT / "results.txt"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "video", "true_label", "prediction", "match", "n_preds"])
        for r, (p, _) in zip(wlasl_rows, wlasl_samples):
            w.writerow(["wlasl", p.name, r["true"], r["pred"] or "", r["true"] == r["pred"], r["n_preds"]])
        for r, (p, _) in zip(citizen_rows, citizen_samples):
            w.writerow(["citizen", p.name, r["true"], r["pred"] or "", r["true"] == r["pred"], r["n_preds"]])
    print(f"Results saved -> {out_csv}")


if __name__ == "__main__":
    main()
