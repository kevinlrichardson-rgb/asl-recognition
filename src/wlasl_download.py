"""
wlasl_download.py — Download WLASL video clips from direct MP4 URLs.

Reads data/WLASL_v0_3_json/WLASL_v0.3.json and downloads every instance
whose URL is a direct MP4 (YouTube links are skipped — use yt-dlp separately
if needed).

Output layout:
    data/wlasl_videos/<gloss>/<video_id>.mp4

Usage:
    python src/wlasl_download.py [--glosses N] [--workers W] [--split SPLIT]

Options:
    --glosses   Only download the first N glosses (default: all 2000)
    --workers   Parallel download threads (default: 8)
    --split     Only download instances from this split: train | val | test | all (default: all)
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "data" / "WLASL_v0_3_json" / "WLASL_v0.3.json"
OUT_DIR = ROOT / "data" / "wlasl_videos"
FAILED_LOG = ROOT / "data" / "wlasl_videos" / "failed_urls.txt"

TIMEOUT = 20          # seconds per request
CHUNK = 1 << 16       # 64 KB read chunks


def is_direct_mp4(url: str) -> bool:
    if not url:
        return False
    if "youtube.com" in url or "youtu.be" in url:
        return False
    return ".mp4" in url or url.lower().endswith(".mp4")


def download_one(gloss: str, video_id: str, url: str, out_path: Path) -> tuple[bool, str]:
    """Download a single video. Returns (success, message)."""
    if out_path.exists() and out_path.stat().st_size > 1024:
        return True, "already exists"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    try:
        resp = requests.get(url, stream=True, timeout=TIMEOUT,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK):
                if chunk:
                    f.write(chunk)
        tmp.rename(out_path)
        return True, f"OK ({out_path.stat().st_size // 1024} KB)"
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        return False, str(exc)


def build_task_list(data: list, max_glosses: int, split_filter: str) -> list[dict]:
    tasks = []
    for gloss_entry in data[:max_glosses]:
        gloss = gloss_entry["gloss"].replace(" ", "_").replace("/", "-")
        for inst in gloss_entry["instances"]:
            if split_filter != "all" and inst.get("split") != split_filter:
                continue
            url = inst.get("url", "")
            if not is_direct_mp4(url):
                continue
            video_id = str(inst["video_id"])
            out_path = OUT_DIR / gloss / f"{video_id}.mp4"
            tasks.append({
                "gloss": gloss,
                "video_id": video_id,
                "url": url,
                "out_path": out_path,
            })
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Download WLASL MP4 video clips")
    parser.add_argument("--glosses", type=int, default=2000,
                        help="Number of glosses to process (default: all)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel download threads (default: 8)")
    parser.add_argument("--split", default="all",
                        choices=["all", "train", "val", "test"],
                        help="Dataset split to download (default: all)")
    args = parser.parse_args()

    print(f"Loading {JSON_PATH} …")
    with open(JSON_PATH) as f:
        data = json.load(f)

    tasks = build_task_list(data, args.glosses, args.split)
    print(f"Found {len(tasks)} direct-MP4 instances "
          f"({args.glosses} glosses, split={args.split})")

    already = sum(1 for t in tasks if t["out_path"].exists()
                  and t["out_path"].stat().st_size > 1024)
    print(f"Already downloaded: {already}  |  To fetch: {len(tasks) - already}\n")

    ok = failed = skipped = 0
    failed_lines: list[str] = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, t["gloss"], t["video_id"],
                        t["url"], t["out_path"]): t
            for t in tasks
        }
        for i, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            success, msg = fut.result()
            if "already" in msg:
                skipped += 1
            elif success:
                ok += 1
            else:
                failed += 1
                failed_lines.append(f"{t['gloss']}\t{t['video_id']}\t{t['url']}\t{msg}")

            if i % 100 == 0 or i == len(tasks):
                elapsed = time.time() - start
                rate = (ok + failed) / max(elapsed, 1)
                print(f"  [{i}/{len(tasks)}]  ok={ok}  failed={failed}  "
                      f"skipped={skipped}  {rate:.1f} dl/s")

    print(f"\nDone.  Downloaded={ok}  Failed={failed}  Skipped={skipped}")

    if failed_lines:
        FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(FAILED_LOG, "w") as f:
            f.write("gloss\tvideo_id\turl\terror\n")
            f.write("\n".join(failed_lines))
        print(f"Failed URLs logged to {FAILED_LOG}")


if __name__ == "__main__":
    main()
