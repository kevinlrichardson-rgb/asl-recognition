"""
msasl_download.py — Download MS-ASL video clips from YouTube via yt-dlp + ffmpeg.

Reads data/msasl/MSASL_{train,val,test}.json, downloads each YouTube video once
to a cache, clips the segment with ffmpeg, and saves:

    data/msasl_videos/<clean_text>/<signer_id>_<yt_id>.mp4

These can then be fed to msasl_extract.py to produce landmarks in the same
data/wlasl_landmarks/ tree used by the training pipeline.

Requires:
    pip install yt-dlp
    sudo apt-get install ffmpeg

Usage:
    python src/msasl_download.py [--workers W] [--splits train val test]
"""

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT      = Path(__file__).resolve().parent.parent
JSON_DIR  = ROOT / "data" / "msasl"
VIDEO_DIR = ROOT / "data" / "msasl_videos"
CACHE_DIR = ROOT / "data" / "msasl_yt_cache"
LOG_PATH  = VIDEO_DIR / "failed.txt"

ALL_SPLITS = ["train", "val", "test"]


def yt_video_id(url: str) -> str:
    """Extract YouTube video ID from a watch URL."""
    qs = parse_qs(urlparse(url).query)
    ids = qs.get("v", [])
    return ids[0] if ids else hashlib.md5(url.encode()).hexdigest()[:10]


def cache_path_for(url: str) -> Path:
    return CACHE_DIR / f"{yt_video_id(url)}.mp4"


def download_full_video(url: str, dst: Path) -> tuple[bool, str]:
    if dst.exists() and dst.stat().st_size > 1024:
        return True, "cached"
    tmp = dst.with_name(dst.stem + ".tmp.mp4")
    cmd = [
        "yt-dlp", url,
        "--format", "bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", str(tmp),
        "--no-playlist",
        "--quiet", "--no-warnings",
        "--socket-timeout", "30",
        "--retries", "3",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 1024:
            for f in dst.parent.glob(dst.stem + ".tmp*"):
                f.unlink(missing_ok=True)
            err = (r.stderr.strip().splitlines() or ["unknown"])[-1]
            return False, err
        tmp.rename(dst)
        return True, f"OK ({dst.stat().st_size // 1024} KB)"
    except subprocess.TimeoutExpired:
        if tmp.exists():
            tmp.unlink()
        return False, "timeout"
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        return False, str(exc)


def clip_video(src: Path, dst: Path, start: float, end: float) -> tuple[bool, str]:
    if dst.exists() and dst.stat().st_size > 1024:
        return True, "already exists"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    duration = max(0.1, end - start)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "aac",
        "-loglevel", "error",
        str(tmp),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 1024:
            if tmp.exists():
                tmp.unlink()
            err = (r.stderr.strip().splitlines() or ["ffmpeg error"])[-1]
            return False, err
        tmp.rename(dst)
        return True, f"OK ({dst.stat().st_size // 1024} KB)"
    except subprocess.TimeoutExpired:
        if tmp.exists():
            tmp.unlink()
        return False, "ffmpeg timeout"
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        return False, str(exc)


def process_url(url: str, clips: list[dict]) -> dict:
    cache = cache_path_for(url)
    res = {"ok": 0, "skipped": 0, "failed": [], "yt_error": None}

    ok, msg = download_full_video(url, cache)
    if not ok:
        res["yt_error"] = msg
        res["failed"] = [(c["out_path"], msg) for c in clips]
        return res

    for clip in clips:
        c_ok, c_msg = clip_video(cache, clip["out_path"], clip["start"], clip["end"])
        if c_ok:
            res["skipped" if "already" in c_msg else "ok"] += 1
        else:
            res["failed"].append((clip["out_path"], c_msg))

    try:
        cache.unlink()
    except OSError:
        pass
    return res


def load_tasks(splits: list[str]) -> tuple[dict, int]:
    url_to_clips: dict[str, list[dict]] = defaultdict(list)
    pre_skipped = 0

    for split in splits:
        jpath = JSON_DIR / f"MSASL_{split}.json"
        if not jpath.exists():
            print(f"  [warn] not found: {jpath}")
            continue
        entries = json.loads(jpath.read_text())
        for entry in entries:
            url        = entry.get("url", "")
            start      = float(entry.get("start_time", 0))
            end        = float(entry.get("end_time",   0))
            gloss      = entry.get("clean_text", str(entry.get("label", "unknown")))
            gloss      = gloss.replace(" ", "_").replace("/", "-")
            signer_id  = entry.get("signer_id", 0)
            yt_id      = yt_video_id(url)
            out_path   = VIDEO_DIR / gloss / f"{signer_id}_{yt_id}.mp4"

            if out_path.exists() and out_path.stat().st_size > 1024:
                pre_skipped += 1
                continue
            if not url or end <= start:
                continue
            url_to_clips[url].append({
                "out_path": out_path,
                "start":    start,
                "end":      end,
                "gloss":    gloss,
            })

    return url_to_clips, pre_skipped


def main():
    parser = argparse.ArgumentParser(description="Download MS-ASL clips from YouTube")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--splits", nargs="+", default=ALL_SPLITS,
                        choices=ALL_SPLITS, metavar="SPLIT")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading MS-ASL JSON from {JSON_DIR} …")
    url_to_clips, pre_skipped = load_tasks(args.splits)
    total_clips = sum(len(v) for v in url_to_clips.values())
    print(f"Unique YouTube URLs : {len(url_to_clips)}")
    print(f"Clips to extract    : {total_clips}")
    print(f"Already present     : {pre_skipped}")
    print(f"Workers             : {args.workers}\n")

    if not url_to_clips:
        print("Nothing to download.")
        return

    ok = skipped = failed = 0
    failed_lines: list[str] = []
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_url, url, clips): url
                   for url, clips in url_to_clips.items()}
        for fut in as_completed(futures):
            res = fut.result()
            done  += 1
            ok      += res["ok"]
            skipped += res["skipped"]
            failed  += len(res["failed"])

            if res["yt_error"]:
                failed_lines.append(f"YT_FAIL\t{futures[fut]}\t{res['yt_error']}")
            for path, msg in res["failed"]:
                failed_lines.append(f"CLIP_FAIL\t{path}\t{msg}")

            if done % 100 == 0 or done == len(url_to_clips):
                print(f"  [{done}/{len(url_to_clips)} URLs]  "
                      f"ok={ok}  skipped={skipped}  failed={failed}")

    print(f"\nDone.  Saved={ok}  Skipped={skipped}  Failed={failed}")
    if failed_lines:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(failed_lines) + "\n")
        print(f"Failure log → {LOG_PATH}")


if __name__ == "__main__":
    main()
