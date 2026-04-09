"""
wlasl_youtube_download.py — Download WLASL clips from YouTube via yt-dlp + ffmpeg.

For each YouTube instance in WLASL_v0.3.json:
  1. Downloads the full YouTube video once to a local cache (data/wlasl_yt_cache/)
  2. Clips frame_start → frame_end with ffmpeg
  3. Saves to data/wlasl_videos/<gloss>/<video_id>.mp4

Requires:
    pip install yt-dlp
    sudo apt-get install ffmpeg

Usage:
    python src/wlasl_youtube_download.py [--workers W] [--glosses N] [--keep-cache]

Options:
    --workers     Parallel YouTube download workers (default: 3)
    --glosses     Only process first N glosses (default: all)
    --keep-cache  Keep full YouTube downloads in data/wlasl_yt_cache/ after clipping
                  (useful to avoid re-downloading on a second run)
"""

import argparse
import json
import subprocess
import hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "data" / "WLASL_v0_3_json" / "WLASL_v0.3.json"
VIDEO_DIR = ROOT / "data" / "wlasl_videos"
CACHE_DIR = ROOT / "data" / "wlasl_yt_cache"
LOG_PATH  = VIDEO_DIR / "youtube_failed.txt"


def is_youtube(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def url_cache_path(url: str) -> Path:
    """Deterministic cache filename based on URL hash."""
    h = hashlib.md5(url.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{h}.mp4"


def download_full_video(url: str, cache_path: Path) -> tuple[bool, str]:
    """Download a YouTube video to cache_path. Returns (success, message)."""
    if cache_path.exists() and cache_path.stat().st_size > 1024:
        return True, "cached"

    tmp = cache_path.with_name(cache_path.stem + ".tmp.mp4")
    cmd = [
        "yt-dlp",
        url,
        "--format", "bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", str(tmp),
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--socket-timeout", "30",
        "--retries", "3",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            for f in CACHE_DIR.glob(cache_path.stem + ".tmp*"):
                f.unlink(missing_ok=True)
            err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
            return False, err
        if not tmp.exists() or tmp.stat().st_size < 1024:
            return False, "output file missing or empty"
        tmp.rename(cache_path)
        return True, f"OK ({cache_path.stat().st_size // 1024} KB)"
    except subprocess.TimeoutExpired:
        if tmp.exists():
            tmp.unlink()
        return False, "timeout"
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        return False, str(exc)


def clip_video(src: Path, out_path: Path,
               frame_start: int, frame_end: int, fps: float) -> tuple[bool, str]:
    """Extract a clip from src using ffmpeg. Returns (success, message)."""
    if out_path.exists() and out_path.stat().st_size > 1024:
        return True, "already exists"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.mp4")

    # Convert 1-indexed frames to timestamps
    start_sec = max(0.0, (frame_start - 1) / fps)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", str(src),
    ]
    if frame_end != -1:
        duration = (frame_end - frame_start + 1) / fps
        cmd += ["-t", f"{duration:.3f}"]

    cmd += [
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "aac",
        "-movflags", "+faststart",
        "-loglevel", "error",
        str(tmp),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 1024:
            if tmp.exists():
                tmp.unlink()
            err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "ffmpeg error"
            return False, err
        tmp.rename(out_path)
        return True, f"OK ({out_path.stat().st_size // 1024} KB)"
    except subprocess.TimeoutExpired:
        if tmp.exists():
            tmp.unlink()
        return False, "ffmpeg timeout"
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        return False, str(exc)


def process_url(url: str, clips: list[dict], keep_cache: bool = False) -> dict:
    """Download one YouTube URL then extract all clips from it."""
    cache_path = url_cache_path(url)
    results = {"ok": 0, "skipped": 0, "failed": [], "yt_error": None}

    ok, msg = download_full_video(url, cache_path)
    if not ok:
        results["yt_error"] = msg
        results["failed"] = [(c["gloss"], c["video_id"], msg) for c in clips]
        return results

    for clip in clips:
        out_path = VIDEO_DIR / clip["gloss"] / f"{clip['video_id']}.mp4"
        if out_path.exists() and out_path.stat().st_size > 1024:
            results["skipped"] += 1
            continue
        clip_ok, clip_msg = clip_video(
            cache_path, out_path,
            clip["frame_start"], clip["frame_end"], clip["fps"]
        )
        if clip_ok:
            if "already" in clip_msg:
                results["skipped"] += 1
            else:
                results["ok"] += 1
        else:
            results["failed"].append((clip["gloss"], clip["video_id"], clip_msg))

    # Clean up cache after all clips extracted to save disk space
    if not keep_cache:
        try:
            cache_path.unlink()
        except OSError:
            pass

    return results


def load_tasks(max_glosses: int) -> dict[str, list[dict]]:
    """Parse JSON and group YouTube instances by URL. Skip already-downloaded clips."""
    with open(JSON_PATH) as f:
        data = json.load(f)

    url_to_clips: dict[str, list[dict]] = defaultdict(list)
    skipped = 0

    for entry in data[:max_glosses]:
        gloss = entry["gloss"].replace(" ", "_").replace("/", "-")
        for inst in entry["instances"]:
            url = inst.get("url", "")
            if not is_youtube(url):
                continue
            out_path = VIDEO_DIR / gloss / f"{inst['video_id']}.mp4"
            if out_path.exists() and out_path.stat().st_size > 1024:
                skipped += 1
                continue
            url_to_clips[url].append({
                "gloss":       gloss,
                "video_id":    str(inst["video_id"]),
                "fps":         float(inst.get("fps") or 25),
                "frame_start": int(inst.get("frame_start") or 1),
                "frame_end":   int(inst.get("frame_end") or -1),
            })

    return url_to_clips, skipped


def main():
    parser = argparse.ArgumentParser(description="Download WLASL YouTube clips")
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel YouTube downloads (default: 3)")
    parser.add_argument("--glosses", type=int, default=2000,
                        help="Max glosses to process (default: all)")
    parser.add_argument("--keep-cache", action="store_true",
                        help="Keep full YouTube videos after clipping (saves re-downloading on reruns)")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {JSON_PATH} …")
    url_to_clips, pre_skipped = load_tasks(args.glosses)
    total_clips = sum(len(v) for v in url_to_clips.values())
    print(f"YouTube URLs to fetch : {len(url_to_clips)}")
    print(f"Clips to extract      : {total_clips}")
    print(f"Already downloaded    : {pre_skipped}")
    print(f"Workers               : {args.workers}")
    print(f"Cache                 : {CACHE_DIR}  (keep={args.keep_cache})\n")

    if not url_to_clips:
        print("Nothing to do.")
        return

    ok = skipped = failed = 0
    failed_lines: list[str] = []
    done_urls = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_url, url, clips, args.keep_cache): url
            for url, clips in url_to_clips.items()
        }
        for fut in as_completed(futures):
            url = futures[fut]
            res = fut.result()
            done_urls += 1

            ok      += res["ok"]
            skipped += res["skipped"]
            failed  += len(res["failed"])

            if res["yt_error"]:
                failed_lines.append(f"YT_FAIL\t{url}\t{res['yt_error']}")
            for gloss, vid, msg in res["failed"]:
                failed_lines.append(f"CLIP_FAIL\t{gloss}/{vid}\t{url}\t{msg}")

            if done_urls % 50 == 0 or done_urls == len(url_to_clips):
                print(f"  [{done_urls}/{len(url_to_clips)} URLs]  "
                      f"clips ok={ok}  skipped={skipped}  failed={failed}")

    print(f"\nDone.  Clips saved={ok}  Skipped={skipped}  Failed={failed}")

    if failed_lines:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(failed_lines) + "\n")
        print(f"Failure log → {LOG_PATH}")


if __name__ == "__main__":
    main()
