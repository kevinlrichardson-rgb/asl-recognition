"""
wlasl_retry_download.py — Retry videos from failed_urls.txt with 2 workers.

Reads data/wlasl_videos/failed_urls.txt, attempts every URL again, and writes
a new failure log to data/wlasl_videos/failed_urls_retry.txt.

SSL certificate errors are retried with verification disabled.

Usage:
    python src/wlasl_retry_download.py
"""

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ROOT         = Path(__file__).resolve().parent.parent
VIDEO_DIR    = ROOT / "data" / "wlasl_videos"
FAILED_IN    = VIDEO_DIR / "failed_urls.txt"
FAILED_OUT   = VIDEO_DIR / "failed_urls_retry.txt"
TIMEOUT      = 25
CHUNK        = 1 << 16
WORKERS      = 2


def download_one(gloss: str, video_id: str, url: str) -> tuple[bool, str]:
    out_path = VIDEO_DIR / gloss / f"{video_id}.mp4"

    if out_path.exists() and out_path.stat().st_size > 1024:
        return True, "already exists"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    headers = {"User-Agent": "Mozilla/5.0"}

    # Try with SSL verification first; fall back to unverified for cert errors
    for verify in (True, False):
        try:
            resp = requests.get(url, stream=True, timeout=TIMEOUT,
                                headers=headers, verify=verify)
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK):
                    if chunk:
                        f.write(chunk)
            tmp.rename(out_path)
            note = " (SSL verify=False)" if not verify else ""
            return True, f"OK ({out_path.stat().st_size // 1024} KB){note}"
        except requests.exceptions.SSLError:
            if not verify:
                if tmp.exists():
                    tmp.unlink()
                return False, "SSL error (even with verify=False)"
            # retry without SSL verification
            continue
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            return False, str(exc)

    return False, "unknown error"


def load_tasks() -> list[dict]:
    if not FAILED_IN.exists():
        print(f"File not found: {FAILED_IN}")
        return []
    tasks = []
    with open(FAILED_IN, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tasks.append({
                "gloss":    row["gloss"],
                "video_id": row["video_id"],
                "url":      row["url"],
            })
    return tasks


def main():
    tasks = load_tasks()
    if not tasks:
        return

    print(f"Retrying {len(tasks)} failed URLs with {WORKERS} workers …\n")

    ok = failed = skipped = 0
    failed_rows: list[dict] = []

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(download_one, t["gloss"], t["video_id"], t["url"]): t
            for t in tasks
        }
        for i, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            success, msg = fut.result()

            if "already" in msg:
                skipped += 1
            elif success:
                ok += 1
                print(f"  ✓ [{i}/{len(tasks)}] {t['gloss']}/{t['video_id']}  {msg}")
            else:
                failed += 1
                failed_rows.append({
                    "gloss":    t["gloss"],
                    "video_id": t["video_id"],
                    "url":      t["url"],
                    "error":    msg,
                })

            if i % 500 == 0:
                print(f"  --- [{i}/{len(tasks)}] ok={ok} failed={failed} skipped={skipped}")

    print(f"\nDone.  Downloaded={ok}  Failed={failed}  Skipped={skipped}")

    # Write new failure log
    with open(FAILED_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["gloss", "video_id", "url", "error"],
                                delimiter="\t")
        writer.writeheader()
        writer.writerows(failed_rows)

    print(f"New failure log ({failed} entries) → {FAILED_OUT}")


if __name__ == "__main__":
    main()
