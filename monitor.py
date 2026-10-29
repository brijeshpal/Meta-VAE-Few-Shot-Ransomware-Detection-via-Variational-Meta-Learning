"""
monitor.py
----------
Watches a folder and scans every NEW file the moment it appears -- this is the
"detect malware as it enters the system" behaviour. Quarantines (moves) files
the model flags as malware so they cannot be executed from the watched folder.

Usage:
    python monitor.py /path/to/watch [/path/to/quarantine]

Uses lightweight polling (no extra dependencies). On a real endpoint you would
hook this to OS file-system events (inotify on Linux, ReadDirectoryChangesW on
Windows) or wire the Detector into an on-access AV driver.
"""

from __future__ import annotations
import os
import shutil
import sys
import time

from detector import Detector


def monitor(watch_dir: str, quarantine_dir: str, interval: float = 1.0):
    os.makedirs(watch_dir, exist_ok=True)
    os.makedirs(quarantine_dir, exist_ok=True)
    det = Detector()
    seen: set[str] = set(os.listdir(watch_dir))  # ignore pre-existing files

    print(f"Watching: {watch_dir}")
    print(f"Quarantine: {quarantine_dir}")
    print("Drop files into the watched folder. Ctrl+C to stop.\n")

    try:
        while True:
            current = set(os.listdir(watch_dir))
            new_files = current - seen
            for name in sorted(new_files):
                path = os.path.join(watch_dir, name)
                if not os.path.isfile(path):
                    continue
                _wait_until_stable(path)  # let the file finish copying
                _handle(det, path, quarantine_dir)
            seen = current
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")


def _wait_until_stable(path: str, checks: int = 3, gap: float = 0.3):
    """Wait until file size stops changing (avoids scanning a partial copy)."""
    last = -1
    stable = 0
    while stable < checks:
        try:
            size = os.path.getsize(path)
        except OSError:
            return
        stable = stable + 1 if size == last else 0
        last = size
        time.sleep(gap)


def _handle(det: Detector, path: str, quarantine_dir: str):
    try:
        result = det.scan(path)
    except Exception as e:
        print(f"[?] {os.path.basename(path)}: scan error {e}")
        return

    p = result["probability"]
    if result["verdict"] == "MALWARE":
        dest = os.path.join(quarantine_dir, os.path.basename(path))
        try:
            shutil.move(path, dest)
            moved = f" -> quarantined to {dest}"
        except Exception as e:
            moved = f" (quarantine failed: {e})"
        print(f"[!] MALWARE  p={p:.3f}  {os.path.basename(path)}{moved}")
        for r in result["reasons"]:
            print(f"        - {r}")
    else:
        print(f"[ ] clean    p={p:.3f}  {os.path.basename(path)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python monitor.py <watch_dir> [<quarantine_dir>]")
        sys.exit(1)
    watch = sys.argv[1]
    quarantine = sys.argv[2] if len(sys.argv) > 2 else os.path.join(watch, "_quarantine")
    monitor(watch, quarantine)
