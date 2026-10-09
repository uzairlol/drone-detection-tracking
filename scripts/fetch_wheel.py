"""Resumable wheel fetch.

pip downloads a 2.6 GB wheel in one request and, when that connection hangs,
it sits there indefinitely rather than retrying. Multi-GB CUDA wheels are
prone to exactly that. This fetches with HTTP Range resume and retries, so a
stalled connection costs seconds instead of the whole transfer.
"""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path


def fetch(url: str, dest: Path, attempts: int = 200) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = 0
    for attempt in range(1, attempts + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if expected and have >= expected:
            return have
        request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if not expected:
                    expected = int(response.headers.get("Content-Range", "0-0/0").split("/")[-1])
                    if not expected:
                        expected = int(response.headers.get("Content-Length", 0)) + have
                mode = "ab" if have and response.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                started, written = time.monotonic(), 0
                with dest.open(mode) as handle:
                    while True:
                        chunk = response.read(1 << 18)
                        if not chunk:
                            break
                        handle.write(chunk)
                        written += len(chunk)
                        if written % (64 << 20) < (1 << 18):
                            speed = written / max(time.monotonic() - started, 1e-6) / 1e6
                            print(
                                f"  {have + written:>12,} / {expected:,} "
                                f"({100 * (have + written) / expected:5.1f}%) {speed:5.2f} MB/s",
                                flush=True,
                            )
                            have += written
                            written = 0
            print(f"  complete: {dest.stat().st_size:,} bytes", flush=True)
            return dest.stat().st_size
        except Exception as exc:
            print(f"  attempt {attempt} interrupted: {exc}", flush=True)
            time.sleep(min(5 * attempt, 30))
    raise SystemExit(f"giving up after {attempts} attempts")


if __name__ == "__main__":
    size = fetch(sys.argv[1], Path(sys.argv[2]))
    print(f"OK {size:,}")
