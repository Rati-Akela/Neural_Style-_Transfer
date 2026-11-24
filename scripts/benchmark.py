"""Measure latency and peak memory, one fresh process per size:  python scripts/benchmark.py"""

import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adain.engine import StyleTransfer  # noqa: E402


def run_one(size: int) -> None:
    eng = StyleTransfer()
    cat = Image.open(ROOT / "examples/content/cat.jpg")
    style = Image.open(ROOT / "examples/styles/la_muse.jpg")
    eng.stylize(cat, style, size=64)  # warm-up
    times = []
    for _ in range(5):
        eng._cache.clear()
        t = time.perf_counter()
        eng.stylize(cat, style, size=size)
        times.append(time.perf_counter() - t)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB -> MB on Linux
    print(f"| {size} px | {statistics.median(times):.2f} s | {rss:.0f} MB |")


def main() -> None:
    if len(sys.argv) > 1:
        return run_one(int(sys.argv[1]))
    print(f"CPU: {os.cpu_count()} cores, {platform.machine()}, Python {platform.python_version()}\n")
    print("| Output size | Latency (median of 5) | Peak memory |\n| --- | --- | --- |")
    for size in (256, 384, 512, 640, 768):
        subprocess.run([sys.executable, __file__, str(size)], check=True)


if __name__ == "__main__":
    main()
