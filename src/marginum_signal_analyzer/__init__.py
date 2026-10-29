import argparse
import sys
from pathlib import Path

import numpy as np

from .analysis import (
    find_busy_window,
    fit_peak_counts,
    process_chunks,
    process_file,
    sample_metadata,
)
from .processing import (
    detect_peaks,
    estimate_baseline,
    find_signal_peaks,
    remove_baseline,
    smooth,
)
from .signal_io import downsample, read_signal, read_signal_chunks, read_signal_window

__all__ = [
    "detect_peaks",
    "downsample",
    "estimate_baseline",
    "find_busy_window",
    "find_signal_peaks",
    "fit_peak_counts",
    "process_chunks",
    "process_file",
    "read_signal",
    "read_signal_chunks",
    "read_signal_window",
    "remove_baseline",
    "sample_metadata",
    "smooth",
]

def main() -> None:
    if "--gui" in sys.argv:
        sys.argv.remove("--gui")
        from .gui import main as gui_main

        gui_main()
        return

    parser = argparse.ArgumentParser(description="Detect peaks in a signal CSV.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("peaks.txt"))
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--column", default="1")
    parser.add_argument("--overlap", type=int, default=200_000)
    parser.add_argument("--height", type=float, default=60)
    parser.add_argument("--distance", type=int, default=25_000)
    args = parser.parse_args()
    column: int | str = int(args.column) if args.column.isdigit() else args.column
    peaks = process_file(
        args.csv_path,
        chunk_size=args.chunk_size,
        column=column,
        overlap=args.overlap,
        height=args.height,
        distance=args.distance,
    )
    np.savetxt(args.output, peaks, fmt="%d")
    print(f"Detected {len(peaks)} peaks")
    print(f"Saved peak positions to {args.output}")
