from __future__ import annotations

import re
from collections.abc import Iterable
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .processing import find_signal_peaks
from .signal_io import read_signal, read_signal_chunks

def process_chunks(
    chunks: Iterable[np.ndarray],
    *,
    overlap: int = 200_000,
    baseline_window: int = 100_001,
    filter_size: int = 301,
    height: float = 60,
    distance: int = 25_000,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Detect globally indexed peaks from chunks while avoiding overlap duplicates."""
    if overlap < 0:
        raise ValueError("overlap must not be negative")

    all_peaks: list[int] = []
    row_offset = 0
    leftover = np.empty(0, dtype=float)
    for chunk_number, current in enumerate(chunks, start=1):
        current = np.asarray(current, dtype=float)
        if current.ndim != 1:
            raise ValueError("each chunk must be one-dimensional")
        combined = np.concatenate((leftover, current))
        _, _, _, peaks = find_signal_peaks(
            combined,
            baseline_window=baseline_window,
            filter_size=filter_size,
            height=height,
            distance=distance,
        )
        for peak in peaks:
            if peak >= len(leftover):
                all_peaks.append(row_offset - len(leftover) + int(peak))
        row_offset += len(current)
        leftover = combined[-overlap:] if overlap else np.empty(0, dtype=float)
        if progress is not None:
            progress(chunk_number, row_offset)
    return np.asarray(all_peaks, dtype=int)

def process_file(
    path: str | Path,
    *,
    chunk_size: int = 1_000_000,
    column: int | str = 1,
    **processing_kwargs,
) -> np.ndarray:
    chunks = read_signal_chunks(path, chunk_size=chunk_size, column=column)
    return process_chunks(chunks, **processing_kwargs)

def find_busy_window(
    path: str | Path,
    *,
    window_samples: int = 3_000_000,
    **processing_kwargs,
) -> tuple[np.ndarray, int]:
    """Return the window containing the most detected peaks."""
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")
    raw = read_signal(path)
    _, _, _, peaks = find_signal_peaks(raw, **processing_kwargs)
    if len(raw) <= window_samples:
        return raw.copy(), int(len(peaks))
    if len(peaks) == 0:
        return raw[:window_samples], 0

    best_start = 0
    best_count = -1
    for peak in peaks:
        start = max(0, min(int(peak) - window_samples // 2, len(raw) - window_samples))
        count = int(np.count_nonzero((peaks >= start) & (peaks < start + window_samples)))
        if count > best_count:
            best_start, best_count = start, count
    return raw[best_start : best_start + window_samples], best_count

def sample_metadata(filename: str) -> tuple[int, bool]:
    """Extract tumour size and resuscitation marker from a filename."""
    match = re.search(r"([123])gr", filename.lower())
    return (int(match.group(1)) if match else 0, "resuc" in filename.lower())

def fit_peak_counts(sizes: np.ndarray, peaks: np.ndarray) -> tuple[float, float, float]:
    """Fit ``peaks = slope * size + intercept`` and return R-squared."""
    sizes = np.asarray(sizes, dtype=float)
    peaks = np.asarray(peaks, dtype=float)
    if sizes.size != peaks.size or sizes.size < 2:
        raise ValueError("sizes and peaks must contain at least two matching values")
    slope, intercept = np.polyfit(sizes, peaks, 1)
    correlation = np.corrcoef(sizes, peaks)[0, 1]
    return float(slope), float(intercept), float(correlation**2)
