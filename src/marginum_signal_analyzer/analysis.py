from __future__ import annotations

from collections.abc import Iterable, Callable

import numpy as np

from .processing import find_signal_peaks

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
