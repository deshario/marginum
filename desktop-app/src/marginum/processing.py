from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

def estimate_baseline(signal: np.ndarray, window: int = 100_001) -> np.ndarray:
    """Estimate slow baseline drift with a centered rolling median."""
    if window <= 0 or window % 2 == 0:
        raise ValueError("window must be a positive odd number")
    values = np.asarray(signal, dtype=float)
    if values.size == 0:
        return values.copy()
    return median_filter(values, size=window, mode="nearest")

def remove_baseline(signal: np.ndarray, baseline: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    if baseline is None:
        baseline = estimate_baseline(values)
    baseline_values = np.asarray(baseline, dtype=float)
    if baseline_values.shape != values.shape:
        raise ValueError("signal and baseline must have the same shape")
    return values - baseline_values

def smooth(signal: np.ndarray, size: int = 301) -> np.ndarray:
    if size <= 0 or size % 2 == 0:
        raise ValueError("size must be a positive odd number")
    return median_filter(np.asarray(signal, dtype=float), size=size, mode="nearest")

def detect_peaks(
    signal: np.ndarray,
    *,
    height: float = 60,
    distance: int = 25_000,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Detect peaks in an already baseline-corrected signal."""
    if distance <= 0:
        raise ValueError("distance must be positive")
    return find_peaks(np.asarray(signal, dtype=float), height=height, distance=distance)

def find_signal_peaks(
    raw_signal: np.ndarray,
    *,
    baseline_window: int = 100_001,
    filter_size: int = 301,
    height: float = 60,
    distance: int = 25_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return baseline, baseline-corrected signal, smoothed signal, and peaks."""
    baseline = estimate_baseline(raw_signal, window=baseline_window)
    flat = remove_baseline(raw_signal, baseline)
    clean = smooth(flat, size=filter_size)
    peaks, _ = detect_peaks(clean, height=height, distance=distance)
    return baseline, flat, clean, peaks
