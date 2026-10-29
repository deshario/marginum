from __future__ import annotations

import numpy as np

def concatenate_windows(windows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate windows and return their sample-time axis at 50 kHz."""
    combined = np.concatenate([np.asarray(window) for window in windows])
    return np.arange(len(combined)) / 50_000, combined
