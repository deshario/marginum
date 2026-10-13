from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter

from marginum_signal_analyzer.chunk_reader import read_window

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "IDP_signal_example_[sami_dmitry_september2023].csv"
)
OUTPUT_PATH = PROJECT_ROOT / "output" / "detected_peaks.png"

SAMPLE_RATE = 50_000
START_TIME = 0
END_TIME = 30
BASELINE_WINDOW = 5_001
BASELINE_POLYNOMIAL_ORDER = 2
SMOOTHING_WINDOW = 101
SMOOTHING_POLYNOMIAL_ORDER = 2
PEAK_PROMINENCE = 5
PEAK_DISTANCE_SECONDS = 0.5

def estimate_baseline(signal: np.ndarray) -> np.ndarray:
    """Estimate the slow-changing baseline of a signal."""
    if len(signal) < BASELINE_WINDOW:
        raise ValueError(
            f"Signal must contain at least {BASELINE_WINDOW} samples"
        )

    return savgol_filter(
        signal,
        window_length=BASELINE_WINDOW,
        polyorder=BASELINE_POLYNOMIAL_ORDER,
    )

def smooth_signal(signal: np.ndarray) -> np.ndarray:
    """Reduce high-frequency noise in the corrected signal."""
    return savgol_filter(
        signal,
        window_length=SMOOTHING_WINDOW,
        polyorder=SMOOTHING_POLYNOMIAL_ORDER,
    )

def detect_peaks(signal: np.ndarray) -> np.ndarray:
    """Find prominent peaks separated by a minimum time distance."""
    peaks, _ = find_peaks(
        signal,
        prominence=PEAK_PROMINENCE,
        distance=int(PEAK_DISTANCE_SECONDS * SAMPLE_RATE),
    )
    return peaks

def main() -> None:
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    start_sample = START_TIME * SAMPLE_RATE
    end_sample = END_TIME * SAMPLE_RATE
    window = read_window(CSV_PATH, start_sample, end_sample)

    signal = np.asarray(window, dtype=np.float64)
    adc2 = signal[:, 1]
    time = np.arange(len(adc2)) / SAMPLE_RATE + START_TIME
    baseline = estimate_baseline(adc2)
    corrected_signal = adc2 - baseline
    smoothed_signal = smooth_signal(corrected_signal)
    peaks = detect_peaks(smoothed_signal)

    figure, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    axes[0].plot(time, adc2, label="adc2", linewidth=0.6, alpha=0.7)
    axes[0].plot(time, baseline, label="estimated baseline", linewidth=2)
    axes[0].set_ylabel("Sensor reading")
    axes[0].set_title(f"ADC2 baseline from {START_TIME} to {END_TIME} seconds")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(time, corrected_signal, label="corrected adc2", linewidth=0.6)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Corrected value")
    axes[1].set_title("Baseline-corrected ADC2")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(time, smoothed_signal, label="smoothed corrected adc2", linewidth=0.8)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Smoothed value")
    axes[2].set_title("Smoothed baseline-corrected ADC2")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[3].plot(time, smoothed_signal, label="smoothed corrected adc2", linewidth=0.8)
    axes[3].scatter(
        time[peaks],
        smoothed_signal[peaks],
        color="red",
        label="detected peaks",
        zorder=3,
    )
    axes[3].axhline(0, color="black", linewidth=0.8)
    axes[3].set_xlabel("Time (seconds)")
    axes[3].set_ylabel("Signal value")
    axes[3].set_title("Detected ADC2 peaks")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.show()

    print(f"Detected peaks: {len(peaks)}")
    print(f"Saved peak plot to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
