from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from marginum_signal_analyzer.chunk_reader import read_window

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "IDP_signal_example_[sami_dmitry_september2023].csv"
)
OUTPUT_PATH = PROJECT_ROOT / "output" / "baseline_corrected.png"

SAMPLE_RATE = 50_000
START_TIME = 10
END_TIME = 11
BASELINE_WINDOW = 5_001
BASELINE_POLYNOMIAL_ORDER = 2

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

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
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
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.show()

    print(f"Saved baseline plot to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
