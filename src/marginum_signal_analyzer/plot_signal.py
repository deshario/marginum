from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from marginum_signal_analyzer.chunk_reader import read_window

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "IDP_signal_example_[sami_dmitry_september2023].csv"
)
OUTPUT_PATH = PROJECT_ROOT / "output" / "signal_window.png"

SAMPLE_RATE = 50_000
START_TIME = 10
END_TIME = 11
PLOT_POINTS = 5_000

def downsample_for_plot(
    time: np.ndarray,
    values: np.ndarray,
    target_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce display points while preserving each bin's minimum and maximum."""
    if target_points <= 0:
        raise ValueError("target_points must be greater than zero")
    if len(values) <= target_points:
        return time, values

    bin_count = max(target_points // 2, 1)
    edges = np.linspace(0, len(values), bin_count + 1, dtype=int)
    output_time: list[float] = []
    output_values: list[float] = []

    for start, end in zip(edges[:-1], edges[1:], strict=True):
        if start == end:
            continue

        bin_values = values[start:end]
        minimum_index = start + int(np.argmin(bin_values))
        maximum_index = start + int(np.argmax(bin_values))

        indexes = sorted((minimum_index, maximum_index))
        output_time.extend(time[index] for index in indexes)
        output_values.extend(values[index] for index in indexes)

    return np.asarray(output_time), np.asarray(output_values)

def main() -> None:
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    start_sample = START_TIME * SAMPLE_RATE
    end_sample = END_TIME * SAMPLE_RATE
    window = read_window(CSV_PATH, start_sample, end_sample)

    signal = np.array(window)
    adc1 = signal[:, 0]
    adc2 = signal[:, 1]
    time = np.arange(len(window)) / SAMPLE_RATE + START_TIME

    plot_time, plot_adc1 = downsample_for_plot(time, adc1, PLOT_POINTS)
    _, plot_adc2 = downsample_for_plot(time, adc2, PLOT_POINTS)

    plt.figure(figsize=(12, 5))
    plt.plot(plot_time, plot_adc1, label="adc1", linewidth=0.8)
    plt.plot(plot_time, plot_adc2, label="adc2", linewidth=0.8)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Sensor reading")
    plt.title(f"Signal from {START_TIME} to {END_TIME} seconds")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.show()

    print(f"Original points: {len(window):,}")
    print(f"Displayed points per sensor: {len(plot_time):,}")
    print(f"Saved plot to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
