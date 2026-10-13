import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from marginum_signal_analyzer.binary_cache import (
    DEFAULT_CACHE_PATH,
    DEFAULT_CSV_PATH,
    convert_csv_to_binary,
    open_binary_signal,
    read_binary_window,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 50_000
WINDOW_SIZE = SAMPLE_RATE
STEP_SIZE = SAMPLE_RATE
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=0, help="Window number")
    args = parser.parse_args()
    if args.window < 0:
        raise ValueError("window must be zero or greater")

    if not DEFAULT_CACHE_PATH.exists():
        convert_csv_to_binary(DEFAULT_CSV_PATH, DEFAULT_CACHE_PATH)
    signal_file = open_binary_signal(DEFAULT_CACHE_PATH)
    start_sample = args.window * STEP_SIZE
    end_sample = start_sample + WINDOW_SIZE
    window = read_binary_window(signal_file, start_sample, end_sample)
    start_time = start_sample / SAMPLE_RATE
    end_time = end_sample / SAMPLE_RATE
    output_path = PROJECT_ROOT / "output" / f"signal_window_{args.window}.png"

    signal = np.asarray(window)
    adc1 = signal[:, 0]
    adc2 = signal[:, 1]
    time = np.arange(len(window)) / SAMPLE_RATE + start_time

    plot_time, plot_adc1 = downsample_for_plot(time, adc1, PLOT_POINTS)
    _, plot_adc2 = downsample_for_plot(time, adc2, PLOT_POINTS)

    output_path.parent.mkdir(exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(plot_time, plot_adc1, label="adc1", linewidth=0.8)
    plt.plot(plot_time, plot_adc2, label="adc2", linewidth=0.8)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Sensor reading")
    plt.title(f"Signal from {start_time} to {end_time} seconds")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()

    print(f"Window: {args.window}")
    print(f"Time: {start_time}..{end_time} seconds")
    print(f"Original points: {len(window):,}")
    print(f"Displayed points per sensor: {len(plot_time):,}")
    print(f"Saved plot to: {output_path}")

if __name__ == "__main__":
    main()
