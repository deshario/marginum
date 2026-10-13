import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from marginum_signal_analyzer.binary_cache import (
    DEFAULT_CACHE_PATH,
    DEFAULT_CSV_PATH,
    convert_csv_to_binary,
    open_binary_signal,
    read_binary_window,
)
from marginum_signal_analyzer.plot_signal import downsample_for_plot

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 50_000
WINDOW_SIZE = SAMPLE_RATE
STEP_SIZE = SAMPLE_RATE
PLOT_POINTS = 5_000

def main() -> None:
    parser = argparse.ArgumentParser(description="Browse signal windows.")
    parser.add_argument("--window", type=int, default=0)
    args = parser.parse_args()
    if args.window < 0:
        raise ValueError("window must be zero or greater")

    if not DEFAULT_CACHE_PATH.exists():
        convert_csv_to_binary(DEFAULT_CSV_PATH, DEFAULT_CACHE_PATH)
    signal_file = open_binary_signal(DEFAULT_CACHE_PATH)
    maximum_window = (len(signal_file) - WINDOW_SIZE) // STEP_SIZE
    if args.window > maximum_window:
        raise ValueError(f"window must be between 0 and {maximum_window}")

    state = {"window": args.window}
    figure, axes = plt.subplots(figsize=(12, 6))
    figure.subplots_adjust(bottom=0.2)
    previous_axis = figure.add_axes((0.30, 0.05, 0.15, 0.075))
    next_axis = figure.add_axes((0.55, 0.05, 0.15, 0.075))
    previous_button = Button(previous_axis, "Previous")
    next_button = Button(next_axis, "Next")

    def update_plot() -> None:
        start_sample = state["window"] * STEP_SIZE
        end_sample = start_sample + WINDOW_SIZE
        window = read_binary_window(signal_file, start_sample, end_sample)
        time = np.arange(len(window)) / SAMPLE_RATE + start_sample / SAMPLE_RATE
        plot_time, plot_adc1 = downsample_for_plot(
            time,
            window[:, 0],
            PLOT_POINTS,
        )
        _, plot_adc2 = downsample_for_plot(time, window[:, 1], PLOT_POINTS)
        axes.clear()
        axes.plot(plot_time, plot_adc1, label="adc1", linewidth=0.8)
        axes.plot(plot_time, plot_adc2, label="adc2", linewidth=0.8)
        axes.set_xlabel("Time (seconds)")
        axes.set_ylabel("Sensor reading")
        axes.set_title(
            f"Window {state['window']}: "
            f"{start_sample / SAMPLE_RATE:.1f}–{end_sample / SAMPLE_RATE:.1f} seconds"
        )
        axes.legend()
        axes.grid(True, alpha=0.3)
        previous_button.set_active(state["window"] > 0)
        next_button.set_active(state["window"] < maximum_window)
        figure.canvas.draw_idle()

    def show_previous(_: object) -> None:
        if state["window"] > 0:
            state["window"] -= 1
            update_plot()

    def show_next(_: object) -> None:
        if state["window"] < maximum_window:
            state["window"] += 1
            update_plot()

    previous_button.on_clicked(show_previous)
    next_button.on_clicked(show_next)
    update_plot()
    plt.show()

if __name__ == "__main__":
    main()
