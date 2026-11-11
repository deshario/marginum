from __future__ import annotations

from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .config import LIVE_MAX_POINTS, SAMPLE_RATE_HZ


class PlotView:
    """The matplotlib figure and every drawing routine for it."""

    def __init__(self, parent: ttk.Frame) -> None:
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.live_line = None
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def clear(self, message: str | None = None) -> None:
        self.live_line = None
        self.figure.clear()
        if message:
            self.figure.text(0.5, 0.5, message, ha="center", va="center")
        self.canvas.draw_idle()

    def draw_empty(self) -> None:
        self.live_line = None
        axis = self.figure.add_subplot(111)
        axis.text(
            0.5,
            0.5,
            "Choose a signal CSV file",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        self.canvas.draw_idle()

    def _decimate_for_display(self, signal: np.ndarray) -> np.ndarray:
        """Shrink to about LIVE_MAX_POINTS using per-bucket min/max, so narrow
        peaks stay visible while the redraw cost stays constant."""
        buckets = LIVE_MAX_POINTS // 2
        size = len(signal) // buckets
        if size < 2:
            return signal
        blocks = signal[: size * buckets].reshape(buckets, size)
        return np.column_stack((blocks.min(axis=1), blocks.max(axis=1))).ravel()

    def draw_live(self, signal: np.ndarray, rows: int) -> None:
        signal = self._decimate_for_display(signal)
        # The points are spread evenly over everything scanned so far.
        live_time = np.linspace(0, rows / SAMPLE_RATE_HZ, len(signal))
        if self.live_line is None:
            self.figure.clear()
            axis = self.figure.add_subplot(111)
            (self.live_line,) = axis.plot(
                live_time, signal, color="#2563eb", linewidth=0.8
            )
            axis.set_xlabel("Time (s)")
            axis.set_ylabel("ADC counts (adc2)")
            axis.grid(alpha=0.25)
            axis.set_title(f"Live signal overview ({rows:,} samples scanned)")
            self.figure.tight_layout()  # after the title so it reserves room for it
        else:
            self.live_line.set_data(live_time, signal)
            axis = self.live_line.axes
            axis.relim()
            axis.autoscale_view()
        axis.set_title(f"Live signal overview ({rows:,} samples scanned)")
        self.canvas.draw_idle()

    def draw_overview(
        self,
        sample_indices: np.ndarray,
        sample_values: np.ndarray,
        total_samples: int,
        full_peaks: np.ndarray,
    ) -> None:
        self.live_line = None
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.plot(
            sample_indices / SAMPLE_RATE_HZ,
            sample_values,
            color="#555555",
            linewidth=0.5,
            label="Signal (downsampled)",
        )
        peak_positions = full_peaks[full_peaks < total_samples]
        peak_indices = np.searchsorted(sample_indices, peak_positions)
        valid = peak_indices < len(sample_values)
        if np.any(valid):
            axis.scatter(
                peak_positions[valid] / SAMPLE_RATE_HZ,
                sample_values[peak_indices[valid]],
                color="red",
                s=15,
                marker="x",
                label=f"{len(peak_positions)} peaks",
            )
        axis.set_title(f"Signal overview - {len(full_peaks):,} peaks detected")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("ADC counts (adc2)")
        axis.legend(loc="upper right")
        axis.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def draw_signal(
        self,
        raw: np.ndarray,
        baseline: np.ndarray,
        flat: np.ndarray,
        clean: np.ndarray,
        peaks: np.ndarray,
        offset: int = 0,
    ) -> None:
        self.live_line = None
        self.figure.clear()
        x = (np.arange(len(raw)) + offset) / SAMPLE_RATE_HZ
        local_peaks = np.asarray(peaks, dtype=int)
        peak_x = (local_peaks + offset) / SAMPLE_RATE_HZ
        axis = self.figure.add_subplot(111)
        axis.plot(
            x,
            raw,
            color="#555555",
            alpha=0.55,
            linewidth=0.8,
            label="Raw signal",
        )
        axis.plot(x, clean, color="blue", linewidth=0.9, label="Smoothed signal")
        if len(peaks):
            axis.plot(
                peak_x,
                clean[local_peaks],
                "x",
                color="red",
                markersize=10,
                markeredgewidth=1.5,
                zorder=4,
                label=f"Detected peaks ({len(peaks)})",
            )
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("ADC counts (adc2)")
        axis.set_title(f"Signal - {len(peaks)} peaks found")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")
        self.figure.tight_layout()
        self.canvas.draw_idle()
