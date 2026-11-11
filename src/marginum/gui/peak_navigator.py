from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

import numpy as np

from ..processing import find_signal_peaks

if TYPE_CHECKING:
    from .app import SignalAnalyzerApp


class PeakNavigator:
    """Previous/Next/Overview buttons and loading the window around a peak."""

    def __init__(self, parent: ttk.Frame, app: SignalAnalyzerApp) -> None:
        self.app = app
        self.selected_index = -1
        self.busy = False
        self.position = tk.StringVar(value="No peaks selected")

        navigation = ttk.Frame(parent)
        navigation.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.previous_button = ttk.Button(
            navigation, text="Previous peak", command=lambda: self.navigate(-1)
        )
        self.previous_button.pack(side=tk.LEFT)
        self.next_button = ttk.Button(
            navigation, text="Next peak", command=lambda: self.navigate(1)
        )
        self.next_button.pack(side=tk.LEFT, padx=(6, 0))
        self.overview_button = ttk.Button(
            navigation, text="Overview", command=self.show_overview
        )
        self.overview_button.pack(side=tk.LEFT, padx=(6, 0))
        self.set_enabled(False)
        ttk.Label(navigation, textvariable=self.position).pack(
            side=tk.LEFT, padx=(12, 0)
        )

    def set_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.previous_button.configure(state=state)
        self.next_button.configure(state=state)
        self.overview_button.configure(state=state)

    def navigate(self, direction: int) -> None:
        app = self.app
        if not len(app.full_peaks) or self.busy:
            app.status.set("No detected peaks to navigate.")
            return
        self.selected_index = (self.selected_index + direction) % len(app.full_peaks)
        self.update_position()
        self.start_load()

    def start_load(self) -> None:
        """Load the window around the selected peak without blocking the UI."""
        self.busy = True
        self.previous_button.configure(state=tk.DISABLED)
        self.next_button.configure(state=tk.DISABLED)
        selected_index = self.selected_index
        threading.Thread(
            target=self._load_selected_peak,
            args=(selected_index,),
            daemon=True,
        ).start()

    def show_overview(self) -> None:
        app = self.app
        if not app.input_path.get():
            app.status.set("Choose a signal file first.")
            return
        if not len(app.full_peaks):
            app.status.set("Run analysis before showing the overview.")
            return
        if not len(app.overview_values):
            app.status.set("Overview data is unavailable. Run analysis again.")
            return
        app.plot.draw_overview(
            app.overview_indices,
            app.overview_values,
            app.overview_total_samples,
            app.full_peaks,
        )
        app.status.set("Full-file overview loaded.")

    def update_position(self) -> None:
        peaks = self.app.full_peaks
        sample = int(peaks[self.selected_index])
        self.position.set(
            f"Peak {self.selected_index + 1} of {len(peaks)}\n"
            f"Sample: {sample:,}"
        )

    def _load_selected_peak(self, selected_index: int) -> None:
        try:
            self._load_selected_peak_unchecked(selected_index)
        except Exception as error:  # always unlock navigation, whatever went wrong
            self.app.root.after(0, self._peak_load_failed, str(error))

    def _peak_load_failed(self, message: str) -> None:
        self.busy = False
        self.previous_button.configure(state=tk.NORMAL)
        self.next_button.configure(state=tk.NORMAL)
        self.app.status.set(f"Could not load peak: {message}")

    def _load_selected_peak_unchecked(self, selected_index: int) -> None:
        app = self.app
        peak = int(app.full_peaks[selected_index])
        settings = app.run_settings
        window = max(1, int(settings["preview_samples"]))
        start = max(0, peak - window // 2)
        raw = app._read_window(start, start + window)
        if raw.size == 0:
            raise ValueError("no samples in the selected window")
        baseline, flat, clean, _ = find_signal_peaks(
            raw,
            height=float(settings["height"]),
            distance=int(settings["distance"]),
        )
        visible_peaks = app.full_peaks[
            (app.full_peaks >= start) & (app.full_peaks < start + len(raw))
        ] - start
        app.root.after(
            0,
            self._focused_peak_loaded,
            selected_index,
            raw,
            baseline,
            flat,
            clean,
            visible_peaks,
            start,
        )

    def _focused_peak_loaded(
        self,
        selected_index: int,
        raw: np.ndarray,
        baseline: np.ndarray,
        flat: np.ndarray,
        clean: np.ndarray,
        peaks: np.ndarray,
        offset: int,
    ) -> None:
        self.busy = False
        self.previous_button.configure(state=tk.NORMAL)
        self.next_button.configure(state=tk.NORMAL)
        if selected_index == self.selected_index:
            self.app.plot.draw_signal(raw, baseline, flat, clean, peaks, offset)
