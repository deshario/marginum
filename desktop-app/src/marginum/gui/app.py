from __future__ import annotations

import threading
import tkinter as tk
import time
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .chunk_cache import ChunkCache
from .config import SAMPLE_RATE_HZ
from .peak_navigator import PeakNavigator
from .plot_view import PlotView
from .settings_panel import SettingsPanel
from .worker import run_analysis

class SignalAnalyzerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Marginum Signal Analyzer")
        self.root.minsize(1000, 620)
        self._maximize_window()

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar(value="peaks.txt")
        self.chunk_size = tk.StringVar(value="1000000")
        self.overlap = tk.StringVar(value="200000")
        self.height = tk.StringVar(value="60")
        self.distance = tk.StringVar(value="0.5")  # seconds
        self.channel = tk.StringVar(value="adc2")
        self.preview_samples = tk.StringVar(value="200000")
        self.status = tk.StringVar(value="")
        self.progress_detail = tk.StringVar(value="")
        self.elapsed_time = tk.StringVar(value="")
        self.summary = tk.StringVar(value="")
        self.progress = tk.DoubleVar()
        self.last_progress_percent = 0
        self.processing_started_at: float | None = None
        self.full_peaks = np.empty(0, dtype=int)
        self.overview_indices = np.empty(0, dtype=int)
        self.overview_values = np.empty(0, dtype=float)
        self.overview_total_samples = 0
        self.cache = ChunkCache()
        self.cancel_event = threading.Event()
        self.running = False
        self.run_settings: dict[str, int | float | str | bool] = {}
        self.show_live = tk.BooleanVar(value=True)
        self.plot: PlotView
        self.navigator: PeakNavigator
        self.settings_panel: SettingsPanel
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self._build()

    def _maximize_window(self) -> None:
        try:
            width = self.root.winfo_screenwidth()
            height = self.root.winfo_screenheight()
            self.root.geometry(f"{width}x{height}+0+0")
        except tk.TclError:
            self.root.geometry("1280x760")

    def _build(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        controls = ttk.LabelFrame(self.root, text="Analysis controls", padding=12)
        controls.grid(
            row=0,
            column=0,
            rowspan=2,
            sticky="nsew",
            padx=(12, 6),
            pady=12,
        )
        controls.configure(width=250)
        controls.grid_propagate(False)
        controls.columnconfigure(0, weight=1)

        self.settings_panel = SettingsPanel(controls, self)
        self.settings_panel.frame.grid(row=0, column=0, sticky="ew")
        self.root.bind("<Escape>", lambda _event: self._stop())

        status_frame = ttk.Frame(controls)
        status_frame.grid(row=1, column=0, sticky="ew")
        status_frame.columnconfigure(0, weight=1)
        self.progress_bar = ttk.Progressbar(
            status_frame, variable=self.progress, maximum=100, mode="determinate"
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew")
        self.progress_bar.grid_remove()  # only shown once a run starts
        self.status_label = ttk.Label(
            status_frame, textvariable=self.status, wraplength=250, justify=tk.LEFT
        )
        self.status_label.grid(row=1, column=0, sticky="w", pady=(12, 0))
        self.progress_detail_label = ttk.Label(
            status_frame,
            textvariable=self.progress_detail,
            wraplength=250,
            justify=tk.LEFT,
            font=("TkDefaultFont", 9),
        )
        self.progress_detail_label.grid(row=2, column=0, sticky="w")
        self.elapsed_label = ttk.Label(
            status_frame,
            textvariable=self.elapsed_time,
            font=("TkDefaultFont", 9),
        )
        self.elapsed_label.grid(row=3, column=0, sticky="w")
        self.results_separator = ttk.Separator(status_frame, orient=tk.HORIZONTAL)
        self.results_separator.grid(row=4, column=0, sticky="ew", pady=12)
        self.results_label = ttk.Label(status_frame, text="Results:")
        self.results_label.grid(row=5, column=0, sticky="w")
        self.summary_label = ttk.Label(
            status_frame, textvariable=self.summary, wraplength=220, justify=tk.LEFT
        )
        self.summary_label.grid(row=6, column=0, sticky="w", pady=(4, 0))
        self.summary.trace_add("write", lambda *_: self._refresh_results_visibility())
        self._refresh_results_visibility()

        plot_frame = ttk.LabelFrame(self.root, text="Signal visualisation", padding=6)
        plot_frame.grid(row=0, column=1, sticky="nsew", padx=6, pady=12)
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        self.plot = PlotView(plot_frame)
        self.navigator = PeakNavigator(plot_frame, self)
        self.plot.draw_empty()

    def _choose_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose signal CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.input_path.set(path)
            if self.output_path.get() == "peaks.txt":
                self.output_path.set(str(Path(path).with_name("peaks.txt")))
            self.progress.set(0)
            self.progress_bar.grid_remove()
            self.last_progress_percent = 0
            self.progress_detail.set("")
            self.elapsed_time.set("")
            self.summary.set("")
            self._clear_cache()
            self.full_peaks = np.empty(0, dtype=int)
            self.navigator.busy = False
            self.navigator.selected_index = -1
            self.navigator.set_enabled(False)
            self.navigator.position.set("No peaks selected")
            self.plot.draw_empty()
            self.status.set("")

    def _start(self) -> None:
        if not self.input_path.get():
            messagebox.showerror("Missing input", "Choose a signal CSV file first.")
            return
        try:
            settings = {
                "chunk_size": int(self.chunk_size.get()),
                "overlap": int(self.overlap.get()),
                "height": float(self.height.get()),
                "distance": round(float(self.distance.get()) * SAMPLE_RATE_HZ),
                "preview_samples": int(self.preview_samples.get()),
                "show_live": self.show_live.get(),
                "input_path": self.input_path.get(),
                "column": self.channel.get(),
            }
            if (
                settings["chunk_size"] <= 0
                or settings["overlap"] < 0
                or settings["preview_samples"] <= 0
                or settings["distance"] <= 0
            ):
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid settings",
                "Chunk size and preview samples must be positive; overlap must "
                "be zero or greater; height must be numeric; distance must be a positive number of seconds.",
            )
            return

        self.run_settings = settings  # peak views reuse exactly what the run used
        self.plot.channel = str(settings["column"])
        self.cancel_event.clear()
        self._set_running(True)
        self.progress.set(0)
        self.progress_bar.grid()
        self.last_progress_percent = 0
        self.status.set("Processing signal... 0%")
        self.progress_detail.set("Preparing signal data...")
        started_at = time.perf_counter()
        self.processing_started_at = started_at
        self._update_elapsed_time()
        self.full_peaks = np.empty(0, dtype=int)
        self._clear_cache()
        self.navigator.selected_index = -1
        self.navigator.set_enabled(False)
        self.navigator.position.set("")
        self.summary.set("")  # hide the previous run's results
        self.plot.clear(
            None
            if settings["show_live"]
            else "Live plot is off. The signal appears when analysis finishes."
        )
        threading.Thread(
            target=run_analysis,
            args=(self, settings, started_at),
            daemon=True,
        ).start()

    def _live_update(self, signal: np.ndarray, rows: int) -> None:
        self.progress_detail.set(f"{rows:,} rows scanned")
        if signal.size:
            self.plot.draw_live(signal, rows)

    def _chunk_progress(self, percent: int, rows: int) -> None:
        self._set_progress(percent)
        self.progress_detail.set(f"{rows:,} rows scanned")

    def _set_progress(self, percent: int) -> None:
        self.last_progress_percent = max(self.last_progress_percent, percent)
        self.progress.set(self.last_progress_percent)
        self.status.set(f"Processing signal... {self.last_progress_percent}%")

    def _update_elapsed_time(self) -> None:
        if self.processing_started_at is None:
            return
        elapsed = time.perf_counter() - self.processing_started_at
        self.elapsed_time.set(f"Processing time: {int(elapsed)} s")
        self.root.after(100, self._update_elapsed_time)

    def _finished(
        self,
        raw: np.ndarray,
        clean: np.ndarray,
        peaks: np.ndarray,
        output: Path,
        overview_indices: np.ndarray,
        overview_values: np.ndarray,
        overview_total_samples: int,
        processing_seconds: float,
    ) -> None:
        self.full_peaks = peaks
        self.overview_indices = overview_indices
        self.overview_values = overview_values
        self.overview_total_samples = overview_total_samples
        self.navigator.selected_index = 0 if len(peaks) else -1
        self.progress.set(100)
        self.last_progress_percent = 100
        self.status.set("Analysis complete.")
        self.progress_detail.set(f"{len(peaks):,} total peaks detected")
        self.elapsed_time.set(f"Processing time: {int(processing_seconds)} s")
        self.processing_started_at = None
        self.summary.set(
            f"Detected peaks: {len(peaks):,}\n"
            f"Preview samples: {len(raw):,}\n"
            f"Processing time: {int(processing_seconds)} s\n"
            f"Saved: {output}"
        )
        if len(peaks):
            self.navigator.set_enabled(True)
            self.navigator.update_position()
            self.navigator.start_load()
        else:
            self.navigator.position.set("No peaks detected")
            self.navigator.set_enabled(False)
            self.plot.draw_signal(raw, clean, peaks, offset=0)
        self._set_running(False)

    def _clear_cache(self) -> None:
        self.cache.clear()
        self.overview_indices = np.empty(0, dtype=int)
        self.overview_values = np.empty(0, dtype=float)
        self.overview_total_samples = 0

    def _create_cache(self) -> None:
        self._clear_cache()
        self.cache.create()

    def _close(self) -> None:
        self._clear_cache()
        self.root.destroy()

    def _stop(self) -> None:
        """Ask the worker to stop; it replies with _cancelled once it has."""
        if str(self.settings_panel.stop_button["state"]) == tk.DISABLED:
            return
        self.cancel_event.set()
        self.settings_panel.stop_button.configure(state=tk.DISABLED)
        self.status.set("Stopping...")

    def _cancelled(self) -> None:
        """Return to the initial state so the user can pick a file and rerun."""
        self.processing_started_at = None
        self._clear_cache()
        self.full_peaks = np.empty(0, dtype=int)
        self.navigator.selected_index = -1
        self.progress.set(0)
        self.last_progress_percent = 0
        self.progress_detail.set("")
        self.elapsed_time.set("")
        self.summary.set("")
        self.navigator.position.set("No peaks selected")
        self.navigator.set_enabled(False)
        self.plot.draw_empty()
        self.status.set("")
        self._set_running(False)

    def _failed(self, message: str) -> None:
        self.status.set(f"Error: {message}")
        self.progress_detail.set("")
        self.processing_started_at = None
        self._clear_cache()  # a failed run must not leave a partial cache on disk
        self.navigator.set_enabled(False)
        self._set_running(False)
        messagebox.showerror("Processing failed", message)

    def _refresh_results_visibility(self) -> None:
        """Show the Results block only while there are results to show."""
        for widget in (self.results_separator, self.results_label, self.summary_label):
            if self.summary.get():
                widget.grid()
            else:
                widget.grid_remove()

    def _set_running(self, running: bool) -> None:
        self.running = running
        self.settings_panel.refresh_run_state()
        self.settings_panel.set_running(running)

def main() -> None:
    ChunkCache.remove_leftovers()  # from runs that were killed or crashed
    root = tk.Tk()
    SignalAnalyzerApp(root)
    root.mainloop()
