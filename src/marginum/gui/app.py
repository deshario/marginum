from __future__ import annotations

import threading
import tkinter as tk
import shutil
import tempfile
import time
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from ..analysis import process_chunks
from ..processing import find_signal_peaks
from ..signal_io import read_signal_chunks
from .config import LIVE_REFRESH_SECONDS, SAMPLE_RATE_HZ
from .peak_navigator import PeakNavigator
from .plot_view import PlotView

class _Cancelled(Exception):
    """Raised inside the worker thread when the user presses Stop."""


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
        self.preview_samples = tk.StringVar(value="200000")
        self.status = tk.StringVar(value="")
        self.progress_detail = tk.StringVar(value="")
        self.elapsed_time = tk.StringVar(value="")
        self.summary = tk.StringVar(value="")
        self.progress = tk.DoubleVar()
        self.processing_seconds = 0.0
        self.advanced_expanded = False
        self.last_progress_percent = 0
        self.processing_started_at: float | None = None
        self.full_peaks = np.empty(0, dtype=int)
        self.overview_indices = np.empty(0, dtype=int)
        self.overview_values = np.empty(0, dtype=float)
        self.overview_total_samples = 0
        self.cache_dir: Path | None = None
        self.cached_chunks: list[tuple[int, int, Path]] = []
        self.start_button: ttk.Button
        self.stop_button: ttk.Button
        self.browse_buttons: list[ttk.Button] = []
        self.cancel_event = threading.Event()
        self.running = False
        self.run_settings: dict[str, int | float | str | bool] = {}
        self.show_live = tk.BooleanVar(value=True)
        self.plot: PlotView
        self.navigator: PeakNavigator
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
        controls.columnconfigure(1, weight=1)

        self._path_row(controls, 0, "Input CSV", self.input_path, self._choose_input)
        self._entry_row(controls, 1, "Peak height", self.height)
        self._entry_row(controls, 2, "Peak distance (s)", self.distance)

        self.advanced_button = ttk.Label(
            controls,
            text="▶ Advanced settings",
            cursor="hand2",
        )
        self.advanced_button.bind("<Button-1>", lambda _event: self._toggle_advanced())
        self.advanced_button.grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )
        self.advanced_frame = ttk.Frame(controls)
        self.advanced_frame.columnconfigure(1, weight=1)
        self._path_row(
            self.advanced_frame,
            0,
            "Output peaks",
            self.output_path,
            self._choose_output,
        )
        self._entry_row(self.advanced_frame, 1, "Chunk size", self.chunk_size)
        self._entry_row(self.advanced_frame, 2, "Overlap", self.overlap)
        self._entry_row(self.advanced_frame, 3, "Preview samples", self.preview_samples)

        run_row = self.run_row = ttk.Frame(controls)
        run_row.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(16, 8))
        run_row.columnconfigure(0, weight=3)
        run_row.columnconfigure(1, weight=1)
        self.start_button = ttk.Button(
            run_row, text="Run analysis", command=self._start
        )
        self.start_button.grid(row=0, column=0, sticky="ew")
        self.stop_button = ttk.Button(
            run_row, text="Stop", command=self._stop, state=tk.DISABLED
        )
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.live_check = ttk.Checkbutton(
            run_row, text="Show live plot (slower)", variable=self.show_live
        )
        self.live_check.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.root.bind("<Escape>", lambda _event: self._stop())
        self.input_path.trace_add("write", lambda *_: self._refresh_run_state())
        self._refresh_run_state()
        self.progress_bar = ttk.Progressbar(
            controls, variable=self.progress, maximum=100, mode="determinate"
        )
        self.progress_bar.grid(row=8, column=0, columnspan=3, sticky="ew")
        self.status_label = ttk.Label(
            controls, textvariable=self.status, wraplength=250, justify=tk.LEFT
        )
        self.status_label.grid(row=9, column=0, columnspan=3, sticky="w", pady=(12, 0))
        self.progress_detail_label = ttk.Label(
            controls,
            textvariable=self.progress_detail,
            wraplength=250,
            justify=tk.LEFT,
            font=("TkDefaultFont", 9),
        )
        self.progress_detail_label.grid(row=10, column=0, columnspan=3, sticky="w")
        self.elapsed_label = ttk.Label(
            controls,
            textvariable=self.elapsed_time,
            font=("TkDefaultFont", 9),
        )
        self.elapsed_label.grid(row=11, column=0, columnspan=3, sticky="w")
        self.results_separator = ttk.Separator(controls, orient=tk.HORIZONTAL)
        self.results_separator.grid(
            row=12, column=0, columnspan=3, sticky="ew", pady=12
        )
        self.results_label = ttk.Label(controls, text="Results:")
        self.results_label.grid(row=13, column=0, columnspan=3, sticky="w")
        self.summary_label = ttk.Label(
            controls, textvariable=self.summary, wraplength=220, justify=tk.LEFT
        )
        self.summary_label.grid(
            row=14, column=0, columnspan=3, sticky="w", pady=(4, 0)
        )
        self.advanced_frame.grid_remove()
        self.summary.trace_add("write", lambda *_: self._refresh_results_visibility())
        self._refresh_results_visibility()

        plot_frame = ttk.LabelFrame(self.root, text="Signal visualisation", padding=6)
        plot_frame.grid(row=0, column=1, sticky="nsew", padx=6, pady=12)
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        self.plot = PlotView(plot_frame)
        self.navigator = PeakNavigator(plot_frame, self)
        self.plot.draw_empty()

    def _toggle_advanced(self) -> None:
        self.advanced_expanded = not self.advanced_expanded
        offset = 0
        if self.advanced_expanded:
            self.advanced_button.configure(text="▼ Advanced settings")
            self.advanced_frame.grid(
                row=7, column=0, columnspan=3, sticky="ew", pady=(4, 0)
            )
            offset = 8
        else:
            self.advanced_button.configure(text="▶ Advanced settings")
            self.advanced_frame.grid_remove()
        rows = (
            self.run_row,
            self.progress_bar,
            self.status_label,
            self.progress_detail_label,
            self.elapsed_label,
            self.results_separator,
            self.results_label,
            self.summary_label,
        )
        for index, widget in enumerate(rows, start=7 + offset):
            widget.grid_configure(row=index)
        # grid_configure re-shows hidden widgets, so hide the empty Results block again
        self._refresh_results_visibility()

    def _path_row(
        self,
        frame: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        command,
    ) -> None:
        ttk.Label(frame, text=f"{label}:").grid(
            row=row * 2, column=0, columnspan=3, sticky="w", pady=(6, 2)
        )
        ttk.Entry(frame, textvariable=variable, width=22).grid(
            row=row * 2 + 1, column=0, columnspan=2, sticky="ew", padx=(0, 6), pady=(0, 4)
        )
        browse = ttk.Button(frame, text="Browse", command=command)
        browse.grid(row=row * 2 + 1, column=2, pady=(0, 4))
        self.browse_buttons.append(browse)

    def _entry_row(
        self, frame: ttk.Frame, row: int, label: str, variable: tk.StringVar
    ) -> None:
        ttk.Label(frame, text=f"{label}:").grid(
            row=row * 2, column=0, columnspan=3, sticky="w", pady=(6, 2)
        )
        ttk.Entry(frame, textvariable=variable, width=12).grid(
            row=row * 2 + 1, column=0, columnspan=3, sticky="ew", pady=(0, 4)
        )

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

    def _choose_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save peak positions",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            self.output_path.set(path)

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
        self.cancel_event.clear()
        self._set_running(True)
        self.progress.set(0)
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
            target=self._process,
            args=(settings, started_at),
            daemon=True,
        ).start()

    def _process(
        self, settings: dict[str, int | float], started_at: float
    ) -> None:
        try:
            preview_size = int(settings["preview_samples"])
            show_live = bool(settings["show_live"])
            preview_parts: list[np.ndarray] = []
            preview_rows = 0
            live_values: list[np.ndarray] = []
            overview_indices: list[np.ndarray] = []
            overview_values: list[np.ndarray] = []
            overview_total_samples = 0
            last_live_update = 0.0
            file_progress = 0

            def update(chunk_number: int, rows: int) -> None:
                nonlocal last_live_update
                if not show_live:
                    return
                now = time.perf_counter()
                if now - last_live_update < LIVE_REFRESH_SECONDS and chunk_number != 1:
                    return
                last_live_update = now
                live_signal = (
                    np.concatenate(live_values) if live_values else np.empty(0)
                )
                self.root.after(
                    0,
                    self._live_update,
                    live_signal,
                    rows,
                    file_progress,
                )

            def chunks():
                nonlocal preview_rows, file_progress, overview_total_samples
                self._create_cache()

                def update_file_progress(read_bytes: int, total_bytes: int) -> None:
                    file_progress = min(
                        99,
                        max(1, int(read_bytes * 100 / max(total_bytes, 1))),
                    )
                    self.root.after(
                        0,
                        self._reading_update,
                        file_progress,
                    )

                for chunk_number, chunk in enumerate(
                    read_signal_chunks(
                        str(settings["input_path"]),
                        chunk_size=int(settings["chunk_size"]),
                        progress=update_file_progress,
                    ),
                    start=1,
                ):
                    if self.cancel_event.is_set():
                        raise _Cancelled
                    chunk_start = overview_total_samples
                    chunk_path = self.cache_dir / f"chunk-{chunk_number:06d}.npy"
                    np.save(chunk_path, chunk)
                    self.cached_chunks.append(
                        (chunk_start, chunk_start + len(chunk), chunk_path)
                    )
                    if preview_rows < preview_size:
                        needed = preview_size - preview_rows
                        preview_parts.append(chunk[:needed])
                        preview_rows += min(len(chunk), needed)
                    step = max(1, len(chunk) // 1_000)
                    sampled_indices = overview_total_samples + np.arange(
                        0, len(chunk), step
                    )
                    sampled_values = chunk[::step]
                    if show_live:
                        live_values.append(sampled_values)
                    overview_indices.append(sampled_indices)
                    overview_values.append(sampled_values)
                    overview_total_samples += len(chunk)
                    self.root.after(
                        0,
                        self._chunk_progress,
                        file_progress,
                        overview_total_samples,
                    )
                    yield chunk

            peaks = process_chunks(
                chunks(),
                progress=update,
                overlap=int(settings["overlap"]),
                height=float(settings["height"]),
                distance=int(settings["distance"]),
            )
            preview = np.concatenate(preview_parts) if preview_parts else np.empty(0)
            if len(preview) > preview_size:
                preview = preview[:preview_size]
            if preview.size == 0:
                raise ValueError("Signal file contains no samples")
            baseline, flat, clean, _ = find_signal_peaks(
                preview,
                height=float(settings["height"]),
                distance=int(settings["distance"]),
            )
            output = Path(self.output_path.get())
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(output, peaks, fmt="%d")
            self.root.after(
                0,
                self._finished,
                preview,
                baseline,
                flat,
                clean,
                peaks,
                output,
                np.concatenate(overview_indices)
                if overview_indices
                else np.empty(0, dtype=int),
                np.concatenate(overview_values)
                if overview_values
                else np.empty(0, dtype=float),
                overview_total_samples,
                time.perf_counter() - started_at,
            )
        except _Cancelled:
            self.root.after(0, self._cancelled)
        except (OSError, ValueError, IndexError) as error:
            self.root.after(0, self._failed, str(error))

    def _live_update(self, signal: np.ndarray, rows: int, percent: int) -> None:
        self.progress_detail.set(f"{rows:,} rows scanned")
        if signal.size:
            self.plot.draw_live(signal, rows)

    def _reading_update(self, percent: int) -> None:
        self._set_progress(percent)

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
        baseline: np.ndarray,
        flat: np.ndarray,
        clean: np.ndarray,
        peaks: np.ndarray,
        output: Path,
        overview_indices: np.ndarray,
        overview_values: np.ndarray,
        overview_total_samples: int,
        processing_seconds: float,
    ) -> None:
        self.processing_seconds = processing_seconds
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
            self.plot.draw_signal(raw, baseline, flat, clean, peaks, offset=0)
        self._set_running(False)

    def _clear_cache(self) -> None:
        if self.cache_dir is not None:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir = None
        self.cached_chunks = []
        self.overview_indices = np.empty(0, dtype=int)
        self.overview_values = np.empty(0, dtype=float)
        self.overview_total_samples = 0

    def _create_cache(self) -> None:
        self._clear_cache()
        self.cache_dir = Path(tempfile.mkdtemp(prefix="marginum-signal-cache-"))

    def _close(self) -> None:
        self._clear_cache()
        self.root.destroy()

    def _read_window(self, start: int, stop: int) -> np.ndarray:
        if self.cached_chunks:
            values: list[np.ndarray] = []
            for chunk_start, chunk_stop, chunk_path in self.cached_chunks:
                if chunk_stop <= start:
                    continue
                if chunk_start >= stop:
                    break
                chunk = np.load(chunk_path, mmap_mode="r")
                left = max(0, start - chunk_start)
                right = min(len(chunk), stop - chunk_start)
                values.append(np.asarray(chunk[left:right]))
            return np.concatenate(values) if values else np.empty(0, dtype=float)

        values: list[np.ndarray] = []
        position = 0
        for chunk in read_signal_chunks(
            str(self.run_settings["input_path"]),
            chunk_size=max(1, min(int(self.run_settings["chunk_size"]), stop - start)),
        ):
            chunk_stop = position + len(chunk)
            if chunk_stop > start and position < stop:
                left = max(0, start - position)
                right = min(len(chunk), stop - position)
                values.append(chunk[left:right])
            position = chunk_stop
            if position >= stop:
                break
        return np.concatenate(values) if values else np.empty(0, dtype=float)

    def _stop(self) -> None:
        """Ask the worker to stop; it replies with _cancelled once it has."""
        if str(self.stop_button["state"]) == tk.DISABLED:
            return
        self.cancel_event.set()
        self.stop_button.configure(state=tk.DISABLED)
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
        self.plot.clear()
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

    def _refresh_run_state(self) -> None:
        """Run is only available when a file is chosen and nothing is running."""
        can_run = not self.running and bool(self.input_path.get().strip())
        self.start_button.configure(state=tk.NORMAL if can_run else tk.DISABLED)

    def _set_running(self, running: bool) -> None:
        self.running = running
        state = tk.DISABLED if running else tk.NORMAL
        self._refresh_run_state()
        self.stop_button.configure(state=tk.NORMAL if running else tk.DISABLED)
        self.live_check.configure(state=state)
        for button in self.browse_buttons:
            button.configure(state=state)

def main() -> None:
    root = tk.Tk()
    SignalAnalyzerApp(root)
    root.mainloop()
