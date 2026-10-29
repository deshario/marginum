from __future__ import annotations

import threading
import tkinter as tk
import shutil
import tempfile
import time
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .analysis import process_chunks
from .processing import find_signal_peaks
from .signal_io import read_signal_chunks

class SignalAnalyzerApp:
    SAMPLE_RATE_HZ = 50_000

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
        self.distance = tk.StringVar(value="25000")
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
        self.peak_position = tk.StringVar(value="No peaks selected")
        self.full_peaks = np.empty(0, dtype=int)
        self.overview_indices = np.empty(0, dtype=int)
        self.overview_values = np.empty(0, dtype=float)
        self.overview_total_samples = 0
        self.cache_dir: Path | None = None
        self.cached_chunks: list[tuple[int, int, Path]] = []
        self.selected_peak_index = -1
        self.navigation_busy = False
        self.start_button: ttk.Button
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas: FigureCanvasTkAgg
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
        self._entry_row(controls, 2, "Peak distance", self.distance)

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

        self.start_button = ttk.Button(
            controls, text="Run analysis", command=self._start
        )
        self.start_button.grid(
            row=7, column=0, columnspan=3, sticky="ew", pady=(16, 8)
        )
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

        plot_frame = ttk.LabelFrame(self.root, text="Signal visualisation", padding=6)
        plot_frame.grid(row=0, column=1, sticky="nsew", padx=6, pady=12)
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        navigation = ttk.Frame(plot_frame)
        navigation.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.previous_button = ttk.Button(
            navigation, text="Previous peak", command=lambda: self._navigate_peak(-1)
        )
        self.previous_button.pack(side=tk.LEFT)
        self.next_button = ttk.Button(
            navigation, text="Next peak", command=lambda: self._navigate_peak(1)
        )
        self.next_button.pack(side=tk.LEFT, padx=(6, 0))
        self.overview_button = ttk.Button(
            navigation, text="Overview", command=self._show_overview
        )
        self.overview_button.pack(side=tk.LEFT, padx=(6, 0))
        self._set_navigation_enabled(False)
        ttk.Label(navigation, textvariable=self.peak_position).pack(
            side=tk.LEFT, padx=(12, 0)
        )
        self._draw_empty_plot()

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
            self.start_button,
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
        ttk.Button(frame, text="Browse", command=command).grid(
            row=row * 2 + 1, column=2, pady=(0, 4)
        )

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
            self.navigation_busy = False
            self.selected_peak_index = -1
            self._set_navigation_enabled(False)
            self.peak_position.set("No peaks selected")
            self._draw_empty_plot()
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
                "distance": int(self.distance.get()),
                "preview_samples": int(self.preview_samples.get()),
            }
            if (
                settings["chunk_size"] <= 0
                or settings["overlap"] < 0
                or settings["preview_samples"] <= 0
            ):
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid settings",
                "Chunk size and preview samples must be positive; overlap must "
                "be zero or greater; height and distance must be numeric.",
            )
            return

        self.start_button.configure(state=tk.DISABLED)
        self.progress.set(0)
        self.last_progress_percent = 0
        self.status.set("Processing signal... 0%")
        self.progress_detail.set("Preparing signal data...")
        started_at = time.perf_counter()
        self.processing_started_at = started_at
        self._update_elapsed_time()
        self.full_peaks = np.empty(0, dtype=int)
        self._clear_cache()
        self.selected_peak_index = -1
        self._set_navigation_enabled(False)
        self.peak_position.set("Detecting peaks...")
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
            preview_parts: list[np.ndarray] = []
            preview_rows = 0
            live_values: list[np.ndarray] = []
            overview_indices: list[np.ndarray] = []
            overview_values: list[np.ndarray] = []
            overview_total_samples = 0
            last_live_update = 0
            file_progress = 0

            def update(chunk_number: int, rows: int) -> None:
                nonlocal last_live_update
                if rows - last_live_update < 5_000_000 and chunk_number != 1:
                    return
                last_live_update = rows
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
                        self.input_path.get(),
                        chunk_size=int(settings["chunk_size"]),
                        progress=update_file_progress,
                    ),
                    start=1,
                ):
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
        except (OSError, ValueError, IndexError) as error:
            self.root.after(0, self._failed, str(error))

    def _live_update(self, signal: np.ndarray, rows: int, percent: int) -> None:
        self.progress_detail.set(f"{rows:,} rows scanned")
        if signal.size:
            self._draw_live_plot(signal, rows)

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

    def _draw_live_plot(self, signal: np.ndarray, rows: int) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.plot(signal, color="#2563eb", linewidth=0.8)
        axis.set_title(f"Live signal overview ({rows:,} samples scanned)")
        axis.set_xlabel("Sample index")
        axis.set_ylabel("Response")
        axis.grid(alpha=0.25)
        self.figure.tight_layout()
        self.canvas.draw_idle()

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
        self.selected_peak_index = 0 if len(peaks) else -1
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
            self._set_navigation_enabled(True)
            self._update_peak_position()
            self.navigation_busy = True
            self.previous_button.configure(state=tk.DISABLED)
            self.next_button.configure(state=tk.DISABLED)
            threading.Thread(
                target=self._load_selected_peak,
                args=(self.selected_peak_index,),
                daemon=True,
            ).start()
        else:
            self.peak_position.set("No peaks detected")
            self._set_navigation_enabled(False)
            self._draw_plot(raw, baseline, flat, clean, peaks, offset=0)
        self.start_button.configure(state=tk.NORMAL)

    def _navigate_peak(self, direction: int) -> None:
        if not len(self.full_peaks) or self.navigation_busy:
            self.status.set("No detected peaks to navigate.")
            return
        self.selected_peak_index = (
            self.selected_peak_index + direction
        ) % len(self.full_peaks)
        self._update_peak_position()
        self.navigation_busy = True
        self.previous_button.configure(state=tk.DISABLED)
        self.next_button.configure(state=tk.DISABLED)
        self.status.set("Loading selected peak...")
        selected_index = self.selected_peak_index
        threading.Thread(
            target=self._load_selected_peak,
            args=(selected_index,),
            daemon=True,
        ).start()

    def _show_overview(self) -> None:
        if not self.input_path.get():
            self.status.set("Choose a signal file first.")
            return
        if not len(self.full_peaks):
            self.status.set("Run analysis before showing the overview.")
            return
        if not len(self.overview_values):
            self.status.set("Overview data is unavailable. Run analysis again.")
            return
        self._draw_overview(
            self.overview_indices,
            self.overview_values,
            self.overview_total_samples,
        )

    def _draw_overview(
        self,
        sample_indices: np.ndarray,
        sample_values: np.ndarray,
        total_samples: int,
    ) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.plot(
            sample_indices,
            sample_values,
            color="#555555",
            linewidth=0.5,
            label="Signal (downsampled)",
        )
        peak_positions = self.full_peaks[
            self.full_peaks < total_samples
        ]
        peak_indices = np.searchsorted(sample_indices, peak_positions)
        valid = peak_indices < len(sample_values)
        if np.any(valid):
            axis.scatter(
                peak_positions[valid],
                sample_values[peak_indices[valid]],
                color="red",
                s=15,
                marker="x",
                label=f"{len(peak_positions)} peaks",
            )
        axis.set_title(f"Signal overview - {len(self.full_peaks):,} peaks detected")
        axis.set_xlabel("Sample index")
        axis.set_ylabel("Response")
        axis.legend(loc="upper right")
        axis.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.status.set("Full-file overview loaded.")

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

    def _update_peak_position(self) -> None:
        sample = int(self.full_peaks[self.selected_peak_index])
        elapsed_seconds = sample / self.SAMPLE_RATE_HZ
        self.peak_position.set(
            f"Peak {self.selected_peak_index + 1} of {len(self.full_peaks)}\n"
            f"Sample: {sample:,}\n"
            f"Time: {elapsed_seconds:.2f} s"
        )

    def _load_selected_peak(self, selected_index: int) -> None:
        peak = int(self.full_peaks[selected_index])
        window = max(1, int(self.preview_samples.get()))
        start = max(0, peak - window // 2)
        raw = self._read_window(start, start + window)
        if raw.size == 0:
            return
        baseline, flat, clean, _ = find_signal_peaks(
            raw,
            height=float(self.height.get()),
            distance=int(self.distance.get()),
        )
        visible_peaks = self.full_peaks[
            (self.full_peaks >= start) & (self.full_peaks < start + len(raw))
        ] - start
        self.root.after(
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
        self.navigation_busy = False
        self.previous_button.configure(state=tk.NORMAL)
        self.next_button.configure(state=tk.NORMAL)
        if selected_index == self.selected_peak_index:
            self._draw_plot(raw, baseline, flat, clean, peaks, offset)
            self.status.set("Selected peak loaded.")

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
            self.input_path.get(),
            chunk_size=max(1, min(int(self.chunk_size.get()), stop - start)),
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

    def _failed(self, message: str) -> None:
        self.status.set(f"Error: {message}")
        self.progress_detail.set("")
        self.processing_started_at = None
        self._set_navigation_enabled(False)
        self.start_button.configure(state=tk.NORMAL)
        messagebox.showerror("Processing failed", message)

    def _set_navigation_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.previous_button.configure(state=state)
        self.next_button.configure(state=state)
        self.overview_button.configure(state=state)

    def _draw_empty_plot(self) -> None:
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

    def _draw_plot(
        self,
        raw: np.ndarray,
        baseline: np.ndarray,
        flat: np.ndarray,
        clean: np.ndarray,
        peaks: np.ndarray,
        offset: int = 0,
    ) -> None:
        self.figure.clear()
        x = np.arange(len(raw)) + offset
        local_peaks = np.asarray(peaks, dtype=int)
        peak_x = local_peaks + offset
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
        axis.set_xlabel("Sample index")
        axis.set_ylabel("Response")
        axis.set_title(f"Signal - {len(peaks)} peaks found")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")
        self.figure.tight_layout()
        self.canvas.draw_idle()

def main() -> None:
    root = tk.Tk()
    SignalAnalyzerApp(root)
    root.mainloop()
