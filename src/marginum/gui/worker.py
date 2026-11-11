from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..analysis import process_chunks
from ..processing import find_signal_peaks
from ..signal_io import read_signal_chunks
from .config import LIVE_REFRESH_SECONDS

if TYPE_CHECKING:
    from .app import SignalAnalyzerApp


class Cancelled(Exception):
    """Raised inside the worker thread when the user presses Stop."""


def run_analysis(
    app: SignalAnalyzerApp, settings: dict[str, int | float], started_at: float
) -> None:
    """Worker-thread body: read, cache, analyse, save; reports back via root.after."""
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
            app.root.after(
                0,
                app._live_update,
                live_signal,
                rows,
                file_progress,
            )

        def chunks():
            nonlocal preview_rows, file_progress, overview_total_samples
            app._create_cache()

            def update_file_progress(read_bytes: int, total_bytes: int) -> None:
                file_progress = min(
                    99,
                    max(1, int(read_bytes * 100 / max(total_bytes, 1))),
                )
                app.root.after(
                    0,
                    app._reading_update,
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
                if app.cancel_event.is_set():
                    raise Cancelled
                chunk_start = overview_total_samples
                app.cache.save(chunk_number, chunk_start, chunk)
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
                app.root.after(
                    0,
                    app._chunk_progress,
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
        output = Path(app.output_path.get())
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(output, peaks, fmt="%d")
        app.root.after(
            0,
            app._finished,
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
    except Cancelled:
        app.root.after(0, app._cancelled)
    except (OSError, ValueError, IndexError) as error:
        app.root.after(0, app._failed, str(error))
