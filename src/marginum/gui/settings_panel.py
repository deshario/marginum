from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import SignalAnalyzerApp


class SettingsPanel:
    """Input/output fields, the advanced section and the Run/Stop row."""

    def __init__(self, parent: ttk.Frame, app: SignalAnalyzerApp) -> None:
        self.app = app
        self.advanced_expanded = False
        self.browse_buttons: list[ttk.Button] = []

        frame = self.frame = ttk.Frame(parent)
        frame.columnconfigure(1, weight=1)

        self._path_row(frame, 0, "Input CSV", app.input_path, app._choose_input)
        self._entry_row(frame, 1, "Peak height", app.height)
        self._entry_row(frame, 2, "Peak distance (s)", app.distance)

        self.advanced_button = ttk.Label(
            frame,
            text="▶ Advanced settings",
            cursor="hand2",
        )
        self.advanced_button.bind("<Button-1>", lambda _event: self._toggle_advanced())
        self.advanced_button.grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )
        self.advanced_frame = ttk.Frame(frame)
        self.advanced_frame.columnconfigure(1, weight=1)
        self._path_row(
            self.advanced_frame,
            0,
            "Output peaks",
            app.output_path,
            self._choose_output,
        )
        self._entry_row(self.advanced_frame, 1, "Chunk size", app.chunk_size)
        self._entry_row(self.advanced_frame, 2, "Overlap", app.overlap)
        self._entry_row(self.advanced_frame, 3, "Preview samples", app.preview_samples)

        run_row = ttk.Frame(frame)
        run_row.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(16, 8))
        run_row.columnconfigure(0, weight=3)
        run_row.columnconfigure(1, weight=1)
        self.start_button = ttk.Button(
            run_row, text="Run analysis", command=app._start
        )
        self.start_button.grid(row=0, column=0, sticky="ew")
        self.stop_button = ttk.Button(
            run_row, text="Stop", command=app._stop, state=tk.DISABLED
        )
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.live_check = ttk.Checkbutton(
            run_row, text="Show live plot (slower)", variable=app.show_live
        )
        self.live_check.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        app.input_path.trace_add("write", lambda *_: self.refresh_run_state())
        self.refresh_run_state()
        self.advanced_frame.grid_remove()

    def _toggle_advanced(self) -> None:
        self.advanced_expanded = not self.advanced_expanded
        if self.advanced_expanded:
            self.advanced_button.configure(text="▼ Advanced settings")
            self.advanced_frame.grid(
                row=7, column=0, columnspan=3, sticky="ew", pady=(4, 0)
            )
        else:
            self.advanced_button.configure(text="▶ Advanced settings")
            self.advanced_frame.grid_remove()

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

    def _choose_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save peak positions",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            self.app.output_path.set(path)

    def refresh_run_state(self) -> None:
        """Run is only available when a file is chosen and nothing is running."""
        can_run = not self.app.running and bool(self.app.input_path.get().strip())
        self.start_button.configure(state=tk.NORMAL if can_run else tk.DISABLED)

    def set_running(self, running: bool) -> None:
        state = tk.DISABLED if running else tk.NORMAL
        self.stop_button.configure(state=tk.NORMAL if running else tk.DISABLED)
        self.live_check.configure(state=state)
        for button in self.browse_buttons:
            button.configure(state=state)
