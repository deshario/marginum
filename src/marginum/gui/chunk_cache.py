from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np


class ChunkCache:
    """Signal chunks saved as .npy files in a temp dir, for fast windowed reads."""

    def __init__(self) -> None:
        self.directory: Path | None = None
        self.chunks: list[tuple[int, int, Path]] = []

    def clear(self) -> None:
        if self.directory is not None:
            shutil.rmtree(self.directory, ignore_errors=True)
        self.directory = None
        self.chunks = []

    def create(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="marginum-signal-cache-"))

    def save(self, chunk_number: int, chunk_start: int, chunk: np.ndarray) -> None:
        chunk_path = self.directory / f"chunk-{chunk_number:06d}.npy"
        np.save(chunk_path, chunk)
        self.chunks.append((chunk_start, chunk_start + len(chunk), chunk_path))

    def read_window(self, start: int, stop: int) -> np.ndarray:
        values: list[np.ndarray] = []
        for chunk_start, chunk_stop, chunk_path in self.chunks:
            if chunk_stop <= start:
                continue
            if chunk_start >= stop:
                break
            chunk = np.load(chunk_path, mmap_mode="r")
            left = max(0, start - chunk_start)
            right = min(len(chunk), stop - chunk_start)
            values.append(np.asarray(chunk[left:right]))
        return np.concatenate(values) if values else np.empty(0, dtype=float)
