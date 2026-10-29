from __future__ import annotations

import csv
from collections.abc import Iterator
from collections.abc import Callable
from pathlib import Path

import numpy as np

def _signal_column(header: list[str], column: int | str) -> int:
    if isinstance(column, int):
        if column < 0 or column >= len(header):
            raise ValueError(f"Signal column index is out of range: {column}")
        return column
    try:
        return header.index(column)
    except ValueError as error:
        raise ValueError(f"Signal column not found: {column}") from error

def read_signal_chunks(
    path: str | Path,
    chunk_size: int = 1_000_000,
    column: int | str = 1,
    progress: Callable[[int, int], None] | None = None,
) -> Iterator[np.ndarray]:
    """Yield one numeric signal column from a CSV file in bounded chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    source_path = Path(path)
    total_bytes = source_path.stat().st_size
    with source_path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("Signal file is empty") from error

        column_index = _signal_column(header, column)
        values: list[float] = []
        processed_bytes = len(",".join(header).encode("utf-8")) + 1
        for line_number, row in enumerate(reader, start=2):
            if len(row) <= column_index:
                raise ValueError(f"Missing signal column at line {line_number}")
            try:
                values.append(float(row[column_index]))
            except ValueError as error:
                raise ValueError(
                    f"Invalid signal value at line {line_number}: {row[column_index]!r}"
                ) from error
            processed_bytes += len(",".join(row).encode("utf-8")) + 1
            if len(values) == chunk_size:
                if progress is not None:
                    progress(min(processed_bytes, total_bytes), total_bytes)
                yield np.asarray(values, dtype=float)
                values = []

        if values:
            if progress is not None:
                progress(min(processed_bytes, total_bytes), total_bytes)
            yield np.asarray(values, dtype=float)
