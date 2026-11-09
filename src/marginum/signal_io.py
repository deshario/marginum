from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd

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
    try:
        header = list(pd.read_csv(source_path, nrows=0).columns)
    except pd.errors.EmptyDataError as error:
        raise ValueError("Signal file is empty") from error
    column_index = _signal_column(header, column)

    rows_read = 0
    with source_path.open("rb") as file:
        try:
            reader = pd.read_csv(
                file,
                usecols=[column_index],
                dtype=np.float64,
                chunksize=chunk_size,
            )
            for frame in reader:
                values = frame.iloc[:, 0].to_numpy()
                if values.size == 0:
                    continue
                bad = np.flatnonzero(np.isnan(values))
                if bad.size:
                    line = rows_read + int(bad[0]) + 2
                    raise ValueError(f"Missing or invalid signal value at line {line}")
                rows_read += len(values)
                if progress is not None:
                    progress(min(file.tell(), total_bytes), total_bytes)
                yield values
        except (ValueError, pd.errors.ParserError) as error:
            if str(error).startswith("Missing or invalid"):
                raise
            raise ValueError(f"Invalid signal data after line {rows_read + 1}: {error}") from error
