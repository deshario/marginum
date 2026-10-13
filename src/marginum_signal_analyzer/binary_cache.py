"""Binary caching for large Marginum signal CSV files."""

import csv
from pathlib import Path

import numpy as np

from marginum_signal_analyzer.chunk_reader import read_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "IDP_signal_example_[sami_dmitry_september2023].csv"
)
DEFAULT_CACHE_PATH = (
    PROJECT_ROOT
    / "cache"
    / "IDP_signal_example_[sami_dmitry_september2023].npy"
)

def count_samples(csv_path: str | Path) -> int:
    """Count data rows in a two-column signal CSV."""
    with Path(csv_path).open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)
        if header != ["adc1", "adc2"]:
            raise ValueError(f"Unexpected CSV header: {header}")
        return sum(1 for row in reader if row)

def convert_csv_to_binary(
    csv_path: str | Path,
    cache_path: str | Path,
    chunk_size: int = 50_000,
) -> Path:
    """Convert a signal CSV to a NumPy binary array."""
    sample_count = count_samples(csv_path)
    output_path = Path(cache_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signal = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.int16,
        shape=(sample_count, 2),
    )

    position = 0
    for chunk in read_chunks(csv_path, chunk_size=chunk_size):
        chunk_array = np.asarray(chunk, dtype=np.int16)
        end_position = position + len(chunk_array)
        signal[position:end_position] = chunk_array
        position = end_position

    signal.flush()
    del signal
    return output_path

def open_binary_signal(cache_path: str | Path) -> np.memmap:
    """Open a cached signal without loading it completely into memory."""
    return np.load(cache_path, mmap_mode="r")

def read_binary_window(
    signal: np.memmap,
    start_sample: int,
    end_sample: int,
) -> np.ndarray:
    """Read a selected sample range from a memory-mapped signal."""
    if start_sample < 0:
        raise ValueError("start_sample must be zero or greater")
    if end_sample <= start_sample:
        raise ValueError("end_sample must be greater than start_sample")
    if end_sample > len(signal):
        raise ValueError("end_sample is outside the signal")
    return np.asarray(signal[start_sample:end_sample])

def main() -> None:
    cache_path = convert_csv_to_binary(DEFAULT_CSV_PATH, DEFAULT_CACHE_PATH)
    signal = open_binary_signal(cache_path)
    print(f"Cache: {cache_path}")
    print(f"Shape: {signal.shape}")
    print(f"Dtype: {signal.dtype}")
    window_size = 50_000
    step_size = 50_000
    for window_number in range(2):
        start_sample = window_number * step_size
        end_sample = start_sample + window_size
        window = read_binary_window(signal, start_sample, end_sample)
        print(
            f"Window {window_number + 1}: "
            f"samples {start_sample}..{end_sample - 1}, "
            f"shape={window.shape}, "
            f"first={window[0].tolist()}, "
            f"last={window[-1].tolist()}"
        )

if __name__ == "__main__":
    main()