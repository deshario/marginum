import csv
from collections.abc import Iterator
from pathlib import Path

def read_chunks(
    csv_path: str | Path,
    chunk_size: int = 50_000,
) -> Iterator[list[tuple[int, int]]]:
    """Yield the signal CSV as small chunks of (adc1, adc2) samples."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    with Path(csv_path).open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)

        if header != ["adc1", "adc2"]:
            raise ValueError(f"Unexpected CSV header: {header}")

        chunk: list[tuple[int, int]] = []

        for line_number, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(
                    f"Line {line_number}: expected two columns, got {row}"
                )

            try:
                sample = (int(row[0]), int(row[1]))
            except ValueError as error:
                raise ValueError(
                    f"Line {line_number}: values must be integers, got {row}"
                ) from error

            chunk.append(sample)

            if len(chunk) == chunk_size:
                yield chunk
                chunk = []

        if chunk:
            yield chunk


def read_window(
    csv_path: str | Path,
    start_sample: int,
    end_sample: int,
    chunk_size: int = 50_000,
) -> list[tuple[int, int]]:
    """Read samples from start_sample, inclusive, to end_sample, exclusive."""
    if start_sample < 0:
        raise ValueError("start_sample must be zero or greater")
    if end_sample <= start_sample:
        raise ValueError("end_sample must be greater than start_sample")

    window: list[tuple[int, int]] = []
    current_sample = 0

    for chunk in read_chunks(csv_path, chunk_size=chunk_size):
        chunk_end = current_sample + len(chunk)

        if chunk_end <= start_sample:
            current_sample = chunk_end
            continue

        if current_sample >= end_sample:
            break

        first_index = max(start_sample - current_sample, 0)
        last_index = min(end_sample - current_sample, len(chunk))
        window.extend(chunk[first_index:last_index])

        current_sample = chunk_end

        if current_sample >= end_sample:
            break

    return window
