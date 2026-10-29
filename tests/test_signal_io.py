import numpy as np
import pytest

from marginum_signal_analyzer.signal_io import (
    downsample,
    read_signal,
    read_signal_chunks,
    read_signal_window,
)

def create_csv(tmp_path):
    path = tmp_path / "signal.csv"
    path.write_text("adc1,adc2\n1,10\n2,20\n3,30\n4,40\n", encoding="utf-8")
    return path

def test_reads_signal_column_and_window(tmp_path):
    path = create_csv(tmp_path)
    np.testing.assert_array_equal(read_signal(path), [10, 20, 30, 40])
    np.testing.assert_array_equal(read_signal_window(path, 1, 3), [20, 30])

def test_reads_chunks_without_losing_samples(tmp_path):
    chunks = list(read_signal_chunks(create_csv(tmp_path), chunk_size=2))
    np.testing.assert_array_equal(np.concatenate(chunks), [10, 20, 30, 40])

def test_rejects_invalid_signal_files(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("adc1\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="out of range"):
        read_signal(path)

def test_downsample_returns_indices_and_values():
    indices, values = downsample(np.arange(7), step=3)
    np.testing.assert_array_equal(indices, [0, 3, 6])
    np.testing.assert_array_equal(values, [0, 3, 6])
