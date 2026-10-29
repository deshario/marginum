import numpy as np
import pytest

from marginum_signal_analyzer.signal_io import read_signal_chunks

def create_csv(tmp_path):
    path = tmp_path / "signal.csv"
    path.write_text("adc1,adc2\n1,10\n2,20\n3,30\n4,40\n", encoding="utf-8")
    return path

def test_reads_chunks_without_losing_samples(tmp_path):
    chunks = list(read_signal_chunks(create_csv(tmp_path), chunk_size=2))
    assert len(chunks) == 2
    np.testing.assert_array_equal(np.concatenate(chunks), [10, 20, 30, 40])

def test_reads_column_by_name(tmp_path):
    chunks = list(read_signal_chunks(create_csv(tmp_path), column="adc1"))
    np.testing.assert_array_equal(np.concatenate(chunks), [1, 2, 3, 4])

def test_reports_progress(tmp_path):
    progress = []
    list(
        read_signal_chunks(
            create_csv(tmp_path),
            chunk_size=2,
            progress=lambda read, total: progress.append((read, total)),
        )
    )
    assert len(progress) == 2
    assert progress[-1][0] <= progress[-1][1]

def test_rejects_invalid_signal_files(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("adc1\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="out of range"):
        list(read_signal_chunks(path))