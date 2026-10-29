import numpy as np

from marginum_signal_analyzer.analysis import (
    find_busy_window,
    fit_peak_counts,
    process_chunks,
    sample_metadata,
)

def test_chunk_processing_uses_global_positions_and_no_duplicates():
    signal = np.zeros(30)
    signal[[8, 24]] = [100, 100]
    peaks = process_chunks(
        [signal[:15], signal[15:]],
        overlap=5,
        baseline_window=3,
        filter_size=1,
        height=60,
        distance=5,
    )
    np.testing.assert_array_equal(peaks, [8, 24])

def test_busy_window_contains_most_peaks(tmp_path):
    path = tmp_path / "signal.csv"
    signal = np.zeros(30)
    signal[[20, 22, 24]] = 100
    path.write_text(
        "adc1,adc2\n" + "".join(f"0,{value}\n" for value in signal),
        encoding="utf-8",
    )
    window, count = find_busy_window(
        path, window_samples=10, baseline_window=3, filter_size=1, height=60, distance=1
    )
    assert count == 3
    assert len(window) == 10

def test_metadata_and_linear_fit():
    assert sample_metadata("high_2gr_resuc.csv") == (2, True)
    slope, intercept, r_squared = fit_peak_counts([1, 2, 3], [10, 20, 30])
    assert slope == 10
    assert np.isclose(intercept, 0)
    assert r_squared == 1

def test_process_file_accepts_chunk_size(tmp_path):
    path = tmp_path / "signal.csv"
    signal = np.zeros(30)
    signal[15] = 100
    path.write_text(
        "adc1,adc2\n" + "".join(f"0,{value}\n" for value in signal),
        encoding="utf-8",
    )
    from marginum_signal_analyzer.analysis import process_file

    peaks = process_file(
        path,
        chunk_size=10,
        overlap=5,
        baseline_window=3,
        filter_size=1,
        height=60,
        distance=5,
    )
    np.testing.assert_array_equal(peaks, [15])

def test_process_chunks_reports_progress():
    progress = []
    process_chunks(
        [np.zeros(3), np.zeros(2)],
        overlap=1,
        baseline_window=3,
        filter_size=1,
        progress=lambda chunk, rows: progress.append((chunk, rows)),
    )
    assert progress == [(1, 3), (2, 5)]
