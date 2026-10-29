import numpy as np

from marginum_signal_analyzer.analysis import process_chunks

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