from marginum_signal_analyzer.chunk_reader import read_chunks, read_window

# Create a small CSV file that can be reused by the reader tests.
def create_sample_signal_csv(tmp_path):
    csv_path = tmp_path / "signal.csv"
    csv_path.write_text(
        "adc1,adc2\n"
        "1,10\n"
        "2,20\n"
        "3,30\n"
        "4,40\n",
        encoding="utf-8",
    )
    return csv_path

# Check that CSV samples are returned in separate chunks.
def test_read_chunks_splits_sample_csv_into_chunks(tmp_path):
    csv_path = create_sample_signal_csv(tmp_path)
    expected_chunks = [
        [(1, 10), (2, 20)],
        [(3, 30), (4, 40)],
    ]

    actual_chunks = list(read_chunks(csv_path, chunk_size=2))

    assert actual_chunks == expected_chunks

# Check that only the requested sample range is returned.
def test_read_window_returns_selected_sample_range(tmp_path):
    csv_path = create_sample_signal_csv(tmp_path)

    assert read_window(csv_path, 1, 3) == [(2, 20), (3, 30)]
