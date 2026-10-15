import numpy as np

from marginum_signal_analyzer.baseline import detect_peaks

# Check that the detector finds two peaks at known positions.
def test_peak_detection_finds_two_known_peaks():
    signal_length = 100_000
    peak_positions = [20_000, 60_000]
    peak_amplitudes = [20, 30]
    expected_peak_positions = np.array(peak_positions)
    signal = np.zeros(signal_length)
    signal[peak_positions] = peak_amplitudes

    peaks, _ = detect_peaks(signal)

    np.testing.assert_array_equal(peaks, expected_peak_positions)

# Check that a flat signal produces no detected peaks.
def test_peak_detection_finds_no_peaks_in_flat_signal():
    signal_length = 100_000
    flat_signal = np.zeros(signal_length)

    detected_peaks, _ = detect_peaks(flat_signal)

    assert len(detected_peaks) == 0
