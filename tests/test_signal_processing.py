import numpy as np

from marginum_signal_analyzer.baseline import detect_peaks

# Check that the detector finds two peaks at known positions.
def test_peak_detection_finds_two_known_peaks():
    signal = np.zeros(100_000)
    signal[[20_000, 60_000]] = [20, 30]

    peaks, _ = detect_peaks(signal)

    np.testing.assert_array_equal(peaks, [20_000, 60_000])

# Check that a flat signal produces no detected peaks.
def test_peak_detection_finds_no_peaks_in_flat_signal():
    peaks, _ = detect_peaks(np.zeros(100_000))

    assert len(peaks) == 0
