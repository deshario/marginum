import numpy as np

from marginum_signal_analyzer.baseline import (
    classify_peak,
    detect_peaks,
    estimate_baseline,
)

def test_baseline_estimation():
    signal = np.full(6_000, 25.0)

    estimated_baseline = estimate_baseline(signal)

    np.testing.assert_allclose(estimated_baseline, signal)

def test_peak_classification():
    assert [classify_peak(amplitude) for amplitude in [5, 10, 20]] == [
        "low",
        "middle",
        "high",
    ]

# Check detection of two known peaks.
def test_detects_two_known_peaks():
    signal_length = 100_000
    peak_positions = [20_000, 60_000]
    peak_amplitudes = [20, 30]
    expected_peak_positions = np.array(peak_positions)
    signal = np.zeros(signal_length)
    signal[peak_positions] = peak_amplitudes

    peaks, _ = detect_peaks(signal)

    np.testing.assert_array_equal(peaks, expected_peak_positions)

# Check that a flat signal has no peaks.
def test_flat_signal_has_no_peaks():
    signal_length = 100_000
    flat_signal = np.zeros(signal_length)

    detected_peaks, _ = detect_peaks(flat_signal)

    assert len(detected_peaks) == 0

# Check that nearby peaks are not counted twice.
def test_ignores_peaks_that_are_too_close():
    signal = np.zeros(100_000)
    signal[[20_000, 20_100, 60_000]] = [20, 25, 30]
    expected_peak_positions = np.array([20_100, 60_000])

    detected_peaks, _ = detect_peaks(signal)

    np.testing.assert_array_equal(detected_peaks, expected_peak_positions)
