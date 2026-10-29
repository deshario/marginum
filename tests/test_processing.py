import numpy as np
import pytest

from marginum_signal_analyzer.processing import (
    detect_peaks,
    estimate_baseline,
    find_signal_peaks,
    remove_baseline,
)

def test_constant_baseline_is_preserved():
    signal = np.full(101, 25.0)
    np.testing.assert_allclose(estimate_baseline(signal, window=9), signal)

def test_baseline_is_removed():
    signal = np.full(9, 25.0)
    np.testing.assert_allclose(remove_baseline(signal, estimate_baseline(signal, window=9)), 0)

def test_detects_known_peaks_and_ignores_flat_signal():
    signal = np.zeros(200)
    signal[[50, 150]] = [80, 100]
    peaks, _ = detect_peaks(signal, height=60, distance=50)
    np.testing.assert_array_equal(peaks, [50, 150])
    assert len(detect_peaks(np.zeros(200), height=60, distance=50)[0]) == 0

def test_full_pipeline_returns_detected_peaks():
    signal = np.full(201, 100.0)
    signal[100] = 180.0
    _, flat, clean, peaks = find_signal_peaks(
        signal, baseline_window=9, filter_size=1, height=60, distance=10
    )
    assert flat[100] > 60
    assert clean[100] > 60
    np.testing.assert_array_equal(peaks, [100])

def test_invalid_processing_parameters_are_rejected():
    with pytest.raises(ValueError):
        estimate_baseline(np.ones(3), window=4)
