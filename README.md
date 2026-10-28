# Marginum Signal Analyzer

## Running the application

Run the interactive signal viewer:

```bash
uv run python -m marginum_signal_analyzer.window_viewer
```

The viewer:

- Opens the cached signal data.
- Shows a selected signal window for ADC1 and ADC2.
- Lets you move between windows with Previous and Next.
- Includes Play/Pause controls for automatic window playback.
- Loops back to the first window after the final window.
- Does not detect or print peak results.

Start playback automatically:

```bash
uv run python -m marginum_signal_analyzer.window_viewer --autoplay
```

Run the signal-processing and peak-detection pipeline:

```bash
uv run python -m marginum_signal_analyzer.baseline
```

The processing pipeline:

1. Reads the configured signal range.
2. Uses ADC2 for analysis.
3. Estimates the changing baseline.
4. Subtracts the baseline.
5. Smooths the corrected signal.
6. Detects peaks using prominence and minimum-distance settings.
7. Classifies each peak as low, middle, or high.
8. Saves the detected-peak plot to `output/detected_peaks.png`.

## Running tests

Run all MVP tests:

```bash
uv run pytest -v
```

The tests currently cover CSV chunk reading, selected signal-window reading,
baseline estimation, peak classification, known peak detection, flat-signal
control behavior, and minimum peak spacing.
