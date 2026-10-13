from pathlib import Path

from industrial_project.chunk_reader import read_chunks, read_window

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "IDP_signal_example_[sami_dmitry_september2023].csv"
)

def main() -> None:
    total_samples = 0
    adc1_min = None
    adc1_max = None
    adc2_min = None
    adc2_max = None

    for chunk_number, chunk in enumerate(read_chunks(CSV_PATH), start=1):
        adc1_values = [sample[0] for sample in chunk]
        adc2_values = [sample[1] for sample in chunk]

        total_samples += len(chunk)
        adc1_min = min(adc1_values) if adc1_min is None else min(adc1_min, min(adc1_values))
        adc1_max = max(adc1_values) if adc1_max is None else max(adc1_max, max(adc1_values))
        adc2_min = min(adc2_values) if adc2_min is None else min(adc2_min, min(adc2_values))
        adc2_max = max(adc2_values) if adc2_max is None else max(adc2_max, max(adc2_values))

        if chunk_number == 1:
            print("First five samples:")
            for sample_index, (adc1, adc2) in enumerate(chunk[:5]):
                print(f"  {sample_index}: adc1={adc1}, adc2={adc2}")

    print(f"Total samples: {total_samples:,}")
    print(f"Duration: {total_samples / 50_000:.3f} seconds")
    print(f"adc1 range: {adc1_min}..{adc1_max}")
    print(f"adc2 range: {adc2_min}..{adc2_max}")

    start_time = 10
    end_time = 11
    sample_rate = 50_000
    start_sample = start_time * sample_rate
    end_sample = end_time * sample_rate
    window = read_window(CSV_PATH, start_sample, end_sample)

    print(f"Window: {start_time}..{end_time} seconds")
    print(f"Window sample indexes: {start_sample}..{end_sample - 1}")
    print(f"Window samples read: {len(window):,}")
    print(f"Window first sample: {window[0]}")
    print(f"Window last sample: {window[-1]}")

if __name__ == "__main__":
    main()