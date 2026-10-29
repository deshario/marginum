## Requirements

Python 3.13+, [uv](https://docs.astral.sh/uv/), and Tk (included with most Python installs).

## Run

```bash
uv sync
uv run marginum
```

## Use

1. Click **Browse** and choose a CSV file.
2. Set **Peak height** and **Peak distance** if needed.
3. Click **Run analysis**.
4. Use **Previous peak**, **Next peak** and **Overview** to inspect the results.

## Tests

```bash
uv run pytest
```