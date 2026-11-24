# Signal analysis notebook

`signal_analysis.ipynb` counts the peaks in sensor recordings. It then compares the number of
peaks with tumour size and with the suctioned volume.

To see the results without running it, open the notebook or view it online:
https://deshario.github.io/marginum/

## What you need

- Python 3
- Jupyter, or VS Code with the Jupyter extension

Open a terminal in this folder and run:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy pandas scipy matplotlib jupyter
```

## Data (not included)

Download [**Marginum Measurement Data**](https://moodle.uef.fi/mod/folder/view.php?id=956460) from Moodle and put the files next to the notebook, named as in the left column.

| File name | Where to find it on Moodle |
|---|---|
| `Signal 1.csv` | `Test signals from previous years/Signal 1.7z` (rename the file inside) |
| `2Gb signal.csv` | `Test signals from previous years/Signal 2.7z` |
| `just water.csv` | `Test signals from previous years/Signal 4.7z` |
| `8Gb signal.csv` | `Data for suction of known volumes + 4½ h hours [2025].7z` → `Original/1. LONG_suction_HIGH_CONC_4ml_(and_3_times_resuction).csv` |

**Tumour files:** extract `Test signals from previous years/Tumor 1 gr - 2 gr - 3 gr with resuction.7z`.
It has a folder with 10 files. Copy the 10 files (not the folder) into this folder.

## Run

1. Open `signal_analysis.ipynb` in one of these:
   - **Jupyter:** in the terminal, run `jupyter notebook signal_analysis.ipynb`
   - **VS Code:** Install the Jupyter extension, open the notebook and then select `.venv` as the kernel.
2. Click **Run All**.
