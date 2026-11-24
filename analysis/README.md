# Signal analysis notebook

`signal_analysis.ipynb` finds peaks in raw sensor recordings and relates the peak
count to tumour size and suctioned volume. All results are already saved in the notebook, so it
can be read without running it.

## Requirements

- Python 3
- Jupyter, or VS Code with the Jupyter extension

Open a terminal in this folder and install the required packages in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy pandas scipy matplotlib jupyter
```

## Data (not included, about 10 GB)

The data is **Marginum Measurement Data** on Moodle. Put each file below in the same folder
as the notebook, named as in the left column.

| Name the notebook expects | Dataset in Moodle (archive → file) |
|---|---|
| `Signal 1.csv` | `Test signals from previous years/Signal 1.7z` (rename the file inside) |
| `2Gb signal.csv` | `Test signals from previous years/Signal 2.7z` |
| `just water.csv` | `Test signals from previous years/Signal 4.7z` |
| `8Gb signal.csv` | `Data for suction of known volumes + 4½ h hours [2025].7z` → `Original/1. LONG_suction_HIGH_CONC_4ml_(and_3_times_resuction).csv` |

**Tumour files**: from `Test signals from previous years/Tumor 1 gr - 2 gr - 3 gr with resuction.7z`.
Copy the 10 files from the folder inside it directly into this folder.

## Run

1. Open `signal_analysis.ipynb` in Jupyter (`jupyter notebook
   signal_analysis.ipynb`) or in VS Code, with this folder as the working folder.
   In VS Code, select the `.venv` environment as the kernel.

2. Click **Run All**.
