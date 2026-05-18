# Machine Learning–Based Wildfire Ignition Risk Prediction for Ontario, Canada

**MSc Applied Data Science — Reykjavik University, 2026**  
**Author:** Anna Lambert  
**Supervisor:** Christoph Lohrmann, Assistant Professor

---

## Overview

This repository contains the code base for a machine learning pipeline 
that predicts wildfire ignition risk and cause attribution across Ontario, 
Canada at 9 km spatial resolution. The pipeline covers data extraction from 
Google Earth Engine, feature engineering, model training (Random Forest, XGBoost and 
ConvLSTM2D), threshold optimisation, and feature importance analysis.

The study period is 2010–2024. Models are trained on 2010–2020 data, 
validated on 2021–2022, and evaluated on a held-out 2023–2024 test set.

---

## Repository Structure

├── src/
│   ├── data_preparation/     
│   ├── random_forest/    # Tabular modelling pipeline (ignition + cause)
│   ├── xgboost/          # Tabular modelling pipeline (ignition + cause + aggregated ignition)
│   └── convlstm/         # Spatiotemporal model and data loading utilities
├── notebooks/            # Exploratory analysis and visualisation
├── figures/              # Generated figures and maps
├── README.md
└── pyproject.toml        # uv-managed dependency specification

---

## Environment Setup

### Requirements

- [Visual Studio Code](https://code.visualstudio.com/)
- [uv](https://github.com/astral-sh/uv) — fast Python package manager
- Python 3.12
- A Google Earth Engine (https://code.earthengine.google.com/) account with access to the 
  `ru-thesis-2026` project assets (request access from annalam22@ru.is)
- A Google Drive account with sufficient storage for `.npy` cache 
  files (~10 GB for the full 2010–2024 dataset)
- A Copernicus Climate Data Store (CDS) API key for ERA5-Land access (https://confluence.ecmwf.int/display/CKB/Climate+Data+Store+%28CDS%29+User+Guide)

---

### 1. Clone the repository

```bash
git clone https://github.com/<Anna-Lambert>/ru-thesis-2026.git
cd ru-thesis-2026
```

---

### 2. Create and activate the virtual environment

This project uses [uv](https://github.com/astral-sh/uv) for dependency 
management. If you do not have uv installed:

```bash
pip install uv
```

Then initialise and activate the environment:

```bash
# Initialise the project (only needed once)
uv init

# Create the virtual environment
uv venv

# Activate — Git Bash / macOS / Linux
source .venv/Scripts/activate     # Git Bash on Windows
source .venv/bin/activate          # macOS / Linux
```

---

### 3. Install dependencies

```bash
uv add pandas geopandas numpy shapely xarray rasterio rioxarray \
       scikit-learn ipykernel netCDF4 cdsapi
```

All packages and their resolved versions are recorded in `pyproject.toml` 
and can be restored exactly in a new environment with:

```bash
uv sync
```

---

### 4. Configure the CDS API key

ERA5-Land data is retrieved programmatically via the Copernicus Climate 
Data Store. Create a `.cdsapirc` file in your home directory:

```bash
touch ~/.cdsapirc
notepad ~/.cdsapirc      # Windows
nano ~/.cdsapirc         # macOS / Linux
```

Paste the following, replacing the placeholder with your personal key 
from [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu):
url: https://cds.climate.copernicus.eu/api
key: <your-personal-api-key>

---

### 5. Connect VS Code to the virtual environment

Open the project in VS Code:

```bash
code .
```

Select the virtual environment as the Python interpreter:

- Press `Ctrl+Shift+P` → **Python: Select Interpreter**
- Choose `.venv` from the list

The `ipykernel` package included in the environment enables the 
virtual environment to be used directly inside Jupyter Notebooks 
within VS Code without additional configuration.

---

### 6. Authenticate Google Earth Engine

```python
import ee
ee.Authenticate()
ee.Initialize(project='ru-thesis-2026')
```

Authentication opens a browser window for OAuth login. This is 
required once per machine. Subsequent sessions call 
`ee.Initialize()` only.

---

### 7. Google Colab setup (ConvLSTM2D only)

The ConvLSTM2D model was developed and trained on 
[Google Colab](https://colab.research.google.com/) using an NVIDIA 
Tesla T4 GPU and TensorFlow 2.20. No local installation is required 
for this component.

To run the ConvLSTM notebooks:

1. Upload the notebooks from `src/convlstm/` to your Google Drive 
   or open them directly in Colab via the GitHub link.

2. Mount your Google Drive at the start of each session:

```python
   from google.colab import drive
   drive.mount('/content/drive')
```

3. Set the expected directory structure on your Drive:
MyDrive/
└── RU_Thesis_2026/
├── gee_cache/       # .npy daily feature grids downloaded from GEE
├── models/          # saved model checkpoints and bundles
└── figures/         # output figures

4. Authenticate GEE at the start of each Colab session:

```python
   import ee
   ee.Authenticate()
   ee.Initialize(project='ru-thesis-2026')
```

5. On first run, the data download cell will retrieve daily feature 
   grids from GEE and cache them as `.npy` files in 
   `MyDrive/RU_Thesis_2026/gee_cache/`. At the start of each 
   session, cached files are copied from Google Drive to the 
   local Colab SSD at `/content/npy_cache/` to reduce read 
   latency during training — Drive I/O is significantly slower 
   than local SSD and would otherwise bottleneck the 
   disk-based tf.data generator. This copy step runs 
   automatically and takes approximately 5 minutes on first 
   run per session.

> **Note on session limits:** Google Colab free tier disconnects 
> after 3–6 hours. The training script includes checkpoint recovery — 
> set `INITIAL_EPOCH` to the last completed epoch number to resume 
> training after a disconnection without losing progress.

> **Recommended runtime:** For reasonable training times 
> (approximately 7 minutes per epoch), select a GPU runtime in 
> Colab: `Runtime → Change runtime type → T4 GPU`.

## Python Dependencies

The environment is managed with uv for reproducible dependency resolution. 
The table below lists the primary packages and their role in the pipeline.

| Category | Package | Version | Role |
|---|---|---|---|
| Data orchestration | `pandas` | 3.0.1 | Tabular data manipulation and temporal indexing |
| | `numpy` | 2.4.3 | Numerical and array computations |
| Geospatial analysis | `geopandas` | 1.1.3 | Vector spatial data management |
| | `shapely` | 2.1.2 | Geometric operations and spatial relationships |
| | `pyproj` | 3.7.2 | Coordinate reference system transformations |
| Atmospheric data | `xarray` | 2026.2.0 | Multi-dimensional ERA5-Land array handling |
| | `netcdf4` | 1.7.4 | HDF5/NetCDF data storage interface |
| | `cdsapi` | 0.7.7 | Programmatic retrieval of ECMWF reanalysis data |
| Raster processing | `rasterio` | 1.5.0 | Geospatial raster access and manipulation |
| | `rioxarray` | 0.22.0 | Raster-to-xarray integration and spatial clipping |
| Machine learning | `scikit-learn` | 1.8.0 | Modelling, feature scaling, and evaluation |
| | `scipy` | 1.17.1 | Statistical and mathematical functions |
| Deep learning (Colab) | `tensorflow` | 2.20.0 | ConvLSTM2D model training and inference |
| | `keras` | — | High-level neural network API (bundled with TF) |
| Development | `ipykernel` | 7.2.0 | Virtual environment integration in VS Code |
| | `jupyter-client` | 8.8.0 | Interactive notebook execution management |

> **Note:** TensorFlow and Keras are pre-installed on Google Colab 
> and do not need to be added to the local uv environment. All other 
> packages in the table above are managed locally via uv.

**Supporting dependencies** installed automatically with the above:

- **Geospatial drivers:** `pyogrio` (0.12.1), `cligj` (0.7.2) — 
  optimised spatial file I/O
- **Networking and API:** `requests` (2.32.5), `urllib3` (2.6.3), 
  `tqdm` (4.67.3) — robust data downloads from Copernicus servers
- **Computational utilities:** `joblib` (1.5.3), 
  `threadpoolctl` (3.6.0) — parallel processing and hardware 
  resource management during model training

---

## Data Pipeline

### Overview

All data preparation was conducted using 
[Google Earth Engine (GEE)](https://earthengine.google.com/), 
a cloud-based geospatial processing platform that provides 
server-side access to a multi-petabyte catalogue of satellite 
imagery and environmental datasets, including climate reanalyses, 
land cover products, and digital elevation models.

The pipeline produces one multi-band raster image per calendar 
month, where each image encodes all predictor and target variables 
for every day in that month at approximately 9 km spatial 
resolution. The spatial context is fully preserved, allowing 
model predictions to be mapped back onto the original grid.

All data sources are reprojected to a common coordinate reference 
system (EPSG:4326) — the geographic coordinate system used by 
ERA5-Land, GPS devices, and most global climate datasets. 
This ensures that the highest-resolution and most frequently 
updated source (ERA5-Land) requires no reprojection, preserving 
its spatial precision, while all other sources are transformed 
prior to feature extraction.

### Pipeline Components

The daily feature pipeline is structured around three components:

1. **Spatial reference grid** — definition of the Ontario study 
   area mask and common grid at ~9 km pixel spacing (EPSG:4326)

2. **Dynamic daily features** — meteorological variables from 
   ERA5-Land, temporal lag aggregates (7-day, 30-day rolling 
   windows), post-fire recovery status, and binary ignition 
   targets from the Canadian National Fire Database (CNFDB)

3. **Export** — monthly multi-band image stacks exported as 
   GEE assets to `projects/ru-thesis-2026/assets/Ontario_Monthly_2010-2024/`

Static and yearly features (land cover, topography, population 
density, road proximity) are produced separately as single-band 
images aligned to the same reference grid, exported to 
`projects/ru-thesis-2026/assets/Ontario_Static_2010-2024/`.

### Data Sources

| Feature group | Source | Resolution | Update frequency |
|---|---|---|---|
| Meteorological | ERA5-Land (ECMWF) via CDS API | ~9 km | Daily |
| Fire occurrences | Canadian National Fire Database (CNFDB) | Point | Annual |
| Land cover | GEE static layer | 300 m | Fixed (2020) |
| Topography | SRTM Digital Elevation Model | 30 m | Fixed (2011) |
| Population density | GEE static layer | ~1 km | Annual (clamped 2020) |
| Road proximity | National Road Network | Vector | Fixed (2024) |

---

## Modelling Pipeline

### Random Forest (tabular, pixel-level)

Script in `src/random_forest/` cover:

- Feature extraction and CSV export from GEE assets
- Training and validation split (chronological, DOY 91–310 fire season)
- Class imbalance handling via 20:1 undersampling of the 
  majority no-fire class during training
- Random Forest classifier with manually selected hyperparameters
- Operating threshold selection at 0.10
- Binary cause classification (Human vs Other, Natural vs Other)
- Feature importance (native gain, permutation)

> **Note:** The Random Forest model is provided as a benchmark 
> and was not carried forward to test set evaluation.

### XGBoost (tabular, pixel-level)

Scripts in `src/xgboost/` cover:

- Feature extraction and CSV export from GEE assets
- Training and validation split (chronological, DOY 91–310 fire season)
- Class imbalance handling (`scale_pos_weight`, SMOTE evaluation)
- Optuna hyperparameter tuning (50 trials, TPE sampler)
- Cross-validated threshold optimisation (min. recall ≥ 0.70)
- Binary cause classification (Human vs Other, Natural vs Other)
- Feature importance (native gain, permutation, LIME)

### ConvLSTM2D (spatiotemporal, Google Colab)

Notebook in `src/convlstm/` are designed to run on 
[Google Colab](https://colab.research.google.com/) with a GPU 
runtime and Google Drive mounted. They cover:

- GEE authentication and Google Drive mount with automatic
  directory creation for cache, models, and figures
- Drive-to-local SSD cache copy at session start for faster
  disk reads during training (~10x faster than Drive)
- Static feature download from GEE with Drive mirroring
  for session persistence
- Disk-based `tf.data` generator pipeline reading `.npy` cache 
  files from Google Drive
- 14-day sequence construction from  from daily `.npy` cache files
- Ontario pixel masking and fire season filtering (DOY 91–310)
- Model training with weighted BCE loss (`pos_weight=500`),
  early stopping, and checkpoint recovery across sessions
- Streaming evaluation on the 2021–2022 validation set

No local installation is required for the ConvLSTM component. 
Open the notebooks directly in Colab and follow the setup 
instructions in Section 7 above.

---

## Results Summary

| Model | ROC-AUC | Recall | FPR | Notes |
|---|---|---|---|---|
| Random Forest ignition | 0.926 | 0.935 | 0.249 | Threshold 0.10 |
| XGBoost ignition (validation) | 0.929 | 0.773 | 0.099 | Threshold 0.729 |
| XGBoost ignition (test) | 0.866 | 0.546 | 0.100 | 2023–2024 |
| ConvLSTM2D (validation) | 0.923 | 0.702 | 0.078 | Unoptimised, SEQ_LEN=14 |
| XGBoost cause (classified only) | — | — | — | 96.0% accuracy |

---

## Citation

```bibtex
@mastersthesis{lambert2026,
  author  = {Lambert, Anna},
  title   = {Machine Learning--Based Wildfire Ignition Risk 
             Prediction for {Ontario}, {Canada}},
  school  = {Reykjavik University},
  year    = {2026},
  type    = {{MSc} Thesis, Applied Data Science}
}
```

---

## AI Assistance Disclosure

Python scripts, figures, and grammatical review in this project 
were developed with partial assistance from 
[Claude](https://www.anthropic.com) (Anthropic), a large language 
model used as a coding and writing aid. All research questions, 
methodological decisions, model design, result interpretation, 
and academic content are the original work of the author.

---

## License

This repository is made available for academic reproducibility. 
Please cite the thesis if you use any part of this code or methodology.