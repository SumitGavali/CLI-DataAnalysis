#  Kaggle Prep

<p align="center">
  <img src="https://i.pinimg.com/736x/aa/db/ac/aadbac8f3df73fe8655a4f2d21923c86.jpg" alt="Kaggle Prep Banner" width="100" height='90' onerror="this.style.display='none'"/>
</p>

<p align="center">
  <strong>One-command CLI tool to download, profile, visualize, and generate production-ready starter code & notebooks for any Kaggle dataset.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/kaggle-prep/"><img src="https://img.shields.io/badge/pypi-v0.3.0-blue.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/kaggle-prep/"><img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg" alt="Python 3.9+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://github.com/SumitGavali/CLI-DataAnalysis/actions"><img src="https://img.shields.io/badge/tests-12%20passed-brightgreen.svg" alt="Tests"></a>
  <a href="https://github.com/SumitGavali/CLI-DataAnalysis/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"></a>
</p>

---

##  Table of Contents

- [Why Kaggle Prep?](#-why-kaggle-prep)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Kaggle Credentials Setup](#-kaggle-credentials-setup)
- [Quick Start in 30 Seconds](#-quick-start-in-30-seconds)
- [CLI Command Reference](#-cli-command-reference)
- [Generated Outputs & Architecture](#-generated-outputs--architecture)
- [Comprehensive Error & Troubleshooting Guide](#-comprehensive-error--troubleshooting-guide)
- [Development & Contributing](#-development--contributing)
- [License](#-license)

---

##  Why Kaggle Prep?

Setting up a new data science project or Kaggle competition usually takes 30–60 minutes of repetitive boilerplate:
1. Downloading and unzipping files.
2. Checking missing values, types, duplicates, and memory footprints.
3. Writing 10+ standard EDA visualization scripts (distributions, outliers, correlations, class balance).
4. Writing preprocessing pipelines (imputation, categorical encoding, scaling).
5. Setting up a starter Jupyter Notebook with baseline ML models.

**`kaggle-prep` automates the entire workflow in a single terminal command.**

```bash
kaggle-prep uciml/iris --all --target Species
```

---

##  Key Features

1. **Zero-Config Smart Download**: Instantly download public Kaggle datasets without needing API keys upfront using built-in `kagglehub` integration.
2. **Interactive Setup Wizard (`--setup`)**: Configure and validate your Kaggle API credentials interactively in seconds.
3. **Instant Automated Profiling**: Compute row/col counts, missing rates, data types, duplicate counts, IQR outliers, skewness, and cardinality.
4. **Standalone HTML Reports**: Generates responsive, self-contained HTML reports with zero external runtime dependencies.
5. **10+ Production EDA Visualizations**:
   - Dataset overview & metric cards
   - Missing value matrix & percent heatmaps
   - Feature distribution histograms & KDE curves
   - Skewness ranking & Q-Q normality plots
   - Violin plots & IQR outlier summaries
   - Correlation heatmaps & top-correlated feature pairs
   - High-cardinality flags & categorical frequency bar charts
   - Datetime row count trends
   - **Target-Aware Analysis**: Class balance bar/pie charts and feature distributions segmented by target class.
6. **Auto-Generated Preprocessing Scripts**: Clean Python code with Scikit-Learn pipelines tailored to your dataset's column schema.
7. **Complete Starter Jupyter Notebooks**: Pre-configured with modular sections: imports, EDA, missing analysis, outlier detection, ML preprocessing, and baseline model training.
8. **Multi-Format & Universal Python Support**: Works natively on **Python 3.9, 3.10, 3.11, 3.12, and 3.13+** across Windows, macOS, and Linux. Supports `.csv`, `.tsv`, `.parquet`, `.xlsx`, and `.json`.

---

## Installation

### Standard Installation (via `pip`)
```bash
pip install kaggle-prep
```

### Full Installation (with ML Baseline and Excel support)
```bash
pip install "kaggle-prep[all]"
```

### Using `pipx` (Isolated CLI application)
```bash
pipx install kaggle-prep
```

### Install from Source
```bash
git clone https://github.com/SumitGavali/CLI-DataAnalysis.git 
cd CLI-DataAnalysis
pip install -e .
```

---

##  Kaggle Credentials Setup

### Option 1: Interactive Wizard (Recommended)
Simply run the setup wizard:
```bash
kaggle-prep --setup
```
The wizard will guide you through entering your username and API key, and automatically creates a secure `~/.kaggle/kaggle.json` file.

### Option 2: Manual Setup
1. Log into your account at [Kaggle](https://www.kaggle.com).
2. Navigate to **Account Settings** -> [kaggle.com/settings/api](https://www.kaggle.com/settings/api).
3. Click **"Create New Token"** to download `kaggle.json`.
4. Move `kaggle.json` to your home directory:
   - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - **Linux / macOS**: `~/.kaggle/kaggle.json`
5. On Linux/macOS, set secure permissions:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Option 3: Zero-Config Mode (No Credentials Needed)
For public datasets, `kaggle-prep` automatically uses `kagglehub` to download data without requiring any API token or login!

---

## Quick Start in 30 Seconds

### 1. Full Automated Analysis (All-in-One)
```bash
kaggle-prep uciml/iris --all
```

### 2. Supervised Analysis with a Target Column
```bash
kaggle-prep uciml/iris --all --target Species
```

### 3. Kaggle Competitions
```bash
kaggle-prep titanic --competition --all --target Survived
```

### 4. Analyze Existing Local Data
```bash
# Point to an existing local dataset folder
kaggle-prep my-dataset --local --all
```

---

##  CLI Command Reference

```text
Usage: kaggle-prep [dataset] [OPTIONS]

Positional Arguments:
  dataset                     Kaggle dataset slug (e.g. 'uciml/iris') or competition name

Workflow & Pipeline Flags:
  -a, --all                   Run full pipeline (profile + report + visualize + preprocess + notebook)
  -p, --profile               Generate data profile JSON and console summary
  -r, --report                Generate a standalone HTML profile report
      --visualize             Generate 10+ EDA visualization charts
      --preprocess            Generate an automated preprocessing Python script
  -n, --notebook              Generate a complete starter Jupyter notebook (.ipynb)

Data & Target Options:
  -t, --target TARGET         Specify target column name for supervised EDA & balance analysis
  -o, --output-dir DIR        Directory to save downloaded data (default: data)
  -l, --local                 Use local data in output directory (skips downloading)
  -c, --competition           Download from Kaggle Competition instead of Dataset
  -s, --sample N              Sample N rows from dataset (ideal for multi-GB datasets)

Visualization Controls:
  -m, --max-cols N            Maximum number of columns to plot in distributions (default: 10)
  -d, --dpi DPI               Plot figure resolution DPI (default: 150)
  -f, --fig-format FORMAT     Plot file format: png, pdf, svg, jpg (default: png)

Utility Flags:
      --setup                 Launch interactive Kaggle credentials setup wizard
  -v, --version               Show program version and exit
      --verbose               Enable verbose diagnostic logs
  -h, --help                  Show help message and exit
```

---

##  Generated Outputs & Architecture

When you run `kaggle-prep <dataset> --all`, the following structured directories are generated:

```text
project_root/
├── data/
│   ├── dataset.csv                  # Downloaded raw dataset
│   └── dataset_preprocess.py        # Ready-to-run preprocessing pipeline
├── data_profiles/
│   └── dataset_profile.json         # JSON schema & metrics summary
├── reports/
│   └── dataset_report.html          # Interactive standalone HTML report
├── eda_plots_<dataset>/             # 10+ high-res EDA charts
│   ├── 01_overview.png
│   ├── 02_missing_values.png
│   ├── 03_distributions.png
│   ├── 04_skewness.png
│   ├── 05_qq_normality.png
│   ├── 06_violin_plots.png
│   ├── 07_outlier_summary.png
│   ├── 08_correlation.png
│   ├── 09_cardinality.png
│   ├── 10_categorical_bars.png
│   ├── 11_target_balance.png
│   └── 12_target_features.png
└── notebooks/
    └── dataset_analysis.ipynb       # Complete starter Jupyter notebook
```

---

##  Comprehensive Error & Troubleshooting Guide

### 1. `401 Unauthorized` / `Authentication Error`
* **Cause**: Your `kaggle.json` token is missing, expired, or corrupted.
* **Resolution**:
  ```bash
  kaggle-prep --setup
  ```
  Follow the prompt to re-enter your Kaggle username and API key.

---

### 2. `403 Forbidden`
* **Cause**: The dataset or competition requires accepting competition rules or terms of service on Kaggle before downloading.
* **Resolution**:
  1. Open your browser and visit: `https://www.kaggle.com/datasets/<dataset_name>` (or `https://www.kaggle.com/competitions/<competition_name>`).
  2. Click **"Download"** or **"Join Competition / I Understand and Accept"**.
  3. Re-run `kaggle-prep <dataset_name> --all`.

---

### 3. `404 Not Found`
* **Cause**: Typo in the dataset identifier or attempting to download a competition without the `--competition` flag.
* **Resolution**:
  - Datasets must follow the `owner/dataset-name` format:
    ```bash
    kaggle-prep uciml/iris --all
    kaggle-prep debayank2024/netflix-movies-and-series --all
    ```
  - For competitions, supply the `--competition` (or `-c`) flag:
    ```bash
    kaggle-prep titanic --competition --all
    kaggle-prep house-prices-advanced-regression-techniques -c --all
    ```

---

### 4. `429 Too Many Requests` (Rate Limit)
* **Cause**: Kaggle API limits requests per hour for single users.
* **Resolution**:
  - Wait a short period (15–30 minutes) before initiating bulk downloads.
  - Use `--local` flag to analyze data already downloaded to your disk without hitting the API.

---

### 5. `ModuleNotFoundError` or Python Dependency Issues
* **Cause**: Old Python environment or rigid numpy version constraints.
* **Resolution**:
  - Ensure you are on Python 3.9+ (`python --version`).
  - Upgrade pip and reinstall:
    ```bash
    python -m pip install --upgrade pip
    pip install --upgrade kaggle-prep
    ```

---

### 6. Windows Terminal Character Encoding (`UnicodeEncodeError`)
* **Cause**: Legacy Windows cmd/powershell consoles using `cp1252` encoding.
* **Resolution**: `kaggle-prep` 0.3.0+ includes automatic stream reconfiguration and ASCII fallbacks. If running inside custom scripts, set:
  ```powershell
  $env:PYTHONIOENCODING = "utf-8"
  ```

---

##  Development & Contributing

Contributions are welcome! Follow these steps to set up the development environment:

```bash
# 1. Clone the repository
git clone https://github.com/SumitGavali/CLI-DataAnalysis.git
cd kaggle-prep

# 2. Install editable version with test dependencies
pip install -e ".[dev]"

# 3. Run the automated test suite
pytest tests/ -v
```

### Running Tests
```bash
python -m pytest tests/ -v
```

---

##  License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/SumitGavali">Sumit Gavali</a>
</p>
