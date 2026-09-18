import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

import argparse
import json
import os
import shutil
import sys

# Ensure UTF-8 output encoding where supported, fallback gracefully
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pathlib import Path
from typing import Optional, Tuple, List, Dict
import time

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from .notebook import generate_notebook
from .profiler import DataProfiler, print_profile_summary, save_profile_json
from .report import generate_standalone_report
from .visualizer import generate_eda_plots, generate_preprocessing_code
from .ingestion import get_source_adapter
from .intelligence import (
    detect_task_type,
    analyze_class_balance,
    recommend_metrics,
    determine_split_strategy,
    format_intelligence_summary,
)
from .audit import DataAuditor, format_audit_summary
from .env_pin import generate_pinned_requirements, generate_conda_environment, check_environment, format_env_check_summary
from .diff import save_snapshot_json, load_snapshot_json, diff_snapshots, format_diff_cli_table
from .pipeline_exporter import export_pipeline_code


# ============================================================
# FUNCTION 1: Parse Arguments
# ============================================================
def parse_arguments():
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(
        prog="kaggle-prep",
        description="One-command Kaggle dataset preparation, profiling, visualization & starter notebook generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Kaggle dataset identifier, e.g. 'uciml/iris' or competition name"
    )

    parser.add_argument(
        "--file", "-f",
        default=None,
        help="Path to local CSV, Parquet, JSON, Excel, or TSV file"
    )

    parser.add_argument(
        "--db",
        default=None,
        help="SQLAlchemy database connection string (e.g., 'sqlite:///data.db')"
    )

    parser.add_argument(
        "--query",
        default=None,
        help="SQL query to execute against database"
    )

    parser.add_argument(
        "--table",
        default=None,
        help="Database table name to query"
    )

    parser.add_argument(
        "--s3",
        default=None,
        help="S3 object URI (e.g., 's3://bucket/data.parquet')"
    )

    parser.add_argument(
        "--setup",
        action="store_true",
        help="Interactive wizard to set up or verify Kaggle API credentials"
    )

    parser.add_argument(
        "--version", "-v",
        action="version",
        version="%(prog)s 0.5.0",
        help="Show program version number and exit"
    )

    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run complete pipeline: profile, report, visualize, preprocess, and notebook"
    )

    parser.add_argument(
        "--output-dir", "-o",
        default="data",
        help="Directory to save downloaded data"
    )

    parser.add_argument(
        "--local", "-l",
        action="store_true",
        help="Use existing local data in output directory instead of downloading"
    )

    parser.add_argument(
        "--competition", "-c",
        action="store_true",
        help="Download from Kaggle competition instead of dataset"
    )

    parser.add_argument(
        "--profile", "-p",
        action="store_true",
        help="Generate data profile JSON and console summary"
    )

    parser.add_argument(
        "--report", "-r",
        action="store_true",
        help="Generate a standalone HTML report"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate EDA visualization plots"
    )

    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Generate auto-preprocessing Python script"
    )

    parser.add_argument(
        "--notebook", "-n",
        action="store_true",
        help="Generate a starter Jupyter notebook"
    )

    parser.add_argument(
        "--target", "-t",
        default=None,
        help="Target column name for supervised EDA & modeling"
    )

    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run active data quality and leakage audit"
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code if critical audit issues are detected"
    )

    parser.add_argument(
        "--test-file",
        default=None,
        help="Path to test dataset file for train/test leakage and drift auditing"
    )

    parser.add_argument(
        "--conda",
        action="store_true",
        help="Generate Conda environment.yml alongside requirements.txt"
    )

    parser.add_argument(
        "--check-env",
        default=None,
        help="Compare current environment against a pinned requirements.txt file"
    )

    parser.add_argument(
        "--update",
        action="store_true",
        help="Perform incremental profiling by reusing cached column stats for unchanged data"
    )

    parser.add_argument(
        "--export-pipeline",
        action="store_true",
        help="Export baseline model as reusable, unit-testable Python pipeline module"
    )

    parser.add_argument(
        "--max-cols", "-m",
        type=int,
        default=10,
        help="Max numeric/categorical columns to plot"
    )

    parser.add_argument(
        "--dpi", "-d",
        type=int,
        default=150,
        help="Figure DPI resolution for plots"
    )

    parser.add_argument(
        "--fig-format", "-f",
        choices=["png", "pdf", "svg", "jpg"],
        default="png",
        help="Output format for visualization plots"
    )

    parser.add_argument(
        "--sample", "-s",
        type=int,
        default=None,
        help="Sample N rows from data for faster processing"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for debugging"
    )

    return parser.parse_args()


# ============================================================
# FUNCTION 2: Interactive Credentials Setup Wizard
# ============================================================
def run_interactive_setup():
    """Interactive wizard to configure ~/.kaggle/kaggle.json"""
    print("\n" + "=" * 65)
    print("  Kaggle API Credentials Setup Wizard")
    print("=" * 65)

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if kaggle_json.exists():
        print(f"\n Found existing credentials at: {kaggle_json}")
        try:
            with open(kaggle_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            username = data.get("username", "unknown")
            print(f" Configured username: {username}")

            print(" Testing authentication with Kaggle API...")
            api = KaggleApi()
            api.authenticate()
            print(" Authentication successful! Your Kaggle CLI is ready to use.\n")
            return True
        except Exception as e:
            print(f" Existing credentials failed authentication: {e}")
            choice = input("\n Would you like to reconfigure your credentials? [y/N]: ").strip().lower()
            if choice != "y":
                return False

    print("\n To get your Kaggle API key:")
    print("   1. Log into https://www.kaggle.com")
    print("   2. Go to 'Account Settings' -> https://www.kaggle.com/settings/api")
    print("   3. Scroll to 'API' section and click 'Create New Token'")
    print("   4. A file named 'kaggle.json' will download with your username and key.\n")

    username = input(" Enter your Kaggle Username: ").strip()
    if not username:
        print(" Setup cancelled: Username cannot be empty.")
        return False

    key = input(" Enter your Kaggle API Key: ").strip()
    if not key:
        print(" Setup cancelled: API key cannot be empty.")
        return False

    try:
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        cred_data = {"username": username, "key": key}
        with open(kaggle_json, "w", encoding="utf-8") as f:
            json.dump(cred_data, f, indent=2)

        # Set secure permissions on POSIX systems
        if os.name != "nt":
            try:
                os.chmod(kaggle_json, 0o600)
            except Exception:
                pass

        print(f"\n Credentials successfully saved to: {kaggle_json}")

        print(" Validating with Kaggle API...")
        os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_dir)
        api = KaggleApi()
        api.authenticate()
        print(" Authentication verified! You are ready to download datasets.\n")
        return True

    except Exception as e:
        print(f"\n Setup failed during validation: {e}")
        print(" Please double-check your username and API key on Kaggle.")
        return False


# ============================================================
# FUNCTION 3: Check Credentials
# ============================================================
def check_credentials(verbose=False):
    """Check if Kaggle credentials exist"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if not kaggle_json.exists():
        if verbose:
            print(f" Kaggle credentials not found at: {kaggle_json}")
        return None

    if verbose:
        print(f" Found Kaggle credentials at: {kaggle_json}")

    return kaggle_dir


# ============================================================
# FUNCTION 4: Authenticate
# ============================================================
def authenticate_kaggle(kaggle_dir, verbose=False):
    """Authenticate with Kaggle API"""
    if verbose:
        print(" Authenticating with Kaggle API...")

    try:
        if kaggle_dir:
            os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_dir)
        api = KaggleApi()
        api.authenticate()

        if verbose:
            print(" Authentication successful!")

        return api

    except Exception as e:
        print(f" Authentication error: {e}")
        print("\n Troubleshooting:")
        print("   1. Run `kaggle-prep --setup` to configure your credentials interactively.")
        print("   2. Make sure kaggle.json is not empty.")
        print("   3. Check your internet connection.")
        return None


# ============================================================
# FUNCTION 5: Create Output Directory
# ============================================================
def create_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


# ============================================================
# FUNCTION 6: Format File Size
# ============================================================
def format_file_size(size_bytes):
    """Convert bytes to human-readable format"""
    if size_bytes > 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    elif size_bytes > 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    elif size_bytes > 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes} bytes"


# ============================================================
# FUNCTION 7: List Files
# ============================================================
def list_downloaded_files(output_path):
    """List all downloaded files with sizes"""
    print("\n Downloaded files:")

    files = [f for f in Path(output_path).rglob("*") if f.is_file()]
    if not files:
        print("  No files found!")
        return

    for file in files:
        size = file.stat().st_size
        size_str = format_file_size(size)
        print(f"  - {file.name} ({size_str})")


# ============================================================
# FUNCTION 8: Load Dataset File (Multi-format support)
# ============================================================
def load_first_csv(data_path, sample: Optional[int] = None):
    """Load the first tabular dataset file found in the directory"""
    data_dir = Path(data_path)

    # Search for supported extensions in priority order
    extensions = ["*.csv", "*.parquet", "*.tsv", "*.xlsx", "*.json"]
    found_files = []
    for ext in extensions:
        found_files.extend(list(data_dir.rglob(ext)))

    if not found_files:
        print(f" No tabular data files (*.csv, *.parquet, *.tsv, *.xlsx, *.json) found in {data_path}!")
        return None

    target_file = found_files[0]
    suffix = target_file.suffix.lower()

    try:
        if suffix == ".csv":
            df = pd.read_csv(target_file, low_memory=False)
        elif suffix == ".tsv":
            df = pd.read_csv(target_file, sep="\t", low_memory=False)
        elif suffix == ".parquet":
            df = pd.read_parquet(target_file)
        elif suffix == ".xlsx":
            df = pd.read_excel(target_file)
        elif suffix == ".json":
            df = pd.read_json(target_file)
        else:
            df = pd.read_csv(target_file)

        if sample and sample < len(df):
            print(f" Sampling {sample:,} rows from {len(df):,} rows...")
            df = df.sample(sample, random_state=42).reset_index(drop=True)

        print(f" Loaded: {target_file.name} ({len(df):,} rows, {len(df.columns)} columns)")
        return df

    except Exception as e:
        print(f" Error loading {target_file.name}: {e}")
        return None


# ============================================================
# FUNCTION 9: Download Dataset (Zero-Config First)
# ============================================================
def download_dataset(api, dataset_name, output_path, verbose=False):
    """Download dataset - tries zero-config kagglehub first, falls back to Kaggle API"""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Method 1: Try kagglehub (Zero-config, no auth required for public datasets)
    try:
        import kagglehub
        print(f" Downloading '{dataset_name}' via kagglehub (Zero-Config mode)...")

        path = kagglehub.dataset_download(dataset_name)
        source_path = Path(path)

        if source_path.is_file():
            shutil.copy(source_path, output_path / source_path.name)
        else:
            for file in source_path.rglob("*"):
                if file.is_file():
                    dest = output_path / file.name
                    shutil.copy(file, dest)

        print(f" Download complete! Files saved to: {output_path}")
        if verbose:
            list_downloaded_files(output_path)
        return True

    except Exception as e:
        if verbose:
            print(f" Direct kagglehub download notice: {e}")
        print(" Falling back to authenticated Kaggle API download...")

        # Method 2: Authenticated Kaggle API
        if not api:
            kaggle_dir = check_credentials(verbose)
            if kaggle_dir:
                api = authenticate_kaggle(kaggle_dir, verbose)

        if not api:
            print("\n Authenticated download requires Kaggle credentials.")
            print(" Run `kaggle-prep --setup` to configure your API key.")
            return False

        try:
            api.dataset_download_files(
                dataset_name,
                path=str(output_path),
                unzip=True
            )
            print(f" Download complete! Files saved to: {output_path}")
            if verbose:
                list_downloaded_files(output_path)
            return True
        except Exception as e2:
            handle_download_error(e2, dataset_name)
            return False


# ============================================================
# FUNCTION 10: Download Competition
# ============================================================
def download_competition(api, competition_name, output_path, verbose=False):
    """Download competition files from Kaggle"""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if not api:
        kaggle_dir = check_credentials(verbose)
        if not kaggle_dir:
            print("\n Competition downloads require Kaggle credentials.")
            print(" Run `kaggle-prep --setup` to configure your credentials.")
            return False
        api = authenticate_kaggle(kaggle_dir, verbose)
        if not api:
            return False

    print(f" Downloading competition files for: {competition_name}...")

    try:
        api.competition_download_files(
            competition_name,
            path=str(output_path)
        )

        # Unzip any downloaded archives
        for zip_file in output_path.glob("*.zip"):
            import zipfile
            try:
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(output_path)
            except Exception:
                pass

        print(f" Download complete! Files saved to: {output_path}")

        if verbose:
            list_downloaded_files(output_path)

        return True

    except Exception as e:
        handle_download_error(e, competition_name)
        return False


# ============================================================
# FUNCTION 11: Handle Download Errors
# ============================================================
def handle_download_error(error, dataset_name):
    """Handle and explain download errors clearly"""
    print(f"\n Error downloading: {error}")
    error_msg = str(error)

    if "401" in error_msg or "Unauthorized" in error_msg:
        print("\n Troubleshooting 401 Unauthorized:")
        print(" 1. Your Kaggle API token may be expired or invalid.")
        print(" 2. Run `kaggle-prep --setup` to re-enter your credentials.")

    elif "403" in error_msg or "Forbidden" in error_msg:
        print("\n Troubleshooting 403 Forbidden:")
        print(" This dataset/competition requires accepting competition rules or terms.")
        print(f" 1. Visit: https://www.kaggle.com/datasets/{dataset_name} (or competitions/{dataset_name})")
        print(" 2. Click 'Download' / 'Join Competition' and accept the rules.")
        print(" 3. Try running kaggle-prep again.")

    elif "404" in error_msg or "Not Found" in error_msg:
        print(f"\n Dataset or competition '{dataset_name}' not found!")
        print(" Check for typos or search Kaggle for the exact identifier (format: 'owner/dataset-name').")
        print("\n Examples:")
        print("   kaggle-prep uciml/iris --all")
        print("   kaggle-prep debayank2024/netflix-movies-and-series --all")
        print("   kaggle-prep titanic --competition --all")

    elif "429" in error_msg or "Too Many Requests" in error_msg:
        print("\n Rate limit reached!")
        print(" Kaggle limits the number of API requests per hour. Please wait a bit and try again.")

    else:
        print("\n Unexpected error occurred.")
        print(f" Details: {error}")


# ============================================================
# FUNCTION 12: Profiling, Reporting, Visualization & Notebook Helpers
# ============================================================
def run_profiling(df: pd.DataFrame, dataset_name: str, generate_report: bool = False):
    """Run profiling on data"""
    print("\n Generating data profile...")
    profiler = DataProfiler(df, dataset_name)
    profile = profiler.profile()

    json_path = save_profile_json(profile)
    print_profile_summary(profile)

    if generate_report:
        generate_standalone_report(profile)

    return profile


def generate_starter_notebook(output_path, dataset_name: str, df=None, profile=None, target: Optional[str] = None, conda: bool = False):
    """Generate starter Jupyter notebook and pinned requirements"""
    print("\n Generating starter Jupyter notebook...")
    if df is None:
        df = load_first_csv(output_path)

    notebook_path = generate_notebook(
        dataset_name=dataset_name,
        df=df,
        profile=profile,
        target=target,
        output_dir="notebooks"
    )
    print(f" Starter notebook saved to: {notebook_path}")

    # Generate reproducible environment files
    req_path = generate_pinned_requirements(output_dir="notebooks")
    print(f" Pinned requirements saved to: {req_path}")
    if conda:
        conda_path = generate_conda_environment(output_dir="notebooks")
        print(f" Conda environment file saved to: {conda_path}")

    return notebook_path


def generate_visualizations(output_path, dataset_name: str, df: pd.DataFrame,
                            target: Optional[str] = None, max_cols: int = 10,
                            fig_dpi: int = 150, fig_format: str = "png"):
    """Generate EDA visualizations"""
    safe_name = dataset_name.replace("/", "_")
    vis_dir = f"eda_plots_{safe_name}"

    plot_path = generate_eda_plots(
        df=df,
        output_dir=vis_dir,
        max_cols=max_cols,
        fig_dpi=fig_dpi,
        fig_format=fig_format,
        target=target
    )
    print(f" Visualizations saved to: {plot_path}")
    return plot_path


def save_preprocessing_code(output_path, dataset_name: str, df: pd.DataFrame):
    """Generate and save preprocessing Python script"""
    print("\n Generating preprocessing pipeline code...")
    code = generate_preprocessing_code(df)

    safe_name = dataset_name.replace("/", "_")
    code_path = Path(output_path) / f"{safe_name}_preprocess.py"

    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code)

    print(f" Preprocessing code saved to: {code_path}")
    return code_path


# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    """Main entry point for the CLI tool"""
    # Check for `kaggle-prep diff <old> <new>` subcommand
    if len(sys.argv) >= 4 and sys.argv[1] == "diff":
        old_path = sys.argv[2]
        new_path = sys.argv[3]
        try:
            old_snap = load_snapshot_json(old_path)
            new_snap = load_snapshot_json(new_path)
            diff_res = diff_snapshots(old_snap, new_snap)
            print("\n" + format_diff_cli_table(diff_res) + "\n")
            return
        except Exception as e:
            print(f" Error running diff: {e}")
            sys.exit(1)

    args = parse_arguments()

    # Handle --check-env mode
    if args.check_env:
        try:
            res = check_environment(args.check_env)
            print("\n" + format_env_check_summary(res) + "\n")
            if res["has_drift"]:
                sys.exit(1)
            return
        except Exception as e:
            print(f" Error checking environment: {e}")
            sys.exit(1)

    # Check if any input source was provided
    has_input = bool(args.dataset or args.file or args.db or args.s3)
    if not has_input:
        print("\n" + "=" * 60)
        print("  Kaggle Prep - One-Command Dataset Preparation & EDA")
        print("=" * 60)
        print("\n Usage:")
        print("   kaggle-prep <dataset_identifier> [options]")
        print("   kaggle-prep --file <path> [options]")
        print("   kaggle-prep --db <conn_str> --table <table> [options]")
        print("   kaggle-prep --s3 <s3_uri> [options]")
        print("\n Common Commands:")
        print("   kaggle-prep uciml/iris --all")
        print("   kaggle-prep --file data.csv --all")
        print("   kaggle-prep --setup")
        print("\n Run `kaggle-prep --help` for full options list.")
        return

    # Handle --all flag
    if args.all:
        args.profile = True
        args.report = True
        args.visualize = True
        args.preprocess = True
        args.notebook = True

    # Default action if no flags provided: profile and report
    has_action = (args.profile or args.report or args.visualize or
                  args.preprocess or args.notebook)
    if not has_action and not args.local:
        # Default behavior: profile + report
        args.profile = True
        args.report = True

    try:
        adapter = get_source_adapter(
            dataset=args.dataset,
            file_path=args.file,
            db_conn=args.db,
            query=args.query,
            table=args.table,
            s3_uri=args.s3,
            output_dir=args.output_dir,
            competition=args.competition,
            local=args.local,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f" Ingestion Error: {e}")
        sys.exit(1)

    dataset_name = adapter.dataset_name
    if args.verbose:
        print(f" Source dataset: {dataset_name}")
        print(f" Output directory: {args.output_dir}")
        if args.target:
            print(f" Target column: {args.target}")

    output_path = Path(args.output_dir)

    try:
        df = adapter.load_dataframe(sample=args.sample)
    except Exception as e:
        print(f" Error loading data: {e}")
        sys.exit(1)

    profile = None
    audit_results = None

    if df is not None:
        # Target intelligence analysis
        if args.target and args.target in df.columns:
            task_type = detect_task_type(df, args.target)
            class_info = analyze_class_balance(df, args.target) if "classification" in task_type else None
            metrics = recommend_metrics(task_type, class_info)
            split_info = determine_split_strategy(task_type)
            summary_str = format_intelligence_summary(args.target, task_type, class_info, metrics, split_info)
            print("\n" + summary_str + "\n")
        elif args.target:
            print(f" Warning: Target column '{args.target}' not found in dataframe.")

        # Data Quality & Leakage Audit
        if args.audit or args.strict:
            test_df = None
            if args.test_file:
                try:
                    test_adapter = get_source_adapter(file_path=args.test_file)
                    test_df = test_adapter.load_dataframe()
                except Exception as e:
                    print(f" Warning: Failed to load test file for audit: {e}")

            auditor = DataAuditor(df, target_col=args.target, test_df=test_df)
            audit_results = auditor.audit()
            print("\n" + format_audit_summary(audit_results) + "\n")

        # Generate profile & report if requested
        if args.profile or args.report:
            prior_profile = None
            snapshot_path = Path("data_profiles") / f"{dataset_name.replace('/', '_')}_snapshot.json"
            if args.update and snapshot_path.exists():
                try:
                    prior_profile = load_snapshot_json(str(snapshot_path))
                    print(f" Incremental profiling: Loaded prior snapshot from {snapshot_path}")
                except Exception:
                    pass

            profiler = DataProfiler(df, dataset_name, prior_profile=prior_profile)
            profile = profiler.profile()
            save_snapshot_json(profile, str(snapshot_path))

            if args.profile:
                save_profile_json(profile)
                print_profile_summary(profile)

            if args.report:
                generate_standalone_report(profile, audit_results=audit_results)

        # Generate visualizations if requested
        if args.visualize:
            generate_visualizations(
                output_path=output_path,
                dataset_name=dataset_name,
                df=df,
                target=args.target,
                max_cols=args.max_cols,
                fig_dpi=args.dpi,
                fig_format=args.fig_format
            )

        # Generate preprocessing code if requested
        if args.preprocess:
            save_preprocessing_code(output_path, dataset_name, df)

        # Generate starter notebook if requested
        if args.notebook:
            generate_starter_notebook(
                output_path=output_path,
                dataset_name=dataset_name,
                df=df,
                profile=profile,
                target=args.target,
                conda=args.conda
            )

        # Export reusable pipeline code if requested
        if args.export_pipeline:
            print("\n Exporting baseline scikit-learn pipeline code...")
            p_py, t_py = export_pipeline_code(
                df=df,
                target_col=args.target,
                task_type=task_type if 'task_type' in locals() else None,
                output_dir="src"
            )
            print(f" Reusable pipeline module saved to: {p_py}")
            print(f" Pipeline unit tests saved to: {t_py}")
    else:
        print(" Skipping downstream EDA tasks (no valid tabular data found).")

    # Step 6: Completion Banner
    print("\n" + "=" * 60)
    print(" All tasks completed successfully! Happy data science!")
    print("=" * 60 + "\n")

    if args.strict and audit_results and audit_results.get("has_critical"):
        print(" Strict mode enabled: Critical audit issues detected! Exiting with status 1.")
        sys.exit(1)


if __name__ == "__main__":
    main()