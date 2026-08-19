import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

import argparse
from pathlib import Path
import os
import profile
import time
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from .notebook import generate_notebook 
from .visualizer import generate_eda_plots, generate_preprocessing_code

# Import our custom profiler and report generator
from .profiler import DataProfiler, save_profile_json, print_profile_summary
from .report import generate_standalone_report


# ============================================================
# FUNCTION 1: Parse Arguments
# ============================================================
def parse_arguments():
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(
        description="Prepare Kaggle datasets for data analysis."
    )
    
    parser.add_argument(
        "dataset",
        help="Kaggle dataset identifier, e.g. adyen/dabstep-benchmark"
    )
    
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to save downloaded data (default: data)"
    )
    parser.add_argument(
        "--notebook",
        action="store_true",
        help="Generate a Jupyter notebook with EDA and preprocessing code"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for debugging"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use existing data - works with --profile, --report, etc"
    )

    
    parser.add_argument(
        "--competition",
        action="store_true",
        help="Download from competition instead of dataset"
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Generate data profile after download"
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate a standalone HTML report after download"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="generate EDA visualization (plots)"
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="generate preprocessing code"
    )
    
    return parser.parse_args()


# ============================================================
# FUNCTION 2: Display Initial Info
# ============================================================
def display_startup_info(args):
    """Display startup information"""
    if args.verbose:
        print(f" Dataset: {args.dataset}")
        print(f" Output directory: {args.output_dir}")
        if args.competition:
            print(f" Competition mode enabled")


# ============================================================
# FUNCTION 3: Check Credentials
# ============================================================
def check_credentials(verbose=False):
    """Check if Kaggle credentials exist"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print(f"  Kaggle credentials not found at: {kaggle_json}")
        print("\n To get credentials:")
        print("   1. Go to https://www.kaggle.com/settings/api")
        print("   2. Click 'Create New Token'")
        print(f"  3. Save kaggle.json to: {kaggle_dir}")
        return None
    
    if verbose:
        print(f" Found Kaggle credentials at: {kaggle_json}")
    
    return kaggle_dir


# ============================================================
# FUNCTION 4: Authenticate
# ============================================================
def authenticate_kaggle(kaggle_dir, verbose=False):
    """Authenticate with Kaggle API"""
    print(f" Authenticating with Kaggle API...")
    
    try:
        os.environ['KAGGLE_CONFIG_DIR'] = str(kaggle_dir)
        api = KaggleApi()
        api.authenticate()
        
        if verbose:
            print(f" Authentication successful!")
        
        return api
        
    except Exception as e:
        print(f"  Authentication error: {e}")
        print("\n Troubleshooting:")
        print("   1. Make sure kaggle.json is not empty")
        print("   2. Try re-downloading from Kaggle settings")
        print("   3. Check your internet connection")
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
    if size_bytes > 1024 * 1024 * 1024:  # GB
        return f"{size_bytes/(1024*1024*1024):.2f} GB"
    elif size_bytes > 1024 * 1024:  # MB
        return f"{size_bytes/(1024*1024):.1f} MB"
    elif size_bytes > 1024:  # KB
        return f"{size_bytes/1024:.1f} KB"
    else:
        return f"{size_bytes} bytes"


# ============================================================
# FUNCTION 7: List Files
# ============================================================
def list_downloaded_files(output_path):
    """List all downloaded files with sizes"""
    print("\n Downloaded files:")
    
    files = list(output_path.rglob("*"))
    if not files:
        print("  No files found!")
        return
    
    for file in files:
        if file.is_file():
            size = file.stat().st_size
            size_str = format_file_size(size)
            print(f"  - {file.name} ({size_str})")


# ============================================================
# FUNCTION 8: Load First CSV
# ============================================================
def load_first_csv(data_path):
    """Load the first CSV file found in the directory"""
    csv_files = list(Path(data_path).rglob("*.csv"))
    
    if not csv_files:
        print(" No CSV files found to profile!")
        return None
    
    try:
        df = pd.read_csv(csv_files[0])
        print(f" Loaded: {csv_files[0].name} ({len(df):,} rows, {len(df.columns)} columns)")
        return df
    except Exception as e:
        print(f" Error loading CSV: {e}")
        return None


# ============================================================
# FUNCTION 9: Download Dataset
# ============================================================
def download_dataset(api, dataset_name, output_path, verbose=False):
    """Download a dataset from Kaggle"""
    print(f"Downloading {dataset_name}...")
    
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
        
    except Exception as e:
        handle_download_error(e, dataset_name)
        return False


# ============================================================
# FUNCTION 10: Download Competition
# ============================================================
def download_competition(api, competition_name, output_path, verbose=False):
    """Download competition files from Kaggle"""
    print(f" Downloading competition: {competition_name}...")
    
    try:
        api.competition_download_files(
            competition_name,
            path=str(output_path)
        )
        
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
    """Handle and explain download errors"""
    print(f" Error downloading: {error}")
    error_msg = str(error)
    
    if "403" in error_msg:
        print("\n Troubleshooting 403 Forbidden error:")
        print("This usually means the dataset requires accepting terms or is restricted.")
        print(f"1. Visit: https://www.kaggle.com/datasets/{dataset_name}")
        print("2. Click 'Download' and accept any terms")
        print("3. Wait a moment, then try again")
        print("\n Or try --competition flag if it's a competition:")
        print(f"  python kaggle_prep/cli.py {dataset_name} --competition --verbose")
        
    elif "404" in error_msg:
        print(f"\n Dataset '{dataset_name}' not found!")
        print("Check the spelling or try searching on Kaggle")
        print("\n Examples:")
        print("  python kaggle_prep/cli.py uciml/iris --verbose")
        print("  python kaggle_prep/cli.py debayank2024/netflix-movies-and-series --verbose")
        
    elif "429" in error_msg:
        print("\n  Rate limit reached!")
        print("Kaggle limits how many requests you can make.")
        print("Wait 1 hour and try again.")
        
    else:
        print(f"\n Unexpected error: {error}")


# ============================================================
# FUNCTION 12: Profile Downloaded Data
# ============================================================
def profile_downloaded_data(output_path, dataset_name, generate_report=False):
    """Run profiling on downloaded data"""
    
    print("\n Generating data profile...")
    
    # Load the data
    df = load_first_csv(Path(output_path))
    if df is None:
        return
    
    # Create profile using our custom profiler
    profiler = DataProfiler(df, dataset_name)
    profile = profiler.profile()
    
    # Save JSON profile
    json_path = save_profile_json(profile)
    
    # Print summary to console
    print_profile_summary(profile)
    
    # Generate HTML report if requested
    if generate_report:
        generate_standalone_report(profile)

def generate_starter_notebook(output_path, dataset_name, df=None, profile=None):
    print("Generating starter notebook...")
    if df is None:
        df=load_first_csv(output_path)

    notebook_path = generate_notebook(
        dataset_name=dataset_name,
        df=df,
        profile=profile,
        output_dir="notebooks"
    )
    print(f" Starter notebook saved to: {notebook_path}")

#===============================================================
def generate_visualizations(output_path, dataset_name,df):
    print("Generating EDA visualizations..")

    vis_dir =f"eda_plots_{dataset_name.replace('/','_')}"

    plot_path = generate_eda_plots(
        df=df,
        output_dir=vis_dir,
        max_cols=10,
        fig_dpi=150
    )
    
    print(f" Visualizations saved to: {plot_path}")
    return plot_path
##==============================================
def save_preprocessing_code(output_path, dataset_name, df):
    """Generate and save preprocessing code"""
    
    print("\n Generating preprocessing code...")
    
    # Generate code
    code = generate_preprocessing_code(df)
    
    # Save to file
    safe_name = dataset_name.replace('/', '_')
    code_path = Path(output_path) / f"{safe_name}_preprocess.py"
    
    with open(code_path, 'w', encoding='utf-8') as f:
        f.write(code)
    
    print(f" Preprocessing code saved to: {code_path}")
    return code_path

# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    """Main entry point for the CLI tool"""
    
    # Step 1: Get user input
    args = parse_arguments()
    
    # Step 2: Show startup info
    display_startup_info(args)
    
    # Step 3: Check if data exists
    output_path = Path(args.output_dir)
    data_exists = any(output_path.glob("*.csv")) or any(output_path.glob("*.xlsx"))
    
    # Step 4: Handle download logic
    if args.local:
        if data_exists:
            print("Using existing data (--local flag detected)")
            print(f"   Data found in: {output_path}")
        else:
            print(f"No data found in {output_path}")
            print("Please download data first (remove --local flag)")
            return
    else:
        # Download normally
        if not data_exists:
            print("No existing data found. Downloading...")
        else:
            print("Downloading data (use --local to skip download next time)")
        
        # Check credentials
        kaggle_dir = check_credentials(args.verbose)
        if not kaggle_dir:
            return
        
        # Authenticate
        api = authenticate_kaggle(kaggle_dir, args.verbose)
        if not api:
            return
        
        # Create output directory
        create_output_directory(args.output_dir)
        
        # Download
        if args.competition:
            download_competition(api, args.dataset, output_path, args.verbose)
        else:
            download_dataset(api, args.dataset, output_path, args.verbose)
    
    # Step 5: Load data for profiling and visualization
    df = None
    profile = None
    
    if args.profile or args.report or args.visualize or args.preprocess or args.notebook:
        df = load_first_csv(Path(args.output_dir))
        
        if df is not None:
            # Generate profile if requested
            if args.profile or args.report:
                profiler = DataProfiler(df, args.dataset)
                profile = profiler.profile()
                
                if args.profile:
                    json_path = save_profile_json(profile)
                
                if args.report:
                    generate_standalone_report(profile)
            
            # Generate visualizations if requested
            if args.visualize:
                generate_visualizations(output_path, args.dataset, df)
            
            # Generate preprocessing code if requested
            if args.preprocess:
                save_preprocessing_code(output_path, args.dataset, df)
            
            # Generate notebook if requested
            if args.notebook:
                generate_starter_notebook(args.dataset, df, profile)
    
    # Step 6: Done!
    print("\nAll done! Happy data science!")

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()