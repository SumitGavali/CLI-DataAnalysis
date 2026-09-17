"""
Kaggle Prep - One-command Kaggle dataset preparation & EDA toolkit.
"""

__version__ = "0.3.0"
__author__ = "Sumit Gavali"
__email__ = "sumitrg0007@gmail.com"

from .cli import main
from .profiler import DataProfiler, save_profile_json, print_profile_summary
from .report import generate_standalone_report
from .visualizer import generate_eda_plots, generate_preprocessing_code
from .notebook import generate_notebook
from .ingestion import get_source_adapter

__all__ = [
    "main",
    "DataProfiler",
    "save_profile_json",
    "print_profile_summary",
    "generate_standalone_report",
    "generate_eda_plots",
    "generate_preprocessing_code",
    "generate_notebook",
    "get_source_adapter",
    "__version__",
]