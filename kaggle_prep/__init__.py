__version__="0.1.0"
__author__="Sumit Gavali"
__email__="Sumitrg0007@gmail.com"

from .cli import main
from .profiler import DataProfiler
from .report import generate_standalone_report
from .visualizer import generate_eda_plots
from .notebook import generate_notebook

__version__ = "0.1.0"
__all__ = [
    'main',
    'DataProfiler',
    'generate_standalone_report',
    'generate_eda_plots',
    'generate_notebook'
]