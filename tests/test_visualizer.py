import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from kaggle_prep.visualizer import generate_eda_plots, generate_preprocessing_code


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "feature_1": np.random.normal(10, 2, n),
        "feature_2": np.random.uniform(0, 100, n),
        "category": np.random.choice(["A", "B", "C"], n),
        "target": np.random.choice([0, 1], n),
    })


def test_generate_eda_plots(sample_df, tmp_path):
    output_dir = tmp_path / "test_plots"
    result_path = generate_eda_plots(
        sample_df,
        output_dir=str(output_dir),
        max_cols=5,
        fig_dpi=72,
        fig_format="png",
        target="target"
    )

    assert Path(result_path).exists()
    png_files = list(Path(result_path).glob("*.png"))
    assert len(png_files) >= 5


def test_generate_preprocessing_code(sample_df):
    code = generate_preprocessing_code(sample_df)
    assert isinstance(code, str)
    assert "import pandas as pd" in code
    assert "StandardScaler" in code
