import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import json

from kaggle_prep.profiler import DataProfiler, save_profile_json, print_profile_summary


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        "num_a": [1.0, 2.0, 3.0, 4.0, 100.0, None],
        "num_b": [10, 20, 30, 40, 50, 60],
        "cat_a": ["cat", "dog", "cat", "bird", "dog", "cat"],
        "const_col": [1, 1, 1, 1, 1, 1],
    })


def test_profiler_basic_stats(sample_df):
    profiler = DataProfiler(sample_df, "test/dataset")
    profile = profiler.profile()

    assert profile["dataset"] == "test/dataset"
    assert "basic_stats" in profile
    assert profile["basic_stats"]["rows"] == 6
    assert profile["basic_stats"]["columns"] == 4
    assert profile["basic_stats"]["missing_cells"] == 1
    assert profile["basic_stats"]["num_cols"] >= 2


def test_profiler_column_stats(sample_df):
    profiler = DataProfiler(sample_df, "test/dataset")
    profile = profiler.profile()

    assert "columns" in profile
    assert "num_a" in profile["columns"]
    assert profile["columns"]["num_a"]["missing"] == 1
    assert "mean" in profile["columns"]["num_a"]
    assert "outliers" in profile["columns"]["num_a"]

    assert "cat_a" in profile["columns"]
    assert "top_values" in profile["columns"]["cat_a"]


def test_profiler_warnings_and_suggestions(sample_df):
    profiler = DataProfiler(sample_df, "test/dataset")
    profile = profiler.profile()

    assert isinstance(profile["warnings"], list)
    assert isinstance(profile["suggestions"], list)
    # Check constant column warning
    const_warn = any("const_col" in w for w in profile["warnings"])
    assert const_warn


def test_save_profile_json(sample_df, tmp_path):
    profiler = DataProfiler(sample_df, "test/dataset")
    profile = profiler.profile()

    json_file = save_profile_json(profile, output_dir=str(tmp_path))
    assert Path(json_file).exists()

    with open(json_file, "r") as f:
        loaded = json.load(f)
    assert loaded["dataset"] == "test/dataset"
