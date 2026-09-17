import time
import pytest
import pandas as pd
import numpy as np
from kaggle_prep.diff import (
    compute_column_hash,
    compute_dataframe_column_hashes,
    diff_snapshots,
    format_diff_cli_table,
    save_snapshot_json,
    load_snapshot_json,
)
from kaggle_prep.profiler import DataProfiler


def test_diff_snapshots_shifted_distribution(tmp_path):
    np.random.seed(42)
    df_old = pd.DataFrame({
        "col_same": range(100),
        "col_shift": np.random.normal(10, 2, 100),
        "col_dropped": [1] * 100
    })

    df_new = pd.DataFrame({
        "col_same": range(100),
        "col_shift": np.random.normal(50, 2, 100),  # Deliberate distribution shift
        "col_added": ["X"] * 100
    })

    prof_old = DataProfiler(df_old, "v1").profile()
    prof_new = DataProfiler(df_new, "v2").profile()

    diff_res = diff_snapshots(prof_old, prof_new)

    assert "col_added" in diff_res["added_columns"]
    assert "col_dropped" in diff_res["dropped_columns"]

    modified_col_names = [c["column"] for c in diff_res["modified_columns"]]
    assert "col_shift" in modified_col_names

    shifted_item = next(c for c in diff_res["modified_columns"] if c["column"] == "col_shift")
    assert shifted_item["distribution_shifted"] is True

    table_text = format_diff_cli_table(diff_res)
    assert "*SHIFT*" in table_text
    assert "col_added" in table_text


def test_incremental_update_profiling_benchmark():
    np.random.seed(42)
    # Generate 500,000 rows x 5 columns for noticeable execution time
    n_rows = 200_000
    df = pd.DataFrame({
        f"col_{i}": np.random.randn(n_rows) for i in range(5)
    })

    # Initial profile
    t0 = time.perf_counter()
    p1 = DataProfiler(df, "big_ds").profile()
    t_full = time.perf_counter() - t0

    # Second profile with --update (reuse cached hashes/stats for unchanged columns)
    t0 = time.perf_counter()
    p2 = DataProfiler(df, "big_ds", prior_profile=p1).profile()
    t_inc = time.perf_counter() - t0

    # Incremental update should be significantly faster because statistics calculation is bypassed
    assert t_inc < t_full
    assert p1["columns"] == p2["columns"]
