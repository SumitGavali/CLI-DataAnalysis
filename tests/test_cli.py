import pytest
import pandas as pd
from pathlib import Path
import sys

from kaggle_prep.cli import (
    parse_arguments,
    format_file_size,
    load_first_csv,
    create_output_directory
)


def test_format_file_size():
    assert format_file_size(500) == "500 bytes"
    assert format_file_size(2048) == "2.0 KB"
    assert format_file_size(1024 * 1024 * 5) == "5.0 MB"
    assert format_file_size(1024 * 1024 * 1024 * 2) == "2.00 GB"


def test_create_output_directory(tmp_path):
    target = tmp_path / "subdir" / "nested"
    created = create_output_directory(str(target))
    assert Path(created).exists()
    assert Path(created).is_dir()


def test_load_first_csv(tmp_path):
    df_orig = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
    csv_file = tmp_path / "sample.csv"
    df_orig.to_csv(csv_file, index=False)

    loaded = load_first_csv(tmp_path)
    assert loaded is not None
    assert len(loaded) == 3
    assert list(loaded.columns) == ["col_a", "col_b"]


def test_load_first_csv_none(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    loaded = load_first_csv(empty_dir)
    assert loaded is None
