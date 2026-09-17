import os
import sqlite3
import pytest
import pandas as pd
from kaggle_prep.ingestion import (
    get_source_adapter,
    FileSourceAdapter,
    DBSourceAdapter,
    KaggleSourceAdapter,
    S3SourceAdapter,
)


def test_file_adapter_csv(tmp_path):
    csv_file = tmp_path / "test.csv"
    df_orig = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df_orig.to_csv(csv_file, index=False)

    adapter = get_source_adapter(file_path=str(csv_file))
    assert isinstance(adapter, FileSourceAdapter)
    assert adapter.dataset_name == "test"

    df_loaded = adapter.load_dataframe()
    pd.testing.assert_frame_equal(df_loaded, df_orig)


def test_file_adapter_parquet(tmp_path):
    pq_file = tmp_path / "test.parquet"
    df_orig = pd.DataFrame({"col1": [1.0, 2.5], "col2": [10, 20]})
    df_orig.to_parquet(pq_file, index=False)

    adapter = get_source_adapter(file_path=str(pq_file))
    assert isinstance(adapter, FileSourceAdapter)
    assert adapter.dataset_name == "test"

    df_loaded = adapter.load_dataframe()
    pd.testing.assert_frame_equal(df_loaded, df_orig)


def test_db_adapter_sqlite(tmp_path):
    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(db_file)
    df_orig = pd.DataFrame({"id": [101, 102], "val": [9.9, 8.8]})
    df_orig.to_sql("my_table", conn, index=False)
    conn.close()

    conn_str = f"sqlite:///{db_file}"
    adapter = get_source_adapter(db_conn=conn_str, table="my_table")
    assert isinstance(adapter, DBSourceAdapter)
    assert adapter.dataset_name == "my_table"

    df_loaded = adapter.load_dataframe()
    pd.testing.assert_frame_equal(df_loaded, df_orig)


def test_missing_file_error():
    with pytest.raises(FileNotFoundError):
        get_source_adapter(file_path="non_existent_file.csv")


def test_unsupported_file_extension(tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("hello world")

    with pytest.raises(ValueError, match="Unsupported file format"):
        get_source_adapter(file_path=str(txt_file))


def test_mutually_exclusive_sources(tmp_path):
    csv_file = tmp_path / "test.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="Mutually exclusive input sources"):
        get_source_adapter(dataset="uciml/iris", file_path=str(csv_file))


def test_no_source_provided():
    with pytest.raises(ValueError, match="No input source provided"):
        get_source_adapter()
