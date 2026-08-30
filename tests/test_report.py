import pandas as pd
import pytest
from pathlib import Path

from kaggle_prep.profiler import DataProfiler
from kaggle_prep.report import generate_standalone_report


def test_generate_standalone_report(tmp_path):
    df = pd.DataFrame({
        "age": [25, 30, 35, 40],
        "salary": [50000.0, 60000.0, 75000.0, 90000.0],
        "dept": ["HR", "Engineering", "Engineering", "Marketing"]
    })

    profiler = DataProfiler(df, "company/salaries")
    profile = profiler.profile()

    report_path = generate_standalone_report(profile, output_dir=str(tmp_path))

    assert Path(report_path).exists()
    assert report_path.suffix == ".html"

    with open(report_path, "r", encoding="utf-8") as f:
        html = f.read()

    assert "company/salaries" in html
    assert "Data Profile Report" in html
    assert "Column Analysis" in html
