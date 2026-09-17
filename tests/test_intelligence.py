import json
import pytest
import pandas as pd
import numpy as np
from kaggle_prep.intelligence import (
    detect_task_type,
    analyze_class_balance,
    recommend_metrics,
    determine_split_strategy,
)
from kaggle_prep.notebook import generate_notebook


def test_binary_classification_task():
    df = pd.DataFrame({
        "feature1": range(100),
        "target": [0] * 90 + [1] * 10
    })
    task_type = detect_task_type(df, "target")
    assert task_type == "binary_classification"

    class_info = analyze_class_balance(df, "target")
    assert class_info["is_imbalanced"] is True
    assert class_info["minority_pct"] == 10.0
    assert class_info["counts"] == {"0": 90, "1": 10}

    metrics = recommend_metrics(task_type, class_info)
    assert "ROC-AUC" in metrics
    assert "F1-Score (minority)" in metrics

    split = determine_split_strategy(task_type)
    assert split["type"] == "stratified"
    assert "stratify=y" in split["code"]


def test_multiclass_classification_task():
    df = pd.DataFrame({
        "feature1": range(100),
        "target": ["A"] * 34 + ["B"] * 33 + ["C"] * 33
    })
    task_type = detect_task_type(df, "target")
    assert task_type == "multiclass_classification"

    class_info = analyze_class_balance(df, "target")
    assert class_info["is_imbalanced"] is False

    metrics = recommend_metrics(task_type, class_info)
    assert "Accuracy" in metrics
    assert "Macro-F1" in metrics


def test_regression_task():
    np.random.seed(42)
    df = pd.DataFrame({
        "feature1": range(100),
        "target": np.random.randn(100) * 10
    })
    task_type = detect_task_type(df, "target")
    assert task_type == "regression"

    metrics = recommend_metrics(task_type)
    assert "RMSE" in metrics
    assert "MAE" in metrics

    split = determine_split_strategy(task_type)
    assert split["type"] == "random"


def test_time_series_task():
    dates = pd.date_range("2023-01-01", periods=100)
    df = pd.DataFrame({
        "date": dates,
        "feature1": range(100),
        "target": np.sin(np.linspace(0, 10, 100))
    })
    task_type = detect_task_type(df, "target")
    assert task_type == "time_series"

    split = determine_split_strategy(task_type)
    assert split["type"] == "time_based"
    assert "shuffle=False" in split["code"]


def test_notebook_split_strategy_matching(tmp_path):
    df = pd.DataFrame({
        "feature1": range(50),
        "label": [0] * 40 + [1] * 10
    })
    nb_path = generate_notebook(
        dataset_name="test_dataset",
        df=df,
        target="label",
        output_dir=str(tmp_path)
    )
    assert nb_path.exists()

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    # Combine all code cells content into one string
    code_text = ""
    for cell in nb_data["cells"]:
        if cell["cell_type"] == "code":
            code_text += "".join(cell["source"])

    assert "stratify=y" in code_text
