import pytest
import pandas as pd
import numpy as np
from kaggle_prep.audit import DataAuditor, format_audit_summary


def test_audit_injected_leakage_and_id():
    np.random.seed(42)
    n_rows = 100
    target = np.random.randint(0, 2, n_rows)

    df = pd.DataFrame({
        "target": target,
        "leakage_col": target,  # Perfect correlation (1.0)
        "id_col": [f"ID_{i}" for i in range(n_rows)],  # 100% unique strings
        "constant_col": [1] * 99 + [2],  # 99% single value
        "normal_feat": np.random.randn(n_rows)
    })

    auditor = DataAuditor(df, target_col="target")
    results = auditor.audit()

    assert results["total_issues"] >= 3
    assert results["has_critical"] is True

    categories = [issue["category"] for issue in results["issues"]]
    assert "Target Leakage" in categories
    assert "ID-like Column" in categories
    assert "Near-Constant Column" in categories

    leakage_issue = next(issue for issue in results["issues"] if issue["category"] == "Target Leakage")
    assert leakage_issue["column"] == "leakage_col"
    assert leakage_issue["severity"] == "CRITICAL"

    summary_text = format_audit_summary(results)
    assert "[CRITICAL]" in summary_text
    assert "leakage_col" in summary_text


def test_audit_train_test_overlap_and_drift():
    train_df = pd.DataFrame({
        "val": range(50),
        "target": [0] * 25 + [1] * 25
    })
    # Overlapping 5 rows with train set
    test_df = pd.DataFrame({
        "val": list(range(45, 55))
    })

    auditor = DataAuditor(train_df, target_col="target", test_df=test_df)
    results = auditor.audit()

    categories = [issue["category"] for issue in results["issues"]]
    assert "Train/Test Leakage" in categories


def test_audit_clean_dataset():
    np.random.seed(42)
    df = pd.DataFrame({
        "feat1": np.random.randn(100),
        "feat2": np.random.choice(["A", "B", "C"], 100),
        "target": np.random.randint(0, 2, 100)
    })

    auditor = DataAuditor(df, target_col="target")
    results = auditor.audit()

    assert results["has_critical"] is False
    assert results["critical_count"] == 0
