"""
Target-aware and task-aware intelligence module.
Provides automatic task detection, class imbalance analysis, metric recommendations, and split strategies.
"""

from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np


def detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Auto-detect ML task type:
    - 'binary_classification'
    - 'multiclass_classification'
    - 'regression'
    - 'time_series'
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    series = df[target_col].dropna()
    num_unique = series.nunique()

    # Check for time-series heuristics:
    # 1) Has a datetime column in df AND target is continuous or time-ordered
    # 2) Target itself is datetime or time-like
    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if not datetime_cols:
        # Check string columns for datetime parses
        for col in df.columns:
            if df[col].dtype == "object":
                # Check if name contains date/time/year/month or sample converts to datetime
                if any(kw in col.lower() for kw in ["date", "time", "timestamp", "year", "month", "day"]):
                    datetime_cols.append(col)
                    break

    if datetime_cols and pd.api.types.is_numeric_dtype(series) and num_unique > 20:
        # Check if datetime column is sorted or monotonically increasing
        try:
            dt_series = pd.to_datetime(df[datetime_cols[0]], errors="coerce")
            if dt_series.is_monotonic_increasing:
                return "time_series"
        except Exception:
            pass

    # Categorical/Object or low cardinality -> Classification
    if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(series):
        if num_unique == 2:
            return "binary_classification"
        else:
            return "multiclass_classification"

    # Integer or Float target
    if pd.api.types.is_numeric_dtype(series):
        # High cardinality -> Regression
        if num_unique > 20 and not (num_unique < 0.05 * len(series) and pd.api.types.is_integer_dtype(series)):
            return "regression"
        elif num_unique == 2:
            return "binary_classification"
        else:
            return "multiclass_classification"

    return "regression"


def analyze_class_balance(df: pd.DataFrame, target_col: str) -> Optional[Dict[str, Any]]:
    """Analyze class counts, ratios, minority percentage, and imbalance status."""
    series = df[target_col].dropna()
    counts = series.value_counts().to_dict()
    total = len(series)
    if total == 0:
        return None

    ratios = {str(k): round(float(v) / total * 100, 2) for k, v in counts.items()}
    minority_class = min(counts, key=counts.get)
    minority_pct = round(float(counts[minority_class]) / total * 100, 2)
    majority_pct = max(ratios.values())

    # Imbalance threshold: majority > 70% or minority < 20%
    is_imbalanced = majority_pct > 70.0 or minority_pct < 20.0

    return {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "ratios": ratios,
        "minority_class": str(minority_class),
        "minority_pct": minority_pct,
        "is_imbalanced": is_imbalanced,
    }


def recommend_metrics(task_type: str, class_info: Optional[Dict[str, Any]] = None) -> List[str]:
    """Recommend baseline metrics based on task type and class balance."""
    if task_type == "binary_classification":
        if class_info and class_info.get("is_imbalanced"):
            return ["ROC-AUC", "F1-Score (minority)", "PR-AUC"]
        else:
            return ["ROC-AUC", "F1-Score", "Accuracy"]
    elif task_type == "multiclass_classification":
        if class_info and class_info.get("is_imbalanced"):
            return ["Macro-F1", "Weighted-F1", "Log-Loss"]
        else:
            return ["Accuracy", "Macro-F1", "Log-Loss"]
    elif task_type == "time_series":
        return ["RMSE", "MAE", "MAPE"]
    else:  # regression
        return ["RMSE", "MAE", "R2-Score"]


def determine_split_strategy(task_type: str) -> Dict[str, str]:
    """Determine train/val split strategy for modelling/notebook."""
    if task_type in ("binary_classification", "multiclass_classification"):
        return {
            "name": "stratified 80/20",
            "type": "stratified",
            "code": "train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)"
        }
    elif task_type == "time_series":
        return {
            "name": "time-based 80/20",
            "type": "time_based",
            "code": "train_test_split(X, y, test_size=0.2, shuffle=False)"
        }
    else:
        return {
            "name": "random 80/20",
            "type": "random",
            "code": "train_test_split(X, y, test_size=0.2, random_state=42)"
        }


def format_intelligence_summary(target_col: str, task_type: str, class_info: Optional[Dict[str, Any]], metrics: List[str], split_info: Dict[str, str]) -> str:
    """Format short human-readable summary block for CLI output."""
    readable_task = task_type.replace("_", " ")
    lines = [
        f" Target Column: '{target_col}'",
        f" Detected task: {readable_task}",
    ]

    if class_info:
        ratios_str = " | ".join([f"{k} -> {v}%" for k, v in class_info["ratios"].items()])
        imbalance_label = " (imbalanced)" if class_info["is_imbalanced"] else " (balanced)"
        lines.append(f" Class balance: {ratios_str}{imbalance_label}")

    primary_metric = metrics[0]
    secondary_metrics = ", ".join(metrics[1:])
    lines.append(f" Recommended metric: {primary_metric} ({secondary_metrics} as secondary)")
    lines.append(f" Split strategy: {split_info['name']}")

    border = "-" * 55
    return "\n".join([border, *lines, border])
