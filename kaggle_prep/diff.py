"""
Incremental profiling and snapshot diffing module for kaggle-prep.
Supports column hashing for fast --update incremental profiling and
diffing between two saved dataset profiling snapshots.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats


def compute_column_hash(series: pd.Series) -> str:
    """Compute a lightweight, deterministic hash of a pandas Series data and metadata."""
    hasher = hashlib.md5()
    hasher.update(series.name.encode("utf-8") if series.name else b"")
    hasher.update(str(series.dtype).encode("utf-8"))
    hasher.update(str(len(series)).encode("utf-8"))

    # Sample values for fast hashing if huge, or hash string representation
    if len(series) > 10000:
        sample = pd.concat([series.head(5000), series.tail(5000)])
    else:
        sample = series

    hasher.update(pd.util.hash_pandas_object(sample, index=False).values.tobytes())
    return hasher.hexdigest()


def compute_dataframe_column_hashes(df: pd.DataFrame) -> Dict[str, str]:
    """Return dictionary mapping column names to their MD5 hashes."""
    return {col: compute_column_hash(df[col]) for col in df.columns}


def load_snapshot_json(snapshot_path: str) -> Dict[str, Any]:
    """Load JSON snapshot file."""
    path = Path(snapshot_path)
    if not path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_snapshot_json(profile: Dict[str, Any], output_path: Optional[str] = None) -> Path:
    """Save profile dict (including column hashes) to data_profiles directory."""
    out_dir = Path("data_profiles")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = profile.get("dataset", "dataset").replace("/", "_")
    target_path = Path(output_path) if output_path else out_dir / f"{dataset_name}_snapshot.json"

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, default=str)

    return target_path


def diff_snapshots(old_snapshot: Dict[str, Any], new_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two profiling snapshots.
    Returns structured diff highlighting:
    - new_columns
    - dropped_columns
    - row_count_delta
    - column_changes (missingness_delta, unique_delta, mean_delta, std_delta, distribution_shift)
    """
    old_cols = old_snapshot.get("columns", {})
    new_cols = new_snapshot.get("columns", {})

    old_names = set(old_cols.keys())
    new_names = set(new_cols.keys())

    added = sorted(list(new_names - old_names))
    dropped = sorted(list(old_names - new_names))
    common = sorted(list(old_names & new_names))

    row_delta = new_snapshot.get("rows", 0) - old_snapshot.get("rows", 0)

    column_diffs = []
    for col in common:
        o = old_cols[col]
        n = new_cols[col]

        missing_delta = round(n.get("missing_percent", 0) - o.get("missing_percent", 0), 2)
        unique_delta = n.get("unique", 0) - o.get("unique", 0)

        mean_delta = None
        if "mean" in o and "mean" in n:
            mean_delta = round(n["mean"] - o["mean"], 4)

        # Check for distribution shift flag
        dist_shifted = False
        if mean_delta is not None and abs(mean_delta) > 0.05 * (abs(o.get("std", 1)) or 1):
            dist_shifted = True

        if missing_delta != 0 or unique_delta != 0 or (mean_delta is not None and mean_delta != 0) or dist_shifted:
            column_diffs.append({
                "column": col,
                "missing_delta": missing_delta,
                "unique_delta": unique_delta,
                "mean_delta": mean_delta,
                "distribution_shifted": dist_shifted
            })

    return {
        "old_dataset": old_snapshot.get("dataset", "old"),
        "new_dataset": new_snapshot.get("dataset", "new"),
        "row_count_delta": row_delta,
        "added_columns": added,
        "dropped_columns": dropped,
        "modified_columns": column_diffs,
    }


def format_diff_cli_table(diff_results: Dict[str, Any]) -> str:
    """Format snapshot diff results as text table for CLI."""
    lines = [
        "=" * 60,
        f"  Dataset Diff: {diff_results['old_dataset']} -> {diff_results['new_dataset']}",
        "=" * 60,
        f" Row count delta: {diff_results['row_count_delta']:+d}",
    ]

    if diff_results["added_columns"]:
        lines.append(f" New columns added ({len(diff_results['added_columns'])}): {', '.join(diff_results['added_columns'])}")

    if diff_results["dropped_columns"]:
        lines.append(f" Columns dropped ({len(diff_results['dropped_columns'])}): {', '.join(diff_results['dropped_columns'])}")

    if diff_results["modified_columns"]:
        lines.append("\n Column Profiling Deltas:")
        lines.append(f" {'Column':<20} | {'Missing % Delta':<16} | {'Unique Delta':<14} | {'Mean Delta':<12}")
        lines.append("-" * 70)
        for c in diff_results["modified_columns"]:
            col_name = c["column"][:19]
            miss_str = f"{c['missing_delta']:+.2f}%"
            uniq_str = f"{c['unique_delta']:+d}"
            mean_str = f"{c['mean_delta']:+.4f}" if c['mean_delta'] is not None else "N/A"
            shift_tag = " *SHIFT*" if c["distribution_shifted"] else ""
            lines.append(f" {col_name:<20} | {miss_str:<16} | {uniq_str:<14} | {mean_str:<12}{shift_tag}")
    else:
        lines.append(" No material profiling changes detected in common columns.")

    lines.append("=" * 60)
    return "\n".join(lines)
