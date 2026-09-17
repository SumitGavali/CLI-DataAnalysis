"""
Data-quality and leakage audit module for kaggle-prep.
Performs active automated checks for target leakage, ID-like columns, near-constant features, duplicates, and distribution drift.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats


class DataAuditor:
    """Audits a dataset (or train/test pair) for common data quality issues and model traps."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        test_df: Optional[pd.DataFrame] = None,
        leakage_threshold: float = 0.95,
        id_uniqueness_threshold: float = 0.98,
        near_constant_threshold: float = 0.99,
        drift_p_threshold: float = 0.01,
    ):
        self.df = df
        self.target_col = target_col
        self.test_df = test_df
        self.leakage_threshold = leakage_threshold
        self.id_uniqueness_threshold = id_uniqueness_threshold
        self.near_constant_threshold = near_constant_threshold
        self.drift_p_threshold = drift_p_threshold

    def audit(self) -> Dict[str, Any]:
        """Run all audit checks and return audit report dictionary."""
        issues: List[Dict[str, Any]] = []

        # 1. Target Leakage Check
        if self.target_col and self.target_col in self.df.columns:
            leakage_issues = self._check_target_leakage()
            issues.extend(leakage_issues)

        # 2. ID-like Columns Check
        id_issues = self._check_id_columns()
        issues.extend(id_issues)

        # 3. Near-Constant Columns Check
        constant_issues = self._check_near_constant()
        issues.extend(constant_issues)

        # 4. Duplicate Rows Check
        dup_issues = self._check_duplicates()
        issues.extend(dup_issues)

        # 5. Train/Test Distribution Drift Check
        if self.test_df is not None:
            drift_issues = self._check_distribution_drift()
            issues.extend(drift_issues)

        critical_count = sum(1 for issue in issues if issue["severity"] == "CRITICAL")
        warning_count = sum(1 for issue in issues if issue["severity"] == "WARNING")

        return {
            "issues": issues,
            "has_critical": critical_count > 0,
            "critical_count": critical_count,
            "warning_count": warning_count,
            "total_issues": len(issues),
        }

    def _check_target_leakage(self) -> List[Dict[str, Any]]:
        results = []
        target_series = self.df[self.target_col]

        # Numeric correlation
        if pd.api.types.is_numeric_dtype(target_series):
            num_target = target_series
        else:
            # Map object/categorical target to integer codes for correlation check
            num_target = pd.Series(pd.factorize(target_series)[0], index=self.df.index)

        for col in self.df.columns:
            if col == self.target_col:
                continue

            feature_series = self.df[col]
            if pd.api.types.is_numeric_dtype(feature_series):
                num_feat = feature_series
            else:
                num_feat = pd.Series(pd.factorize(feature_series)[0], index=self.df.index)

            # Drop missing values pairwise
            valid_mask = num_feat.notna() & num_target.notna()
            if valid_mask.sum() < 5:
                continue

            corr = abs(num_feat[valid_mask].corr(num_target[valid_mask]))
            if not np.isnan(corr) and corr >= self.leakage_threshold:
                results.append({
                    "category": "Target Leakage",
                    "column": col,
                    "severity": "CRITICAL",
                    "message": f"Feature '{col}' has extreme correlation ({corr:.4f}) with target '{self.target_col}'."
                })

        return results

    def _check_id_columns(self) -> List[Dict[str, Any]]:
        results = []
        total_rows = len(self.df)
        if total_rows == 0:
            return results

        for col in self.df.columns:
            if col == self.target_col:
                continue

            nunique = self.df[col].nunique(dropna=True)
            uniqueness_ratio = nunique / total_rows

            if uniqueness_ratio >= self.id_uniqueness_threshold and total_rows > 10:
                results.append({
                    "category": "ID-like Column",
                    "column": col,
                    "severity": "WARNING",
                    "message": f"Column '{col}' is near-100% unique ({uniqueness_ratio * 100:.1f}% unique values). Likely identifier column."
                })

        return results

    def _check_near_constant(self) -> List[Dict[str, Any]]:
        results = []
        total_rows = len(self.df)
        if total_rows == 0:
            return results

        for col in self.df.columns:
            if col == self.target_col:
                continue

            val_counts = self.df[col].value_counts(dropna=False)
            if len(val_counts) == 0:
                continue

            top_freq_pct = val_counts.iloc[0] / total_rows
            if top_freq_pct >= self.near_constant_threshold:
                top_val = val_counts.index[0]
                results.append({
                    "category": "Near-Constant Column",
                    "column": col,
                    "severity": "WARNING",
                    "message": f"Column '{col}' has single value ('{top_val}') accounting for {top_freq_pct * 100:.1f}% of rows."
                })

        return results

    def _check_duplicates(self) -> List[Dict[str, Any]]:
        results = []
        exact_dups = int(self.df.duplicated().sum())
        if exact_dups > 0:
            dup_pct = (exact_dups / len(self.df)) * 100
            severity = "CRITICAL" if dup_pct > 10 else "WARNING"
            results.append({
                "category": "Duplicate Rows",
                "column": "ALL",
                "severity": severity,
                "message": f"Dataset contains {exact_dups:,} exact duplicate rows ({dup_pct:.1f}% of total)."
            })

        if self.test_df is not None:
            # Check overlap between train and test
            common_cols = [c for c in self.df.columns if c in self.test_df.columns and c != self.target_col]
            if common_cols:
                train_sub = self.df[common_cols]
                test_sub = self.test_df[common_cols]
                merged = pd.merge(train_sub, test_sub, how="inner").drop_duplicates()
                overlap_count = len(merged)
                if overlap_count > 0:
                    results.append({
                        "category": "Train/Test Leakage",
                        "column": "ALL",
                        "severity": "CRITICAL",
                        "message": f"Found {overlap_count:,} duplicate feature rows across train and test sets!"
                    })

        return results

    def _check_distribution_drift(self) -> List[Dict[str, Any]]:
        results = []
        if self.test_df is None:
            return results

        common_cols = [c for c in self.df.columns if c in self.test_df.columns and c != self.target_col]
        for col in common_cols:
            train_series = self.df[col].dropna()
            test_series = self.test_df[col].dropna()

            if len(train_series) < 10 or len(test_series) < 10:
                continue

            if pd.api.types.is_numeric_dtype(train_series) and pd.api.types.is_numeric_dtype(test_series):
                ks_stat, p_val = stats.ks_2samp(train_series, test_series)
                if p_val < self.drift_p_threshold:
                    results.append({
                        "category": "Distribution Drift",
                        "column": col,
                        "severity": "WARNING",
                        "message": f"Column '{col}' exhibits significant distribution drift between train and test (KS p-value = {p_val:.4e})."
                    })

        return results


def format_audit_summary(audit_results: Dict[str, Any]) -> str:
    """Format audit results as text for CLI output."""
    issues = audit_results["issues"]
    lines = [
        "=" * 55,
        "  Data Quality & Leakage Audit Report",
        "=" * 55,
    ]

    if not issues:
        lines.append(" Clean! No data quality or leakage issues detected.")
    else:
        for issue in issues:
            symbol = " [CRITICAL]" if issue["severity"] == "CRITICAL" else "! [WARNING]"
            lines.append(f"{symbol} {issue['category']} ({issue['column']}): {issue['message']}")

        lines.append("-" * 55)
        lines.append(f" Total Issues: {audit_results['total_issues']} ({audit_results['critical_count']} Critical, {audit_results['warning_count']} Warnings)")

    lines.append("=" * 55)
    return "\n".join(lines)
