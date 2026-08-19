"""
Automated EDA Visualization Generator - Compatible with Kaggle Prep CLI
"""

import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ──────────────────────────────────────────────────────────────

def generate_eda_plots(
    df: pd.DataFrame,
    output_dir: str = "eda_plots",
    max_cols: int = 10,
    fig_dpi: int = 150,
    fig_format: str = "png",
    target: Optional[str] = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    numeric_cols  = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols      = df.select_dtypes(include=["object", "category"]).columns.tolist()
    bool_cols     = df.select_dtypes(include=["bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    if target and target in numeric_cols:
        feature_num = [c for c in numeric_cols if c != target]
    else:
        feature_num = numeric_cols

    print("\nGenerating EDA visualizations...")
    print(f"   Rows: {len(df):,}   Columns: {len(df.columns)}")
    print(f"   Numeric: {len(numeric_cols)}  Categorical: {len(cat_cols)}  "
          f"Boolean: {len(bool_cols)}  Datetime: {len(datetime_cols)}")
    print(f"   Saving to: {output_path}\n")

    idx = 1

    # NOTE: _plot calls fn(idx, *args), so args order must match function signature.
    # All column-specific helpers use (idx, df, cols, output_path, dpi, fmt).
    idx = _plot(create_overview,         idx, df,                           output_path, fig_dpi, fig_format)
    idx = _plot(create_missing_values,   idx, df,                           output_path, fig_dpi, fig_format)

    if numeric_cols:
        idx = _plot(create_distributions,  idx, df, feature_num[:max_cols], output_path, fig_dpi, fig_format)
        if SCIPY_AVAILABLE:
            idx = _plot(create_skewness,   idx, df, feature_num[:max_cols], output_path, fig_dpi, fig_format)
            idx = _plot(create_qq_plots,   idx, df, feature_num[:max_cols], output_path, fig_dpi, fig_format)
        idx = _plot(create_violin_plots,   idx, df, feature_num[:max_cols], output_path, fig_dpi, fig_format)
        idx = _plot(create_outlier_summary,idx, df, feature_num[:max_cols], output_path, fig_dpi, fig_format)

    if len(numeric_cols) > 1:
        idx = _plot(create_correlation,    idx, df, numeric_cols,           output_path, fig_dpi, fig_format)

    if cat_cols:
        idx = _plot(create_cardinality,    idx, df, cat_cols[:max_cols],    output_path, fig_dpi, fig_format)
        idx = _plot(create_categorical_bars,idx,df, cat_cols[:max_cols],    output_path, fig_dpi, fig_format)

    if datetime_cols:
        idx = _plot(create_datetime_trends,idx, df, datetime_cols,          output_path, fig_dpi, fig_format)

    if target:
        idx = _plot(create_target_analysis,idx, df, target, numeric_cols, cat_cols, output_path, fig_dpi, fig_format)

    n_files = len(list(output_path.glob(f"*.{fig_format}")))
    print(f"\nDone - {n_files} plots saved to: {output_path}")
    return output_path


# ──────────────────────────────────────────────────────────────
# Internal dispatcher
# ──────────────────────────────────────────────────────────────

def _plot(fn, idx, *args):
    """Call fn(idx, *args), catch errors, return next index."""
    try:
        count = fn(idx, *args)
        return idx + (count or 1)
    except Exception as exc:
        print(f"   Warning: {fn.__name__} skipped: {exc}")
        return idx + 1


def _save(fig, output_path, idx, name, dpi, fmt):
    filename = output_path / f"{idx:02d}_{name}.{fmt}"
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {filename.name}")


# ──────────────────────────────────────────────────────────────
# 1. Dataset overview
# ──────────────────────────────────────────────────────────────

def create_overview(idx, df, output_path, dpi, fmt):
    dupes         = df.duplicated().sum()
    missing_total = df.isna().sum().sum()
    mem_mb        = round(df.memory_usage(deep=True).sum() / 1024**2, 3)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dataset Overview", fontsize=15, fontweight="bold")

    type_counts = df.dtypes.astype(str).value_counts()
    axes[0].bar(type_counts.index, type_counts.values,
                color=["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336"][:len(type_counts)],
                edgecolor="black")
    axes[0].set_title("Column dtypes")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=30)

    labels = ["Rows", "Columns", "Missing\nCells", "Duplicate\nRows", "Memory\n(MB)"]
    values = [len(df), len(df.columns), missing_total, dupes, mem_mb]
    bars   = axes[1].bar(labels, values,
                         color=["#4CAF50", "#2196F3", "#FF9800", "#E53935", "#9C27B0"],
                         edgecolor="black")
    axes[1].set_title("Key metrics")
    for bar, val in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() * 1.01,
                     f"{val:,.3g}", ha="center", fontsize=9)

    plt.tight_layout()
    _save(fig, output_path, idx, "overview", dpi, fmt)
    return 1


# ──────────────────────────────────────────────────────────────
# 2. Missing values
# ──────────────────────────────────────────────────────────────

def create_missing_values(idx, df, output_path, dpi, fmt):
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("   No missing values found")
        return 1

    missing_pct = (missing / len(df) * 100).sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Missing Values", fontsize=15, fontweight="bold")

    sample = df[missing.index].sample(min(500, len(df)), random_state=0)
    sns.heatmap(sample.isna(), yticklabels=False, cbar=False, cmap="YlOrRd", ax=axes[0])
    axes[0].set_title("Missingness pattern (sample)")
    axes[0].set_xlabel("Columns")

    colors = ["#E53935" if p > 20 else "#FF9800" if p > 5 else "#4CAF50"
              for p in missing_pct]
    axes[1].barh(missing_pct.index, missing_pct.values, color=colors, edgecolor="black")
    axes[1].axvline(5,  color="orange", linestyle="--", alpha=0.7, label="5%")
    axes[1].axvline(20, color="red",    linestyle="--", alpha=0.7, label="20%")
    axes[1].set_xlabel("Missing %")
    axes[1].set_title("Missing % per column")
    axes[1].legend()

    plt.tight_layout()
    _save(fig, output_path, idx, "missing_values", dpi, fmt)
    return 1


# ──────────────────────────────────────────────────────────────
# 3. Numeric distributions          signature: (idx, df, cols, ...)
# ──────────────────────────────────────────────────────────────

def create_distributions(idx, df, cols, output_path, dpi, fmt):
    if not cols:
        return 1
    n_cols_grid = 3
    n_rows = (len(cols) + n_cols_grid - 1) // n_cols_grid
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(16, 4 * n_rows))
    axes = np.array(axes).flatten()
    fig.suptitle("Numeric Distributions", fontsize=15, fontweight="bold")

    for i, col in enumerate(cols):
        data = df[col].dropna()
        axes[i].hist(data, bins=30, edgecolor="black", alpha=0.6, color="#4CAF50", density=True)
        data.plot.kde(ax=axes[i], color="navy", linewidth=2)
        axes[i].axvline(data.mean(),   color="red",  linestyle="--", linewidth=1.2, label="mean")
        axes[i].axvline(data.median(), color="blue", linestyle=":",  linewidth=1.2, label="median")
        axes[i].set_title(col, fontsize=9)
        axes[i].legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    _save(fig, output_path, idx, "distributions", dpi, fmt)
    return 1


# ──────────────────────────────────────────────────────────────
# 4. Skewness ranking               signature: (idx, df, cols, ...)
# ──────────────────────────────────────────────────────────────

def create_skewness(idx, df, cols, output_path, dpi, fmt):
    if not cols or not SCIPY_AVAILABLE:
        return 1
    skew = df[cols].skew().sort_values()

    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.6), 5))
    colors = ["#E53935" if abs(v) > 1 else "#FF9800" if abs(v) > 0.5 else "#4CAF50"
              for v in skew]
    ax.barh(skew.index, skew.values, color=colors, edgecolor="black")
    ax.axvline(0,    color="black",  linewidth=0.8)
    ax.axvline( 1,   color="red",    linestyle="--", alpha=0.6, label="|skew| = 1")
    ax.axvline(-1,   color="red",    linestyle="--", alpha=0.6)
    ax.axvline( 0.5, color="orange", linestyle=":",  alpha=0.5)
    ax.axvline(-0.5, color="orange", linestyle=":",  alpha=0.5)
    ax.set_xlabel("Skewness")
    ax.set_title("Feature Skewness  (red = |skew| > 1, consider log transform)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save(fig, output_path, idx, "skewness", dpi, fmt)
    return 1


# ──────────────────────────────────────────────────────────────
# 5. Q-Q normality plots            signature: (idx, df, cols, ...)
# ──────────────────────────────────────────────────────────────

def create_qq_plots(idx, df, cols, output_path, dpi, fmt):
    if not cols or not SCIPY_AVAILABLE:
        return 1
    n_cols_grid = 3
    n_rows = (len(cols) + n_cols_grid - 1) // n_cols_grid
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(16, 4 * n_rows))
    axes = np.array(axes).flatten()
    fig.suptitle("Q-Q Normality Plots  (points off line = non-normal)",
                 fontsize=13, fontweight="bold")

    for i, col in enumerate(cols):
        data = df[col].dropna()
        (osm, osr), (slope, intercept, _) = stats.probplot(data, dist="norm")
        axes[i].scatter(osm, osr, s=10, alpha=0.5, color="#2196F3")
        axes[i].plot(osm, np.array(osm) * slope + intercept, color="red", linewidth=1.5)
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("Theoretical quantiles", fontsize=7)
        axes[i].set_ylabel("Sample quantiles",      fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    _save(fig, output_path, idx, "qq_normality", dpi, fmt)
    return 1


# ──────────────────────────────────────────────────────────────
# 6. Violin plots                   signature: (idx, df, cols, ...)
# ──────────────────────────────────────────────────────────────

def create_violin_plots(idx, df, cols, output_path, dpi, fmt):
    if not cols:
        return 1
    melted = df[cols].melt(var_name="Column", value_name="Value").dropna()

    fig, ax = plt.subplots(figsize=(max(10, len(cols) * 1.2), 7))
    sns.violinplot(data=melted, x="Column", y="Value", hue="Column",
                   palette="viridis", inner="box", density_norm="width",
                   legend=False, ax=ax)
    ax.set_title("Violin Plots - Numeric Columns", fontsize=13, fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("Value")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _save(fig, output_path, idx, "violin_plots", dpi, fmt)
    return 1


# ──────────────────────────────────────────────────────────────
# 7. Outlier summary                signature: (idx, df, cols, ...)
# ──────────────────────────────────────────────────────────────

def create_outlier_summary(idx, df, cols, output_path, dpi, fmt):
    if not cols:
        return 1
    outlier_counts = {}
    for col in cols:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr     = q3 - q1
        n_out   = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        if n_out:
            outlier_counts[col] = n_out

    if not outlier_counts:
        print("   No IQR outliers detected")
        return 1

    series = pd.Series(outlier_counts).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(8, len(series) * 0.8), 5))
    ax.bar(series.index, series.values, color="#E53935", edgecolor="black", alpha=0.8)
    ax.set_title("Outliers per Column (IQR x 1.5 rule)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("# Outliers")
    plt.xticks(rotation=45, ha="right")
    for i, (col, v) in enumerate(series.items()):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    _save(fig, output_path, idx, "outlier_summary", dpi, fmt)
    return 1


# ──────────────────────────────────────────────────────────────
# 8. Correlation heatmap            signature: (idx, df, cols, ...)
# ──────────────────────────────────────────────────────────────

def create_correlation(idx, df, cols, output_path, dpi, fmt):
    if not cols or len(cols) < 2:
        return 1
    corr = df[cols].corr()
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Correlation Analysis", fontsize=15, fontweight="bold")

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=len(cols) <= 12, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.4,
                cbar_kws={"shrink": 0.8}, ax=axes[0])
    axes[0].set_title("Pearson Correlation Matrix")

    flat = corr.where(~mask).stack().abs().sort_values(ascending=False).head(10)
    if len(flat) > 0:
        labels = [f"{a}->{b}" for a, b in flat.index]
        axes[1].barh(labels[::-1], flat.values[::-1], color="#2196F3", edgecolor="black")
        axes[1].set_xlabel("|Pearson r|")
        axes[1].set_title("Top-10 Correlated Pairs")
        axes[1].axvline(0.8, color="red",    linestyle="--", alpha=0.6, label="r=0.8")
        axes[1].axvline(0.5, color="orange", linestyle="--", alpha=0.5, label="r=0.5")
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "No correlations found", ha="center", va="center")

    plt.tight_layout()
    _save(fig, output_path, idx, "correlation", dpi, fmt)
    return 1


# ──────────────────────────────────────────────────────────────
# 9. Cardinality check              signature: (idx, df, cols, ...)
# ──────────────────────────────────────────────────────────────

def create_cardinality(idx, df, cols, output_path, dpi, fmt):
    if not cols:
        return 1
    card      = {c: df[c].nunique() for c in cols}
    series    = pd.Series(card).sort_values(ascending=False)
    threshold = max(20, len(df) * 0.05)

    fig, ax = plt.subplots(figsize=(max(8, len(series) * 0.8), 5))
    colors = ["#E53935" if v >= threshold else "#4CAF50" for v in series.values]
    ax.bar(series.index, series.values, color=colors, edgecolor="black")
    ax.axhline(threshold, color="red", linestyle="--", alpha=0.7,
               label=f"High-cardinality threshold ({int(threshold)})")
    ax.set_title("Categorical Cardinality  (red = high, consider target-encoding)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("# Unique Values")
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    for i, (col, v) in enumerate(series.items()):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    _save(fig, output_path, idx, "cardinality", dpi, fmt)
    return 1


# ──────────────────────────────────────────────────────────────
# 10. Categorical bar charts        signature: (idx, df, cols, ...)
# ──────────────────────────────────────────────────────────────

def create_categorical_bars(idx, df, cols, output_path, dpi, fmt):
    if not cols:
        return 1
    n_cols_grid = 3
    n_rows = (len(cols) + n_cols_grid - 1) // n_cols_grid
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(16, 4 * n_rows))
    axes = np.array(axes).flatten()
    fig.suptitle("Categorical Value Distributions (top 10)", fontsize=13, fontweight="bold")

    for i, col in enumerate(cols):
        vc = df[col].value_counts().head(10)
        axes[i].bar(vc.index.astype(str), vc.values,
                    color="#2196F3", edgecolor="black", alpha=0.8)
        axes[i].set_title(col, fontsize=9)
        axes[i].set_ylabel("Count")
        axes[i].tick_params(axis="x", rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    _save(fig, output_path, idx, "categorical_bars", dpi, fmt)
    return 1


# ──────────────────────────────────────────────────────────────
# 11. Datetime trends               signature: (idx, df, datetime_cols, ...)
# ──────────────────────────────────────────────────────────────

def create_datetime_trends(idx, df, datetime_cols, output_path, dpi, fmt):
    if not datetime_cols:
        return 1
    n_plots = len(datetime_cols)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]
    fig.suptitle("Row Count Over Time (Datetime Columns)", fontsize=13, fontweight="bold")

    for ax, col in zip(axes, datetime_cols):
        ts    = df[col].dropna().sort_values()
        span  = ts.max() - ts.min()
        freq  = "ME" if span.days > 365 else "W" if span.days > 60 else "D"
        counts = ts.dt.to_period(freq).value_counts().sort_index()
        ax.plot(counts.index.astype(str), counts.values,
                marker="o", markersize=3, color="#4CAF50", linewidth=1.5)
        ax.set_title(col)
        ax.set_ylabel("# Records")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    _save(fig, output_path, idx, "datetime_trends", dpi, fmt)
    return 1


# ──────────────────────────────────────────────────────────────
# 12. Target-aware analysis
# ──────────────────────────────────────────────────────────────

def create_target_analysis(idx, df, target, numeric_cols, cat_cols,
                            output_path, dpi, fmt):
    if target not in df.columns:
        print(f"   Warning: Target column '{target}' not found")
        return 1

    is_numeric_target = pd.api.types.is_numeric_dtype(df[target])
    plots_made = 0

    if is_numeric_target:
        feature_cols = [c for c in numeric_cols if c != target]

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(f"Target Analysis: {target}", fontsize=13, fontweight="bold")

        data = df[target].dropna()
        axes[0].hist(data, bins=40, edgecolor="black", alpha=0.7, color="#9C27B0", density=True)
        data.plot.kde(ax=axes[0], color="navy", linewidth=2)
        axes[0].axvline(data.mean(),   color="red",  linestyle="--", label="mean")
        axes[0].axvline(data.median(), color="blue", linestyle=":",  label="median")
        axes[0].set_title(f"Distribution of {target}")
        axes[0].legend()

        if feature_cols:
            corr_with_target = (df[feature_cols + [target]]
                                .corr()[target]
                                .drop(target)
                                .abs()
                                .sort_values(ascending=True)
                                .tail(20))
            axes[1].barh(corr_with_target.index, corr_with_target.values,
                         color="#FF9800", edgecolor="black")
            axes[1].set_title(f"|Correlation| with {target} (top 20)")
            axes[1].set_xlabel("|Pearson r|")
            axes[1].axvline(0.3, color="green", linestyle="--", alpha=0.6, label="r=0.3")
            axes[1].legend()

        plt.tight_layout()
        _save(fig, output_path, idx, f"target_{target}_numeric", dpi, fmt)
        plots_made += 1

    else:
        vc = df[target].value_counts()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Target Analysis: {target}", fontsize=13, fontweight="bold")

        axes[0].bar(vc.index.astype(str), vc.values, color="#9C27B0", edgecolor="black")
        axes[0].set_title("Class Balance")
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis="x", rotation=30)
        for i, v in enumerate(vc.values):
            axes[0].text(i, v, f"{v/len(df)*100:.1f}%", ha="center", va="bottom", fontsize=9)

        axes[1].pie(vc.values, labels=vc.index.astype(str),
                    autopct="%1.1f%%", startangle=90, pctdistance=0.85)
        axes[1].set_title("Class Distribution (%)")

        plt.tight_layout()
        _save(fig, output_path, idx, f"target_{target}_balance", dpi, fmt)
        plots_made += 1

        plot_feats = [c for c in numeric_cols if c != target][:6]
        if plot_feats:
            n_cols_grid = 3
            n_rows = (len(plot_feats) + n_cols_grid - 1) // n_cols_grid
            fig2, axes2 = plt.subplots(n_rows, n_cols_grid, figsize=(16, 4 * n_rows))
            axes2 = np.array(axes2).flatten()
            fig2.suptitle(f"Feature Distributions by {target}", fontsize=13, fontweight="bold")

            classes = df[target].dropna().unique()
            palette = sns.color_palette("Set2", len(classes))

            for i, col in enumerate(plot_feats):
                for cls, color in zip(classes, palette):
                    subset = df[df[target] == cls][col].dropna()
                    axes2[i].hist(subset, bins=25, alpha=0.55,
                                  color=color, edgecolor="black", label=str(cls), density=True)
                axes2[i].set_title(col, fontsize=9)
                axes2[i].legend(fontsize=7)

            for j in range(i + 1, len(axes2)):
                axes2[j].set_visible(False)

            plt.tight_layout()
            _save(fig2, output_path, idx + 1, f"target_{target}_features", dpi, fmt)
            plots_made += 1

    return plots_made


# ──────────────────────────────────────────────────────────────
# Preprocessing code generator (kept for cli.py compatibility)
# ──────────────────────────────────────────────────────────────

def generate_preprocessing_code(df):
    """Generate preprocessing code based on data profile"""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols     = df.select_dtypes(include=["object", "category"]).columns.tolist()
    bool_cols    = df.select_dtypes(include=["bool"]).columns.tolist()

    code_lines = []
    code_lines.append(f"""
# Auto-Generated Preprocessing Pipeline
# Generated from {len(df)} rows, {len(df.columns)} columns

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# 1. Load Data
df = pd.read_csv('data/your_dataset.csv')
print(f"Loaded: {{df.shape}}")

# 2. Identify Column Types
numeric_cols    = {numeric_cols}
categorical_cols = {cat_cols}
boolean_cols    = {bool_cols}

# 3. Handle Missing Values
""")

    if numeric_cols:
        code_lines.append("for col in numeric_cols:")
        code_lines.append("    if df[col].isna().sum() > 0:")
        code_lines.append("        df[col].fillna(df[col].median(), inplace=True)")

    if cat_cols:
        code_lines.append("")
        code_lines.append("for col in categorical_cols:")
        code_lines.append("    if df[col].isna().sum() > 0:")
        code_lines.append("        df[col].fillna(df[col].mode()[0], inplace=True)")

    if cat_cols:
        code_lines.append("""
# 4. Encode Categorical Variables
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
""")

    if numeric_cols:
        code_lines.append("""
# 5. Scale Numeric Features
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
""")

    code_lines.append("""
# 6. Prepare for Modeling
# target_column = 'target'
# X = df.drop(target_column, axis=1)
# y = df[target_column]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Preprocessing complete! Final shape: {df.shape}")
""")
    return "\n".join(code_lines)


# ──────────────────────────────────────────────────────────────
# Standalone CLI entry point
# ──────────────────────────────────────────────────────────────

def main():
    import argparse, sys

    parser = argparse.ArgumentParser(
        prog="visualizer",
        description="Auto-generate EDA plots for any CSV file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input",         help="Path to the CSV file to analyse")
    parser.add_argument("--output-dir",  "-o", default="eda_plots")
    parser.add_argument("--target",      "-t", default=None)
    parser.add_argument("--max-cols",    "-m", type=int, default=10)
    parser.add_argument("--format",      "-f", default="png",
                        choices=["png", "pdf", "svg", "jpg"])
    parser.add_argument("--dpi",         "-d", type=int, default=150)
    parser.add_argument("--sample",      "-s", type=int, default=None)
    parser.add_argument("--sep",         default=",")

    args     = parser.parse_args()
    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    sep = "\t" if args.sep == "\\t" else args.sep
    df  = pd.read_csv(csv_path, sep=sep, low_memory=False)

    if args.sample and args.sample < len(df):
        print(f"Sampling {args.sample:,} rows from {len(df):,}")
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)

    generate_eda_plots(
        df,
        output_dir=args.output_dir,
        max_cols=args.max_cols,
        fig_dpi=args.dpi,
        fig_format=args.format,
        target=args.target,
    )


if __name__ == "__main__":
    main()