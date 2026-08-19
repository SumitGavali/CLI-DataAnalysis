# kaggle_prep/visualizer.py
"""
Automated EDA Visualization Generator
Creates professional plots for data exploration
"""

import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np


def generate_eda_plots(
    df: pd.DataFrame,
    output_dir: str = "eda_plots",
    max_cols: int = 10,
    fig_dpi: int = 150,
    fig_format: str = 'png'
) -> Path:
    """
    Auto-generate comprehensive EDA visualizations
    
    Args:
        df: DataFrame to analyze
        output_dir: Directory to save plots
        max_cols: Maximum number of columns to plot per category
        fig_dpi: Resolution of saved figures
        fig_format: File format (png, jpg, pdf, svg)
    
    Returns:
        Path to the output directory
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Generating EDA visualizations...")
    print(f"📁 Saving to: {output_path}")
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    
    print(f"   Numeric: {len(numeric_cols)} columns")
    print(f"   Categorical: {len(categorical_cols)} columns")
    print(f"   Datetime: {len(datetime_cols)} columns")
    print(f"   Boolean: {len(bool_cols)} columns")
    
    # ============================================================
    # 1. Dataset Overview
    # ============================================================
    create_overview_plots(df, output_path, fig_dpi, fig_format)
    
    # ============================================================
    # 2. Missing Values Analysis
    # ============================================================
    if df.isna().sum().sum() > 0:
        create_missing_values_plots(df, output_path, fig_dpi, fig_format)
    
    # ============================================================
    # 3. Numeric Column Distributions
    # ============================================================
    if numeric_cols:
        create_numeric_distributions(
            df, numeric_cols[:max_cols], output_path, fig_dpi, fig_format
        )
        
        # Box plots for numeric columns
        if len(numeric_cols) > 1:
            create_box_plots(
                df, numeric_cols[:max_cols], output_path, fig_dpi, fig_format
            )
    
    # ============================================================
    # 4. Correlation Analysis
    # ============================================================
    if len(numeric_cols) > 1:
        create_correlation_plots(
            df, numeric_cols, output_path, fig_dpi, fig_format
        )
    
    # ============================================================
    # 5. Categorical Column Analysis
    # ============================================================
    if categorical_cols:
        create_categorical_plots(
            df, categorical_cols[:max_cols], output_path, fig_dpi, fig_format
        )
    
    # ============================================================
    # 6. Pair Plot (for small datasets)
    # ============================================================
    if len(numeric_cols) <= 6 and len(df) <= 1000:
        create_pair_plot(df, numeric_cols, output_path, fig_dpi, fig_format)
    
    # ============================================================
    # 7. Outlier Detection
    # ============================================================
    if numeric_cols:
        create_outlier_plots(
            df, numeric_cols[:max_cols], output_path, fig_dpi, fig_format
        )
    
    print(f"\n✅ EDA visualization complete! {len(list(output_path.glob('*')))} files generated.")
    print(f"📁 Plots saved to: {output_path}")
    
    return output_path


# ============================================================
# Helper Functions for Individual Plot Types
# ============================================================

def create_overview_plots(df, output_path, dpi, fmt):
    """Create dataset overview visualizations"""
    
    # 1. Column types distribution
    plt.figure(figsize=(10, 6))
    type_counts = pd.Series(df.dtypes).value_counts()
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336']
    type_counts.plot(kind='bar', color=colors[:len(type_counts)], edgecolor='black')
    plt.title('Column Types Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Data Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / f'01_column_types.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 2. Data size summary
    plt.figure(figsize=(8, 6))
    summary_data = {
        'Metric': ['Rows', 'Columns', 'Missing Values', 'Memory (MB)'],
        'Value': [
            len(df),
            len(df.columns),
            df.isna().sum().sum(),
            round(df.memory_usage().sum() / 1024**2, 2)
        ]
    }
    plt.bar(summary_data['Metric'], summary_data['Value'], 
            color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
    plt.title('Dataset Summary', fontsize=14, fontweight='bold')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(output_path / f'02_dataset_summary.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close()


def create_missing_values_plots(df, output_path, dpi, fmt):
    """Create missing values visualizations"""
    
    # 1. Missing values heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isna(), yticklabels=False, cbar=True, cmap='viridis')
    plt.title('Missing Values Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Columns')
    plt.tight_layout()
    plt.savefig(output_path / f'03_missing_values_heatmap.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 2. Missing values bar chart
    missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    
    if len(missing_pct) > 0:
        plt.figure(figsize=(10, 6))
        missing_pct.plot(kind='bar', color='#FF9800', edgecolor='black')
        plt.title('Missing Values by Column', fontsize=14, fontweight='bold')
        plt.xlabel('Columns')
        plt.ylabel('Missing Percentage (%)')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% threshold')
        plt.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='20% threshold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / f'04_missing_values_barchart.{fmt}', dpi=dpi, bbox_inches='tight')
        plt.close()


def create_numeric_distributions(df, numeric_cols, output_path, dpi, fmt):
    """Create distribution plots for numeric columns"""
    
    # Individual histograms
    n_cols = 2
    n_rows = (len(numeric_cols) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            axes[i].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7, color='#4CAF50')
            axes[i].set_title(f'Distribution of {col}', fontweight='bold')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].axvline(df[col].mean(), color='red', linestyle='--', alpha=0.7, label='Mean')
            axes[i].axvline(df[col].median(), color='blue', linestyle='--', alpha=0.7, label='Median')
            axes[i].legend()
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path / f'05_numeric_distributions.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # Combined KDE plot (if more than 1 numeric column)
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 8))
        for col in numeric_cols[:6]:  # Limit to 6 for readability
            sns.kdeplot(df[col].dropna(), label=col, linewidth=2)
        plt.title('Density Plots - All Numeric Columns', fontsize=14, fontweight='bold')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / f'06_kde_combined.{fmt}', dpi=dpi, bbox_inches='tight')
        plt.close()


def create_box_plots(df, numeric_cols, output_path, dpi, fmt):
    """Create box plots for numeric columns"""
    
    # Prepare data for box plot
    df_melted = df[numeric_cols].melt(var_name='Column', value_name='Value')
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df_melted, x='Column', y='Value', hue='Column', palette='viridis', legend=False)

    plt.title('Box Plots - Numeric Columns', fontsize=14, fontweight='bold')
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / f'07_box_plots.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close()


def create_correlation_plots(df, numeric_cols, output_path, dpi, fmt):
    """Create correlation visualizations"""
    
    # 1. Correlation heatmap
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, 
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / f'08_correlation_heatmap.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 2. Pairwise scatter plots (limit to 6 columns)
    if len(numeric_cols) <= 6:
        scatter_cols = numeric_cols
    else:
        # Select columns with highest correlation
        corr_flat = corr_matrix.unstack().sort_values(ascending=False)
        corr_flat = corr_flat[corr_flat < 1]  # Remove self-correlations
        top_pairs = corr_flat.head(4).index
        scatter_cols = list(set([col for pair in top_pairs for col in pair]))[:6]
    
    if len(scatter_cols) > 1:
        # Create scatter matrix
        n_features = len(scatter_cols)
        fig, axes = plt.subplots(n_features, n_features, figsize=(12, 12))
        
        for i, col1 in enumerate(scatter_cols):
            for j, col2 in enumerate(scatter_cols):
                if i == j:
                    axes[i, j].hist(df[col1].dropna(), bins=20, color='#4CAF50', edgecolor='black')
                    axes[i, j].set_title(f'{col1}', fontsize=8)
                else:
                    axes[i, j].scatter(df[col1], df[col2], alpha=0.5, s=10)
                axes[i, j].set_xticklabels([])
                axes[i, j].set_yticklabels([])
        
        plt.suptitle('Pairwise Scatter Plots', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / f'09_scatter_matrix.{fmt}', dpi=dpi, bbox_inches='tight')
        plt.close()


def create_categorical_plots(df, categorical_cols, output_path, dpi, fmt):
    """Create categorical column visualizations"""
    
    # Bar plots for categorical columns
    n_cols = 2
    n_rows = (len(categorical_cols) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    for i, col in enumerate(categorical_cols):
        if i < len(axes):
            value_counts = df[col].value_counts().head(10)
            axes[i].bar(value_counts.index, value_counts.values, 
                       color='#2196F3', edgecolor='black', alpha=0.7)
            axes[i].set_title(f'{col} (Top 10 values)', fontweight='bold')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
            axes[i].tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path / f'10_categorical_barcharts.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close()


def create_pair_plot(df, numeric_cols, output_path, dpi, fmt):
    """Create pair plot for small datasets"""
    
    if len(numeric_cols) <= 6 and len(df) <= 1000:
        # Add categorical variable for coloring if exists
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        hue_col = cat_cols[0] if len(cat_cols) > 0 else None
        
        try:
            plt.figure(figsize=(12, 12))
            pair_grid = sns.pairplot(
                df[numeric_cols + list(cat_cols[:1]) if hue_col else numeric_cols],
                hue=hue_col if hue_col else None,
                diag_kind='kde',
                plot_kws={'alpha': 0.5, 's': 15},
                diag_kws={'fill': True}
            )
            pair_grid.fig.suptitle('Pair Plot - All Numeric Columns', 
                                   fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(output_path / f'11_pair_plot.{fmt}', dpi=dpi, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"   ⚠️ Could not generate pair plot: {e}")


def create_outlier_plots(df, numeric_cols, output_path, dpi, fmt):
    """Create outlier detection visualizations"""
    
    # Prepare data for outlier detection
    outlier_data = {}
    for col in numeric_cols:
        if len(df[col]) > 0:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_data[col] = len(outliers)
    
    if any(outlier_data.values()):
        plt.figure(figsize=(10, 6))
        outlier_series = pd.Series(outlier_data).sort_values(ascending=False)
        outlier_series = outlier_series[outlier_series > 0]
        plt.bar(outlier_series.index, outlier_series.values, 
                color='#F44336', edgecolor='black', alpha=0.7)
        plt.title('Outliers Detected by Column', fontsize=14, fontweight='bold')
        plt.xlabel('Columns')
        plt.ylabel('Number of Outliers')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path / f'12_outlier_summary.{fmt}', dpi=dpi, bbox_inches='tight')
        plt.close()


# ============================================================
# Additional: Generate Preprocessing Code
# ============================================================
def generate_preprocessing_code(df):
    """Generate preprocessing code based on data profile"""
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    
    # Build the code as a string with proper variable scoping
    code_lines = []
    
    # Header
    code_lines.append(f"""
# 🧹 Auto-Generated Preprocessing Pipeline
# Generated from {len(df)} rows, {len(df.columns)} columns

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# 1. Load Data
df = pd.read_csv('data/your_dataset.csv')
print(f"Loaded: {{df.shape}}")

# 2. Identify Column Types
numeric_cols = {numeric_cols}
categorical_cols = {cat_cols}
boolean_cols = {bool_cols}

print(f"Numeric: {{len(numeric_cols)}} columns")
print(f"Categorical: {{len(categorical_cols)}} columns")
print(f"Boolean: {{len(boolean_cols)}} columns")

# 3. Handle Missing Values
print("\\n📊 Handling missing values...")
""")
    
    # Handle numeric columns
    if numeric_cols:
        code_lines.append("for col in numeric_cols:")
        code_lines.append("    if df[col].isna().sum() > 0:")
        code_lines.append("        df[col].fillna(df[col].median(), inplace=True)")
        code_lines.append("        print(f'  ✓ {col}: filled with median')")
    
    # Handle categorical columns
    if cat_cols:
        code_lines.append("")
        code_lines.append("for col in categorical_cols:")
        code_lines.append("    if df[col].isna().sum() > 0:")
        code_lines.append("        df[col].fillna(df[col].mode()[0], inplace=True)")
        code_lines.append("        print(f'  ✓ {col}: filled with mode')")
    
    # Encoding
    if cat_cols:
        code_lines.append("""
# 4. Encode Categorical Variables
print("\\n🔢 Encoding categorical variables...")
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    print(f"  ✓ {col}: encoded")
""")
    
    # Scaling
    if numeric_cols:
        code_lines.append("""
# 5. Scale Numeric Features
print("\\n📏 Scaling numeric features...")
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print(f"  ✓ {len(numeric_cols)} numeric columns scaled")
""")
    
    # Final
    code_lines.append("""
# 6. Prepare for Modeling
# If you have a target column, uncomment and modify below:
# target_column = 'target'  # Replace with your target
# X = df.drop(target_column, axis=1)
# y = df[target_column]
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
# print(f"\\n✅ Train: {X_train.shape}, Test: {X_test.shape}")

print(f"\\n✅ Preprocessing complete!")
print(f"   Final shape: {df.shape}")
print(f"   Memory: {df.memory_usage().sum() / 1024**2:.2f} MB")
""")
    
    return "\n".join(code_lines)
