# kaggle_prep/notebook.py
"""
Jupyter Notebook Generator - Creates structured starter notebooks
"""

import json
from pathlib import Path
from typing import Dict, List
import pandas as pd


from .intelligence import (
    detect_task_type,
    analyze_class_balance,
    recommend_metrics,
    determine_split_strategy,
)


def generate_notebook(
    dataset_name: str,
    df: pd.DataFrame = None,
    profile: Dict = None,
    target: str = None,
    output_dir: str = "notebooks"
) -> Path:
    """
    Generate a complete Jupyter notebook with starter analysis code
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get dataset info if df is provided
    if df is not None:
        num_rows = len(df)
        num_cols = len(df.columns)
        num_numeric = len(df.select_dtypes(include=['int64', 'float64']).columns)
        num_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
        missing_cells = int(df.isna().sum().sum())
        missing_pct = round((missing_cells / (num_rows * num_cols)) * 100, 2) if num_rows > 0 else 0
        duplicate_rows = int(df.duplicated().sum())
    else:
        num_rows = "?"
        num_cols = "?"
        num_numeric = "?"
        num_categorical = "?"
        missing_cells = "?"
        missing_pct = "?"
        duplicate_rows = "?"
    
    # Get profile info if provided
    suggestions = profile.get('suggestions', []) if profile else []
    warnings = profile.get('warnings', []) if profile else []
    
    # Get column names for code generation
    if df is not None:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = df.columns.tolist()
    else:
        numeric_cols = []
        cat_cols = []
        all_cols = []
    
    # Task intelligence
    task_type = None
    split_strategy = None
    recommended_metrics = None
    if df is not None and target and target in df.columns:
        task_type = detect_task_type(df, target)
        class_info = analyze_class_balance(df, target) if "classification" in task_type else None
        recommended_metrics = recommend_metrics(task_type, class_info)
        split_strategy = determine_split_strategy(task_type)

    # Generate the notebook content
    notebook_content = generate_notebook_content(
        dataset_name=dataset_name,
        num_rows=num_rows,
        num_cols=num_cols,
        num_numeric=num_numeric,
        num_categorical=num_categorical,
        missing_cells=missing_cells,
        missing_pct=missing_pct,
        duplicate_rows=duplicate_rows,
        numeric_cols=numeric_cols,
        cat_cols=cat_cols,
        all_cols=all_cols,
        suggestions=suggestions,
        warnings=warnings,
        target=target,
        task_type=task_type,
        split_strategy=split_strategy,
        recommended_metrics=recommended_metrics
    )
    
    # Save notebook
    safe_name = dataset_name.replace('/', '_')
    notebook_path = output_path / f"{safe_name}_analysis.ipynb"
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        f.write(notebook_content)
    
    print(f" Starter notebook generated: {notebook_path}")
    return notebook_path


def generate_notebook_content(
    dataset_name: str,
    num_rows,
    num_cols,
    num_numeric,
    num_categorical,
    missing_cells,
    missing_pct,
    duplicate_rows,
    numeric_cols: List[str],
    cat_cols: List[str],
    all_cols: List[str],
    suggestions: List[str],
    warnings: List[str],
    target: str = None,
    task_type: str = None,
    split_strategy: Dict = None,
    recommended_metrics: List[str] = None
) -> str:
    """
    Generate the actual notebook JSON content
    """
    target_str = f"'{target}'" if target else "'target'"
    split_code = split_strategy["code"] if split_strategy else "train_test_split(X, y, test_size=0.2, random_state=42)"

    # Build the notebook as a JSON structure
    notebook = {
        "cells": [
            # ============================================================
            # Cell 1: Title and Setup
            # ============================================================
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {dataset_name} - Exploratory Analysis\n",
                    "\n",
                    "## Dataset Overview\n",
                    "\n",
                    f"- **Dataset:** `{dataset_name}`\n",
                    f"- **Rows:** {num_rows:,}\n",
                    f"- **Columns:** {num_cols}\n",
                    f"- **Numeric Features:** {num_numeric}\n",
                    f"- **Categorical Features:** {num_categorical}\n",
                    f"- **Missing Values:** {missing_cells:,} ({missing_pct}%)\n",
                    f"- **Duplicate Rows:** {duplicate_rows:,}\n",
                    "\n",
                    "---\n"
                ]
            },
            
            # ============================================================
            # Cell 2: Import Libraries
            # ============================================================
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# 1. Import Required Libraries\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
                    "from sklearn.metrics import accuracy_score, classification_report\n",
                    "\n",
                    "# Set visualization style\n",
                    "sns.set_style('whitegrid')\n",
                    "plt.rcParams['figure.figsize'] = (12, 8)\n",
                    "plt.rcParams['font.size'] = 11\n",
                    "\n",
                    "# Display settings\n",
                    "pd.set_option('display.max_columns', None)\n",
                    "pd.set_option('display.width', None)\n",
                    "\n",
                    "print(f'Pandas: {pd.__version__} | NumPy: {np.__version__} | Seaborn: {sns.__version__}')"
                ]
            },
            
            # ============================================================
            # Cell 3: Load Data
            # ============================================================
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# 2. Load Dataset\n",
                    "# Update file path if using a custom location\n",
                    "df = pd.read_csv('data/your_dataset.csv')\n",
                    "\n",
                    "print(f\"Loaded shape: {df.shape}\")\n",
                    "print(f\"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB\")\n",
                    "\n",
                    "df.head()"
                ]
            },
            
            # ============================================================
            # Cell 4: Data Information
            # ============================================================
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# 3. Data Structure and Summary Statistics\n",
                    "print(\"Column Data Types:\")\n",
                    "df.info()\n",
                    "\n",
                    "print(\"\\nSummary Statistics:\")\n",
                    "df.describe(include='all').transpose()"
                ]
            },
            
            # ============================================================
            # Cell 5: Missing Values Analysis
            # ============================================================
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# 4. Missing Values Analysis\n",
                    "missing_df = pd.DataFrame({\n",
                    "    'Column': df.columns,\n",
                    "    'Missing': df.isna().sum().values,\n",
                    "    'Missing %': (df.isna().sum() / len(df) * 100).round(2).values\n",
                    "}).sort_values('Missing', ascending=False)\n",
                    "\n",
                    "missing_cols = missing_df[missing_df['Missing'] > 0]\n",
                    "if not missing_cols.empty:\n",
                    "    print(\"Columns with Missing Values:\")\n",
                    "    print(missing_cols)\n",
                    "    plt.figure(figsize=(12, 6))\n",
                    "    sns.heatmap(df.isna(), yticklabels=False, cbar=True, cmap='viridis')\n",
                    "    plt.title('Missing Values Heatmap')\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n",
                    "else:\n",
                    "    print(\"No missing values found in dataset.\")"
                ]
            },
            
            # ============================================================
            # Cell 6: Duplicate Analysis
            # ============================================================
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# 5. Duplicate Row Analysis\n",
                    "duplicates = df.duplicated().sum()\n",
                    "print(f\"Duplicate rows: {duplicates:,} ({duplicates/len(df)*100:.2f}%)\")\n",
                    "if duplicates > 0:\n",
                    "    display(df[df.duplicated(keep='first')].head())"
                ]
            },
            
            # ============================================================
            # Cell 7: Distribution Plots
            # ============================================================
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# 6. Feature Distributions\n",
                    "numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()\n",
                    "cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()\n",
                    "\n",
                    "if numeric_cols:\n",
                    "    n_rows = (len(numeric_cols) + 1) // 2\n",
                    "    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))\n",
                    "    axes = axes.flatten() if len(numeric_cols) > 1 else [axes]\n",
                    "    for i, col in enumerate(numeric_cols):\n",
                    "        axes[i].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)\n",
                    "        axes[i].set_title(f'Distribution: {col}')\n",
                    "        axes[i].set_xlabel(col)\n",
                    "        axes[i].set_ylabel('Frequency')\n",
                    "    for j in range(i + 1, len(axes)):\n",
                    "        axes[j].set_visible(False)\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n",
                    "\n",
                    "if cat_cols:\n",
                    "    for col in cat_cols[:4]:\n",
                    "        plt.figure(figsize=(10, 5))\n",
                    "        df[col].value_counts().head(10).plot(kind='bar', edgecolor='black')\n",
                    "        plt.title(f'Top Categories: {col}')\n",
                    "        plt.xlabel(col)\n",
                    "        plt.ylabel('Count')\n",
                    "        plt.xticks(rotation=45)\n",
                    "        plt.tight_layout()\n",
                    "        plt.show()"
                ]
            },
            
            # ============================================================
            # Cell 8: Correlation Analysis
            # ============================================================
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# 7. Correlation Analysis\n",
                    "if len(numeric_cols) > 1:\n",
                    "    correlation_matrix = df[numeric_cols].corr()\n",
                    "    plt.figure(figsize=(10, 8))\n",
                    "    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)\n",
                    "    plt.title('Feature Correlation Matrix')\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n",
                    "else:\n",
                    "    print(\"Correlation analysis requires at least 2 numeric features.\")"
                ]
            },
            
            # ============================================================
            # Cell 9: Outlier Detection
            # ============================================================
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# 8. Outlier Detection (IQR Method)\n",
                    "def detect_outliers_iqr(data, column):\n",
                    "    q1 = data[column].quantile(0.25)\n",
                    "    q3 = data[column].quantile(0.75)\n",
                    "    iqr = q3 - q1\n",
                    "    lower_bound = q1 - 1.5 * iqr\n",
                    "    upper_bound = q3 + 1.5 * iqr\n",
                    "    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]\n",
                    "    return outliers, lower_bound, upper_bound\n",
                    "\n",
                    "outlier_summary = []\n",
                    "for col in numeric_cols:\n",
                    "    outliers, lower, upper = detect_outliers_iqr(df, col)\n",
                    "    if len(outliers) > 0:\n",
                    "        outlier_summary.append({\n",
                    "            'Column': col,\n",
                    "            'Outliers': len(outliers),\n",
                    "            'Outlier %': round((len(outliers) / len(df)) * 100, 2),\n",
                    "            'Lower Bound': round(lower, 2),\n",
                    "            'Upper Bound': round(upper, 2)\n",
                    "        })\n",
                    "\n",
                    "if outlier_summary:\n",
                    "    print(pd.DataFrame(outlier_summary))\n",
                    "else:\n",
                    "    print(\"No outliers detected using the 1.5 x IQR rule.\")"
                ]
            },
            
            # ============================================================
            # Cell 10: Preprocessing & Train/Test Split
            # ============================================================
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# 9. Preprocessing & Train/Test Split\n",
                    "def preprocess_data(df, target_col=None):\n",
                    "    data = df.copy()\n",
                    "    if target_col and target_col in data.columns:\n",
                    "        X = data.drop(target_col, axis=1)\n",
                    "        y = data[target_col]\n",
                    "    else:\n",
                    "        X = data\n",
                    "        y = None\n",
                    "    \n",
                    "    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()\n",
                    "    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()\n",
                    "    \n",
                    "    for col in num_cols:\n",
                    "        X[col].fillna(X[col].median(), inplace=True)\n",
                    "    for col in cat_cols:\n",
                    "        X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'missing', inplace=True)\n",
                    "        le = LabelEncoder()\n",
                    "        X[col] = le.fit_transform(X[col].astype(str))\n",
                    "    \n",
                    "    if num_cols:\n",
                    "        scaler = StandardScaler()\n",
                    "        X[num_cols] = scaler.fit_transform(X[num_cols])\n",
                    "    \n",
                    "    return X, y\n",
                    "\n",
                    f"target_column = {target_str}\n",
                    "if target_column in df.columns:\n",
                    "    X, y = preprocess_data(df, target_col=target_column)\n",
                    f"    X_train, X_test, y_train, y_test = {split_code}\n",
                    "    print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')\n"
                ]
            },
            
            # ============================================================
            # Cell 11: Model Training
            # ============================================================
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# 10. Baseline Model Evaluation\n",
                    "if 'X_train' in locals() and y_train is not None:\n",
                    "    is_clf = " + ("True" if task_type and "classification" in task_type else "False") + "\n",
                    "    if is_clf:\n",
                    "        from sklearn.ensemble import RandomForestClassifier\n",
                    "        from sklearn.linear_model import LogisticRegression\n",
                    "        models = {\n",
                    "            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),\n",
                    "            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)\n",
                    "        }\n",
                    "        for name, model in models.items():\n",
                    "            model.fit(X_train, y_train)\n",
                    "            y_pred = model.predict(X_test)\n",
                    "            print(f'{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}')\n",
                    "            print(classification_report(y_test, y_pred))\n",
                    "    else:\n",
                    "        from sklearn.ensemble import RandomForestRegressor\n",
                    "        from sklearn.linear_model import LinearRegression\n",
                    "        from sklearn.metrics import mean_squared_error, r2_score\n",
                    "        models = {\n",
                    "            'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),\n",
                    "            'Linear Regression': LinearRegression()\n",
                    "        }\n",
                    "        for name, model in models.items():\n",
                    "            model.fit(X_train, y_train)\n",
                    "            y_pred = model.predict(X_test)\n",
                    "            rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n",
                    "            r2 = r2_score(y_test, y_pred)\n",
                    "            print(f'{name} RMSE: {rmse:.4f}, R2: {r2:.4f}')\n"
                ]
            }
        ],
        
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.9.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Convert to JSON string
    return json.dumps(notebook, indent=2, ensure_ascii=False)