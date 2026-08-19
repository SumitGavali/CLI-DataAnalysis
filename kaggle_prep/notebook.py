# kaggle_prep/notebook.py
"""
Jupyter Notebook Generator - Creates structured starter notebooks
"""

import json
from pathlib import Path
from typing import Dict, List
import pandas as pd


def generate_notebook(
    dataset_name: str,
    df: pd.DataFrame = None,
    profile: Dict = None,
    output_dir: str = "notebooks"
) -> Path:
    """
    Generate a complete Jupyter notebook with starter analysis code
    
    Args:
        dataset_name: Name of the dataset
        df: DataFrame (optional, for data insights)
        profile: Data profile dict (optional)
        output_dir: Directory to save the notebook
    
    Returns:
        Path to the generated notebook file
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
        warnings=warnings
    )
    
    # Save notebook
    safe_name = dataset_name.replace('/', '_')
    notebook_path = output_path / f"{safe_name}_analysis.ipynb"
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        f.write(notebook_content)
    
    print(f"📓 Starter notebook generated: {notebook_path}")
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
    warnings: List[str]
) -> str:
    """
    Generate the actual notebook JSON content
    """
    
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
                    f"# 📊 {dataset_name} - Data Analysis\n",
                    "\n",
                    "## 📋 Dataset Overview\n",
                    "\n",
                    f"- **Dataset:** {dataset_name}\n",
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
                    "# 📦 Import Required Libraries\n",
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
                    "plt.rcParams['font.size'] = 12\n",
                    "\n",
                    "# Display all columns\n",
                    "pd.set_option('display.max_columns', None)\n",
                    "pd.set_option('display.width', None)\n",
                    "\n",
                    "print('✅ Libraries imported successfully!')\n",
                    "print(f'🐼 Pandas version: {pd.__version__}')\n",
                    "print(f'🔢 NumPy version: {np.__version__}')\n",
                    "print(f'📊 Seaborn version: {sns.__version__}')"
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
                    "# 📂 Load Data\n",
                    "# Update this path to your actual data file\n",
                    "df = pd.read_csv('data/your_dataset.csv')\n",
                    "\n",
                    "print(f\"📊 Dataset loaded successfully!\")\n",
                    "print(f\"📏 Shape: {df.shape}\")\n",
                    "print(f\"💾 Memory: {df.memory_usage().sum() / 1024**2:.2f} MB\")\n",
                    "\n",
                    "# Preview first few rows\n",
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
                    "# ℹ️ Data Information\n",
                    "print(\"🔍 Data Types and Memory Usage:\")\n",
                    "df.info()\n",
                    "\n",
                    "print(\"\\n\" + \"=\"*70)\n",
                    "print(\"📊 Summary Statistics:\")\n",
                    "print(\"=\"*70)\n",
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
                    "# ❓ Missing Values Analysis\n",
                    "missing_df = pd.DataFrame({\n",
                    "    'Column': df.columns,\n",
                    "    'Missing': df.isna().sum().values,\n",
                    "    'Missing %': (df.isna().sum() / len(df) * 100).round(2).values\n",
                    "}).sort_values('Missing', ascending=False)\n",
                    "\n",
                    "print(\"📊 Missing Values Summary:\")\n",
                    "print(\"=\"*70)\n",
                    "print(missing_df[missing_df['Missing'] > 0])\n",
                    "\n",
                    "# Visualize missing values\n",
                    "if df.isna().sum().sum() > 0:\n",
                    "    plt.figure(figsize=(12, 6))\n",
                    "    sns.heatmap(df.isna(), yticklabels=False, cbar=True, cmap='viridis')\n",
                    "    plt.title('Missing Values Heatmap')\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n",
                    "else:\n",
                    "    print(\"✅ No missing values found in the dataset!\")"
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
                    "# 🔁 Duplicate Analysis\n",
                    "duplicates = df.duplicated().sum()\n",
                    "print(f\"🔁 Duplicate rows: {duplicates:,}\")\n",
                    "\n",
                    "if duplicates > 0:\n",
                    "    print(f\"\\n📊 Duplicate rows represent {duplicates/len(df)*100:.2f}% of the data\")\n",
                    "    print(\"\\nExample of duplicate rows:\")\n",
                    "    display(df[df.duplicated(keep='first')].head())\n",
                    "else:\n",
                    "    print(\"✅ No duplicate rows found!\")"
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
                    "# 📈 Distribution Analysis\n",
                    "\n",
                    "# Numeric columns\n",
                    "numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()\n",
                    "print(f\"📊 Numeric columns: {len(numeric_cols)}\")\n",
                    "print(numeric_cols)\n",
                    "\n",
                    "if numeric_cols:\n",
                    "    # Plot distributions\n",
                    "    fig, axes = plt.subplots((len(numeric_cols) + 1) // 2, 2, figsize=(14, 5*((len(numeric_cols)+1)//2)))\n",
                    "    axes = axes.flatten() if len(numeric_cols) > 1 else [axes]\n",
                    "    \n",
                    "    for i, col in enumerate(numeric_cols):\n",
                    "        axes[i].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)\n",
                    "        axes[i].set_title(f'Distribution of {col}')\n",
                    "        axes[i].set_xlabel(col)\n",
                    "        axes[i].set_ylabel('Frequency')\n",
                    "    \n",
                    "    # Hide empty subplots\n",
                    "    for j in range(i+1, len(axes)):\n",
                    "        axes[j].set_visible(False)\n",
                    "    \n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n",
                    "\n",
                    "# Categorical columns\n",
                    "cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()\n",
                    "print(f\"\\n📊 Categorical columns: {len(cat_cols)}\")\n",
                    "print(cat_cols)\n",
                    "\n",
                    "if cat_cols:\n",
                    "    for col in cat_cols[:3]:  # Limit to first 3 categorical columns\n",
                    "        plt.figure(figsize=(10, 6))\n",
                    "        df[col].value_counts().plot(kind='bar', edgecolor='black')\n",
                    "        plt.title(f'Distribution of {col}')\n",
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
                    "# 🔗 Correlation Analysis\n",
                    "if len(numeric_cols) > 1:\n",
                    "    correlation_matrix = df[numeric_cols].corr()\n",
                    "    \n",
                    "    plt.figure(figsize=(10, 8))\n",
                    "    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)\n",
                    "    plt.title('Feature Correlation Matrix')\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n",
                    "    \n",
                    "    # Find highly correlated features\n",
                    "    high_corr = []\n",
                    "    for i in range(len(correlation_matrix.columns)):\n",
                    "        for j in range(i+1, len(correlation_matrix.columns)):\n",
                    "            corr_val = correlation_matrix.iloc[i, j]\n",
                    "            if abs(corr_val) > 0.7:\n",
                    "                high_corr.append((\n",
                    "                    correlation_matrix.columns[i],\n",
                    "                    correlation_matrix.columns[j],\n",
                    "                    corr_val\n",
                    "                ))\n",
                    "    \n",
                    "    if high_corr:\n",
                    "        print(\"\\n⚠️ Highly correlated features (|corr| > 0.7):\")\n",
                    "        for col1, col2, corr in high_corr:\n",
                    "            print(f\"  • {col1} ↔ {col2}: {corr:.3f}\")\n",
                    "    else:\n",
                    "        print(\"\\n✅ No highly correlated features found\")\n",
                    "else:\n",
                    "    print(\"⚠️ Need at least 2 numeric columns for correlation analysis\")"
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
                    "# 🔍 Outlier Detection\n",
                    "def detect_outliers_iqr(data, column):\n",
                    "    Q1 = data[column].quantile(0.25)\n",
                    "    Q3 = data[column].quantile(0.75)\n",
                    "    IQR = Q3 - Q1\n",
                    "    lower_bound = Q1 - 1.5 * IQR\n",
                    "    upper_bound = Q3 + 1.5 * IQR\n",
                    "    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]\n",
                    "    return outliers, lower_bound, upper_bound\n",
                    "\n",
                    "outlier_summary = []\n",
                    "for col in numeric_cols:\n",
                    "    outliers, lower, upper = detect_outliers_iqr(df, col)\n",
                    "    outlier_count = len(outliers)\n",
                    "    outlier_pct = (outlier_count / len(df)) * 100\n",
                    "    if outlier_count > 0:\n",
                    "        outlier_summary.append({\n",
                    "            'Column': col,\n",
                    "            'Outliers': outlier_count,\n",
                    "            'Outlier %': round(outlier_pct, 2),\n",
                    "            'Lower Bound': round(lower, 2),\n",
                    "            'Upper Bound': round(upper, 2)\n",
                    "        })\n",
                    "\n",
                    "if outlier_summary:\n",
                    "    print(\"📊 Outlier Summary:\")\n",
                    "    print(\"=\"*70)\n",
                    "    outlier_df = pd.DataFrame(outlier_summary)\n",
                    "    print(outlier_df)\n",
                    "else:\n",
                    "    print(\"✅ No outliers detected using IQR method!\")"
                ]
            },
            
            # ============================================================
            # Cell 10: Preprocessing
            # ============================================================
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# 🧹 Preprocessing Pipeline\n",
                    "def preprocess_data(df, target_col=None):\n",
                    "    \"\"\"\n",
                    "    Preprocess the data for machine learning\n",
                    "    \"\"\"\n",
                    "    data = df.copy()\n",
                    "    \n",
                    "    # Separate features and target\n",
                    "    if target_col:\n",
                    "        X = data.drop(target_col, axis=1)\n",
                    "        y = data[target_col]\n",
                    "    else:\n",
                    "        X = data\n",
                    "        y = None\n",
                    "    \n",
                    "    # Identify column types\n",
                    "    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()\n",
                    "    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()\n",
                    "    \n",
                    "    print(f\"🔢 Numeric columns: {len(numeric_cols)}\")\n",
                    "    print(f\"📝 Categorical columns: {len(categorical_cols)}\")\n",
                    "    \n",
                    "    # Handle missing values\n",
                    "    for col in numeric_cols:\n",
                    "        X[col].fillna(X[col].median(), inplace=True)\n",
                    "    \n",
                    "    for col in categorical_cols:\n",
                    "        X[col].fillna(X[col].mode()[0], inplace=True)\n",
                    "    \n",
                    "    # Encode categorical variables\n",
                    "    for col in categorical_cols:\n",
                    "        le = LabelEncoder()\n",
                    "        X[col] = le.fit_transform(X[col].astype(str))\n",
                    "    \n",
                    "    # Scale numeric features\n",
                    "    if numeric_cols:\n",
                    "        scaler = StandardScaler()\n",
                    "        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])\n",
                    "    \n",
                    "    print(f\"✅ Preprocessing complete!\")\n",
                    "    print(f\"📏 Shape: {X.shape}\")\n",
                    "    \n",
                    "    return X, y\n",
                    "\n",
                    "# Example usage (uncomment and modify target column)\n",
                    "# target_column = 'target'  # Replace with your target column name\n",
                    "# X_processed, y_processed = preprocess_data(df, target_col=target_column)"
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
                    "# 🤖 Model Training\n",
                    "from sklearn.ensemble import RandomForestClassifier\n",
                    "from sklearn.linear_model import LogisticRegression\n",
                    "from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score\n",
                    "\n",
                    "def train_models(X_train, X_test, y_train, y_test):\n",
                    "    \"\"\"\n",
                    "    Train and evaluate multiple models\n",
                    "    \"\"\"\n",
                    "    models = {\n",
                    "        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),\n",
                    "        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)\n",
                    "    }\n",
                    "    \n",
                    "    results = {}\n",
                    "    \n",
                    "    for name, model in models.items():\n",
                    "        print(f\"\\n{'='*70}\")\n",
                    "        print(f\"🎯 Training: {name}\")\n",
                    "        print('='*70)\n",
                    "        \n",
                    "        # Train\n",
                    "        model.fit(X_train, y_train)\n",
                    "        \n",
                    "        # Predict\n",
                    "        y_pred = model.predict(X_test)\n",
                    "        \n",
                    "        # Evaluate\n",
                    "        accuracy = accuracy_score(y_test, y_pred)\n",
                    "        results[name] = accuracy\n",
                    "        \n",
                    "        print(f\"✅ Accuracy: {accuracy:.4f}\")\n",
                    "        print(f\"\\n📊 Classification Report:\")\n",
                    "        print(classification_report(y_test, y_pred))\n",
                    "        \n",
                    "        # Confusion Matrix\n",
                    "        cm = confusion_matrix(y_test, y_pred)\n",
                    "        plt.figure(figsize=(8, 6))\n",
                    "        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\n",
                    "        plt.title(f'Confusion Matrix - {name}')\n",
                    "        plt.ylabel('Actual')\n",
                    "        plt.xlabel('Predicted')\n",
                    "        plt.tight_layout()\n",
                    "        plt.show()\n",
                    "    \n",
                    "    # Print best model\n",
                    "    print(f\"\\n{'='*70}\")\n",
                    "    print(\"🏆 Best Model:\")\n",
                    "    print('='*70)\n",
                    "    best_model = max(results, key=results.get)\n",
                    "    print(f\"{best_model}: {results[best_model]:.4f} accuracy\")\n",
                    "    \n",
                    "    return results\n",
                    "\n",
                    "# Example usage (uncomment after preprocessing)\n",
                    "# X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)\n",
                    "# train_models(X_train, X_test, y_train, y_test)"
                ]
            },
            
            # ============================================================
            # Cell 12: Feature Importance
            # ============================================================
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# 📊 Feature Importance\n",
                    "def plot_feature_importance(model, feature_names):\n",
                    "    \"\"\"\n",
                    "    Plot feature importance for tree-based models\n",
                    "    \"\"\"\n",
                    "    if hasattr(model, 'feature_importances_'):\n",
                    "        importance = model.feature_importances_\n",
                    "        \n",
                    "        # Create DataFrame\n",
                    "        feature_importance_df = pd.DataFrame({\n",
                    "            'Feature': feature_names,\n",
                    "            'Importance': importance\n",
                    "        }).sort_values('Importance', ascending=False)\n",
                    "        \n",
                    "        # Plot\n",
                    "        plt.figure(figsize=(10, 6))\n",
                    "        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])\n",
                    "        plt.xlabel('Feature Importance')\n",
                    "        plt.title('Feature Importance Analysis')\n",
                    "        plt.gca().invert_yaxis()\n",
                    "        plt.tight_layout()\n",
                    "        plt.show()\n",
                    "        \n",
                    "        return feature_importance_df\n",
                    "    else:\n",
                    "        print(\"⚠️ This model doesn't provide feature importance\")\n",
                    "        return None\n",
                    "\n",
                    "# Example usage (uncomment after training)\n",
                    "# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
                    "# rf_model.fit(X_train, y_train)\n",
                    "# plot_feature_importance(rf_model, X_train.columns)"
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