# kaggle_prep/profiler.py
"""
Pure Python Data Profiler - No external dependencies!
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class DataProfiler:
    """Fast, dependency-free data profiler"""
    
    def __init__(self, df: pd.DataFrame, dataset_name: str):
        self.df = df
        self.dataset_name = dataset_name
        self._profile_data = None
        
    def profile(self) -> Dict[str, Any]:
        """Generate complete data profile"""
        
        self._profile_data = {
            'dataset': self.dataset_name,
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'basic_stats': self._get_basic_stats(),
            'columns': self._get_column_stats(),
            'warnings': self._get_warnings(),
            'suggestions': self._get_suggestions()
        }
        
        return self._profile_data
    
    def _get_basic_stats(self) -> Dict:
        """Basic dataset statistics"""
        df = self.df
        
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_mb': round(df.memory_usage().sum() / 1024**2, 2),
            'duplicates': int(df.duplicated().sum()),
            'missing_cells': int(df.isna().sum().sum()),
            'missing_percent': round(
                (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100, 2
            ),
            'num_cols': len(df.select_dtypes(include=['int64', 'float64']).columns),
            'cat_cols': len(df.select_dtypes(include=['object', 'category']).columns),
            'bool_cols': len(df.select_dtypes(include=['bool']).columns)
        }
    
    def _get_column_stats(self) -> Dict:
        """Detailed column analysis"""
        column_stats = {}
        
        for col in self.df.columns:
            col_data = self.df[col]
            
            stats = {
                'dtype': str(col_data.dtype),
                'missing': int(col_data.isna().sum()),
                'missing_percent': round((col_data.isna().sum() / len(col_data)) * 100, 2),
                'unique': int(col_data.nunique()),
                'unique_percent': round((col_data.nunique() / len(col_data)) * 100, 2)
            }
            
            # Numeric columns
            if pd.api.types.is_numeric_dtype(col_data):
                if col_data.count() > 0:
                    stats.update({
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'mean': float(col_data.mean()),
                        'median': float(col_data.median()),
                        'std': float(col_data.std()),
                        'skew': float(col_data.skew()),
                        'kurtosis': float(col_data.kurtosis()),
                        'q1': float(col_data.quantile(0.25)),
                        'q3': float(col_data.quantile(0.75)),
                        'iqr': float(col_data.quantile(0.75) - col_data.quantile(0.25))
                    })
                    
                    # Detect outliers
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    outliers = col_data[(col_data < q1 - 1.5 * iqr) | (col_data > q3 + 1.5 * iqr)]
                    stats['outliers'] = len(outliers)
                    stats['outlier_percent'] = round((len(outliers) / len(col_data)) * 100, 2)
                    
                else:
                    stats.update({
                        'min': None, 'max': None, 'mean': None,
                        'median': None, 'std': None, 'skew': None,
                        'kurtosis': None
                    })
            
            # Categorical columns
            else:
                # Top values
                top_values = col_data.value_counts().head(5)
                stats['top_values'] = {str(k): int(v) for k, v in top_values.items()}
                
                # For text columns, get length stats
                if col_data.dtype == 'object':
                    try:
                        str_lengths = col_data.dropna().astype(str).str.len()
                        if len(str_lengths) > 0:
                            stats.update({
                                'avg_length': float(str_lengths.mean()),
                                'min_length': int(str_lengths.min()),
                                'max_length': int(str_lengths.max())
                            })
                    except:
                        pass
            
            column_stats[col] = stats
        
        return column_stats
    
    def _get_warnings(self) -> List[str]:
        """Generate warnings about data quality"""
        warnings = []
        df = self.df
        
        for col in df.columns:
            # High missing values
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            if missing_pct > 30:
                warnings.append(
                    f"  {col}: {missing_pct:.1f}% missing values (consider imputation)"
                )
            
            # Constant columns
            if df[col].nunique() == 1:
                warnings.append(
                    f"  {col}: Constant column (all values are identical)"
                )
            
            # High cardinality
            if pd.api.types.is_object_dtype(df[col]):
                unique_pct = (df[col].nunique() / len(df)) * 100
                if unique_pct > 80:
                    warnings.append(
                        f"  {col}: Very high cardinality ({df[col].nunique():,} unique values)"
                    )
        
        # Dataset level warnings
        if df.duplicated().sum() > 0:
            warnings.append(
                f"  Dataset has {df.duplicated().sum():,} duplicate rows"
            )
        
        if df.isna().sum().sum() > 0:
            warnings.append(
                f"  Dataset has {df.isna().sum().sum():,} missing values"
            )
        
        return warnings
    
    def _get_suggestions(self) -> List[str]:
        """Generate actionable suggestions"""
        suggestions = []
        df = self.df
        
        # Missing data suggestions
        missing_cols = [col for col in df.columns if df[col].isna().sum() > 0]
        if missing_cols:
            suggestions.append(
                f" Consider imputing missing values in: {', '.join(missing_cols[:3])}"
            )
        
        # Numeric columns with high outliers
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            if len(df[col]) > 0:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                outliers = df[col][(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
                if len(outliers) > len(df) * 0.05:  # >5% outliers
                    suggestions.append(
                        f" Column '{col}' has {len(outliers):,} outliers - consider winsorizing"
                    )
        
        # Categorical suggestions
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            suggestions.append(
                f" Consider encoding categorical columns: {', '.join(cat_cols[:3])}"
            )
        
        # Scale suggestions
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 1:
            # Check if scaling is needed (ignore constant columns with range == 0)
            ranges = [float(df[col].max() - df[col].min()) for col in numeric_cols if len(df[col]) > 0]
            positive_ranges = [r for r in ranges if r > 0]
            if positive_ranges and len(positive_ranges) > 1:
                if max(positive_ranges) / min(positive_ranges) > 100:
                    suggestions.append(
                        " Columns have vastly different scales - consider standardization"
                    )
        
        return suggestions


def save_profile_json(profile: Dict, output_dir: str = "data_profiles"):
    """Save profile as JSON"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dataset_name = profile['dataset'].replace('/', '_')
    file_path = output_path / f"{dataset_name}_profile.json"
    
    with open(file_path, 'w') as f:
        json.dump(profile, f, indent=2)
    
    return file_path


def print_profile_summary(profile: Dict):
    """Print clean profile summary across all terminal encodings"""
    
    print("\n" + "=" * 70)
    print(" DATA PROFILE SUMMARY")
    print("=" * 70)
    
    stats = profile['basic_stats']
    print(f"\n Dataset: {profile['dataset']}")
    print(f" Shape: {stats['rows']:,} rows x {stats['columns']} columns")
    print(f" Memory: {stats['memory_mb']:.2f} MB")
    print(f" Duplicates: {stats['duplicates']:,}")
    print(f" Missing: {stats['missing_cells']:,} ({stats['missing_percent']:.2f}%)")
    print(f" Numeric: {stats['num_cols']} | Categorical: {stats['cat_cols']}")
    
    # Column details
    print("\n Column Analysis:")
    print("-" * 70)
    
    for col, col_stats in profile['columns'].items():
        print(f"\n  > {col} ({col_stats['dtype']})")
        print(f"    Missing: {col_stats['missing_percent']:.1f}%")
        print(f"    Unique: {col_stats['unique']:,} ({col_stats['unique_percent']:.1f}%)")
        
        if 'mean' in col_stats:
            print(f"    Mean: {col_stats['mean']:.2f} (+/- {col_stats['std']:.2f})")
            print(f"    Range: [{col_stats['min']:.2f}, {col_stats['max']:.2f}]")
            if col_stats.get('outliers', 0) > 0:
                print(f"    Outliers: {col_stats['outliers']} ({col_stats['outlier_percent']:.1f}%)")
        elif 'top_values' in col_stats:
            print("    Top values:")
            for val, count in list(col_stats['top_values'].items())[:3]:
                print(f"      * {val}: {count:,} ({count/stats['rows']*100:.1f}%)")
    
    # Warnings
    if profile['warnings']:
        print("\n  WARNINGS:")
        for warning in profile['warnings']:
            # Strip any unprintable leading character if present
            clean_w = warning.strip()
            print(f"  * {clean_w}")
    
    # Suggestions
    if profile['suggestions']:
        print("\n  SUGGESTIONS:")
        for suggestion in profile['suggestions']:
            clean_s = suggestion.strip()
            print(f"  * {clean_s}")
    
    print("\n" + "=" * 70)
    print(f" Full profile saved to: data_profiles/{profile['dataset'].replace('/', '_')}_profile.json")