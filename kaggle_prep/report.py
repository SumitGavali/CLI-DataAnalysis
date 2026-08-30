# kaggle_prep/report.py
"""
Self-contained HTML Report Generator
"""

from pathlib import Path
import json
import time
from typing import Dict


def generate_standalone_report(profile: Dict, output_dir: str = "reports"):
    """Generate a beautiful HTML report with zero dependencies"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dataset_name = profile['dataset'].replace('/', '_')
    html_path = output_path / f"{dataset_name}_report.html"
    
    # Build HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Profile Report - {profile['dataset']}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: #f7f9fc;
            padding: 40px;
            line-height: 1.6;
            color: #1e293b;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 8px;
            padding: 36px;
            box-shadow: 0 1px 6px rgba(0,0,0,0.06);
            border: 1px solid #e2e8f0;
        }}
        h1 {{
            color: #0f172a;
            font-size: 24px;
            font-weight: 700;
            border-bottom: 2px solid #3b82f6;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        h2 {{
            color: #1e293b;
            margin: 28px 0 14px 0;
            font-size: 18px;
            font-weight: 600;
            border-left: 3px solid #3b82f6;
            padding-left: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            margin: 16px 0;
        }}
        .stat-card {{
            background: #f8fafc;
            padding: 14px 16px;
            border-radius: 6px;
            border: 1px solid #e2e8f0;
        }}
        .stat-value {{
            font-size: 22px;
            font-weight: 700;
            color: #0f172a;
        }}
        .stat-label {{
            color: #64748b;
            font-size: 13px;
            margin-top: 2px;
        }}
        .stat-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            margin-left: 6px;
        }}
        .badge-success {{ background: #dcfce7; color: #15803d; }}
        .badge-warning {{ background: #fef3c7; color: #b45309; }}
        .badge-danger {{ background: #fee2e2; color: #b91c1c; }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 13px;
        }}
        th {{
            background: #f1f5f9;
            padding: 10px 12px;
            text-align: left;
            font-weight: 600;
            color: #334155;
            border-bottom: 2px solid #cbd5e1;
        }}
        td {{
            padding: 9px 12px;
            border-bottom: 1px solid #e2e8f0;
        }}
        tr:hover {{
            background: #f8fafc;
        }}
        
        .warning-box {{
            background: #fffbeb;
            border-left: 3px solid #d97706;
            padding: 14px;
            border-radius: 4px;
            margin: 12px 0;
            font-size: 13px;
        }}
        .suggestion-box {{
            background: #eff6ff;
            border-left: 3px solid #2563eb;
            padding: 14px;
            border-radius: 4px;
            margin: 12px 0;
            font-size: 13px;
        }}
        .success-box {{
            background: #f0fdf4;
            border-left: 3px solid #16a34a;
            padding: 14px;
            border-radius: 4px;
            margin: 12px 0;
            font-size: 13px;
        }}
        
        .meta-info {{
            color: #64748b;
            font-size: 13px;
            margin: 10px 0 20px 0;
        }}
        
        .footer {{
            margin-top: 36px;
            padding-top: 16px;
            border-top: 1px solid #e2e8f0;
            color: #94a3b8;
            font-size: 12px;
            text-align: center;
        }}
        
        @media (max-width: 768px) {{
            body {{ padding: 12px; }}
            .container {{ padding: 16px; }}
            .stats-grid {{ grid-template-columns: 1fr 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Data Profile Report</h1>
        
        <div class="meta-info">
            <strong>Dataset:</strong> {profile['dataset']} | 
            <strong>Generated:</strong> {profile['generated_at']}
        </div>
        
        <h2>Overview</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{profile['basic_stats']['rows']:,}</div>
                <div class="stat-label">Total Rows</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{profile['basic_stats']['columns']}</div>
                <div class="stat-label">Total Columns</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{profile['basic_stats']['missing_percent']:.1f}%</div>
                <div class="stat-label">Missing Data</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{profile['basic_stats']['memory_mb']:.1f} MB</div>
                <div class="stat-label">Memory Usage</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{profile['basic_stats']['duplicates']:,}</div>
                <div class="stat-label">Duplicate Rows</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{profile['basic_stats']['num_cols']}</div>
                <div class="stat-label">Numeric Features</div>
            </div>
        </div>
        
        <h2>Column Analysis</h2>
        <table>
            <thead>
                <tr>
                    <th>Column</th>
                    <th>Type</th>
                    <th>Missing</th>
                    <th>Unique</th>
                    <th>Statistics</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Add column rows
    for col, stats in profile['columns'].items():
        missing_badge = ""
        if stats['missing_percent'] > 30:
            missing_badge = '<span class="stat-badge badge-danger">High</span>'
        elif stats['missing_percent'] > 10:
            missing_badge = '<span class="stat-badge badge-warning">Moderate</span>'
        
        if 'mean' in stats:
            stats_text = f"Mean={stats['mean']:.2f}, Std={stats['std']:.2f}"
            if stats.get('outliers', 0) > 0:
                stats_text += f", <span style='color:#dc2626;'>{stats['outliers']} outliers</span>"
        else:
            top_vals = list(stats.get('top_values', {}).items())[:2]
            stats_text = ", ".join([f'"{k}"' for k, v in top_vals]) if top_vals else "-"
        
        html += f"""
            <tr>
                <td><strong>{col}</strong></td>
                <td><code>{stats['dtype']}</code></td>
                <td>{stats['missing_percent']:.1f}% {missing_badge}</td>
                <td>{stats['unique']:,}</td>
                <td>{stats_text}</td>
            </tr>
        """
    
    html += """
            </tbody>
        </table>
    """
    
    # Warnings section
    if profile['warnings']:
        html += """
        <h2>Data Quality Warnings</h2>
        <div class="warning-box">
            <ul>
        """
        for warning in profile['warnings']:
            html += f"<li>{warning}</li>"
        html += """
            </ul>
        </div>
        """
    else:
        html += """
        <h2>Data Quality</h2>
        <div class="success-box">
            No data quality issues detected.
        </div>
        """
    
    # Suggestions section
    if profile['suggestions']:
        html += """
        <h2>Recommendations</h2>
        <div class="suggestion-box">
            <ul>
        """
        for suggestion in profile['suggestions']:
            html += f"<li>{suggestion}</li>"
        html += """
            </ul>
        </div>
        """
    
    html += f"""
        <div class="footer">
            Generated by kaggle-prep | {profile['generated_at']}
        </div>
    </div>
</body>
</html>
    """
    
    # Write HTML file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f" HTML report saved to: {html_path}")
    print(f" Open in browser: file://{html_path.absolute()}")
    
    return html_path