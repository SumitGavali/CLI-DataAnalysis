import sys
import subprocess
import pytest
import pandas as pd
import numpy as np
from kaggle_prep.pipeline_exporter import export_pipeline_code


def test_pipeline_exporter_and_equivalence(tmp_path):
    np.random.seed(42)
    df = pd.DataFrame({
        "num1": np.random.randn(50),
        "num2": np.random.randn(50),
        "cat1": np.random.choice(["A", "B", "C"], 50),
        "target": np.random.choice([0, 1], 50)
    })

    p_py, t_py = export_pipeline_code(
        df=df,
        target_col="target",
        task_type="binary_classification",
        output_dir=str(tmp_path)
    )

    assert p_py.exists()
    assert t_py.exists()

    # Add tmp_path to sys.path to dynamically import pipeline.py
    sys.path.insert(0, str(tmp_path))
    try:
        import pipeline

        X = df.drop(columns=["target"])
        y = df["target"]

        fitted_pipe = pipeline.train(X, y)
        preds = pipeline.predict(fitted_pipe, X)

        assert len(preds) == len(df)
        assert not np.isnan(preds).any()

        # Run pytest on generated test_pipeline.py
        result = subprocess.run([sys.executable, "-m", "pytest", str(t_py)], capture_output=True, text=True)
        assert result.returncode == 0, f"Generated test failed: {result.stdout} {result.stderr}"

    finally:
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))
