import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import json

from kaggle_prep.notebook import generate_notebook


def test_generate_notebook(tmp_path):
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [10.0, 20.0, 30.0, 40.0],
        "c": ["x", "y", "x", "y"]
    })

    nb_path = generate_notebook(
        dataset_name="test/iris",
        df=df,
        output_dir=str(tmp_path)
    )

    assert Path(nb_path).exists()
    assert nb_path.suffix == ".ipynb"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_json = json.load(f)

    assert "cells" in nb_json
    assert len(nb_json["cells"]) >= 10
    assert nb_json["nbformat"] == 4
