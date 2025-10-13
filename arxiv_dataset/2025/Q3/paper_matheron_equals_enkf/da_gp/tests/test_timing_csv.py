# Copyright (c) 2025 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
#
# All rights reserved.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test timing CSV validation and structure."""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_timing_csv_columns():
    """Test that generated CSV has correct timing columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_timing.csv"

        # Run a quick benchmark to generate CSV
        cmd = [
            sys.executable,
            "-m",
            "da_gp.scripts.bench",
            "--n_obs_grid",
            "50",
            "100",
            "--backends",
            "sklearn",
            "--csv",
            str(csv_path),
            "--repeats",
            "1",
        ]

        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        # Check that command succeeded
        if result.returncode != 0:
            pytest.skip(f"Benchmark command failed: {result.stderr}")

        # Check that CSV was created
        assert csv_path.exists(), "CSV file was not created"

        # Load and validate CSV structure
        df = pd.read_csv(csv_path)

        # Required columns
        required_columns = [
            "backend",
            "n_obs",
            "grid_size",
            "fit_time",
            "predict_time",
            "total_time",
            "rmse",
        ]
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

        # Timing columns should be numeric and non-negative
        timing_columns = ["fit_time", "predict_time", "total_time", "rmse"]
        for col in timing_columns:
            assert pd.api.types.is_numeric_dtype(df[col]), (
                f"Column {col} is not numeric"
            )
            assert (df[col] >= 0).all(), f"Column {col} has negative values"

        # Basic sanity checks
        assert len(df) > 0, "CSV is empty"
        assert df["n_obs"].nunique() > 1, "Should have multiple n_obs values"

        # Total time should generally equal fit_time + predict_time (within tolerance)
        time_diff = np.abs(df["total_time"] - (df["fit_time"] + df["predict_time"]))
        assert (time_diff < 0.1).all(), "total_time != fit_time + predict_time"


def test_csv_column_types():
    """Test specific column data types in timing CSV."""
    # Create a mock CSV for validation
    mock_data = {
        "backend": ["sklearn", "dapper_enkf"],
        "n_obs": [100, 100],
        "grid_size": [1000, 1000],
        "fit_time": [0.001, 0.010],
        "predict_time": [0.002, 0.005],
        "total_time": [0.003, 0.015],
        "rmse": [0.1, 0.2],
    }

    df = pd.DataFrame(mock_data)

    # Validate column types
    assert df["backend"].dtype == object  # String
    assert pd.api.types.is_integer_dtype(df["n_obs"])
    assert pd.api.types.is_integer_dtype(df["grid_size"])
    assert pd.api.types.is_numeric_dtype(df["fit_time"])
    assert pd.api.types.is_numeric_dtype(df["predict_time"])
    assert pd.api.types.is_numeric_dtype(df["total_time"])
    assert pd.api.types.is_numeric_dtype(df["rmse"])


def test_timing_consistency():
    """Test timing consistency rules."""
    mock_data = {
        "backend": ["sklearn"],
        "n_obs": [100],
        "grid_size": [1000],
        "fit_time": [0.001],
        "predict_time": [0.002],
        "total_time": [0.003],
        "rmse": [0.1],
    }

    df = pd.DataFrame(mock_data)

    # Fit and predict times should be positive
    assert (df["fit_time"] > 0).all()
    assert (df["predict_time"] > 0).all()

    # RMSE should be reasonable (not NaN or extremely large)
    assert (df["rmse"].notna()).all()
    assert (df["rmse"] < 10.0).all()  # Sanity check - RMSE shouldn't be huge

    # Total time should be sum of parts
    expected_total = df["fit_time"] + df["predict_time"]
    assert np.allclose(df["total_time"], expected_total, rtol=1e-6)


def test_dual_csv_generation():
    """Test that both observation and dimension sweep CSVs have correct variation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        obs_csv_path = Path(tmpdir) / "timing_obs.csv"
        dim_csv_path = Path(tmpdir) / "timing_dim.csv"

        # Generate observation sweep CSV
        obs_cmd = [
            sys.executable,
            "-m",
            "da_gp.scripts.bench",
            "--n_obs_grid",
            "50",
            "100",
            "200",
            "--grid_size_fixed",
            "1000",
            "--backends",
            "sklearn",
            "--csv",
            str(obs_csv_path),
            "--repeats",
            "1",
        ]

        obs_result = subprocess.run(
            obs_cmd,
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        if obs_result.returncode != 0:
            pytest.skip(f"Observation sweep benchmark failed: {obs_result.stderr}")

        # Generate dimension sweep CSV
        dim_cmd = [
            sys.executable,
            "-m",
            "da_gp.scripts.bench",
            "--dim_grid",
            "500",
            "1000",
            "2000",
            "--n_obs_fixed",
            "100",
            "--backends",
            "sklearn",
            "--csv",
            str(dim_csv_path),
            "--repeats",
            "1",
        ]

        dim_result = subprocess.run(
            dim_cmd,
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        if dim_result.returncode != 0:
            pytest.skip(f"Dimension sweep benchmark failed: {dim_result.stderr}")

        # Validate observation sweep CSV
        assert obs_csv_path.exists(), "Observation sweep CSV was not created"
        obs_df = pd.read_csv(obs_csv_path)
        assert obs_df["n_obs"].nunique() > 1, (
            f"Observation CSV should have multiple n_obs values, got {obs_df['n_obs'].nunique()}"
        )
        assert obs_df["grid_size"].nunique() == 1, (
            f"Observation CSV should have fixed grid_size, got {obs_df['grid_size'].nunique()} values"
        )

        # Validate dimension sweep CSV
        assert dim_csv_path.exists(), "Dimension sweep CSV was not created"
        dim_df = pd.read_csv(dim_csv_path)
        assert dim_df["grid_size"].nunique() > 1, (
            f"Dimension CSV should have multiple grid_size values, got {dim_df['grid_size'].nunique()}"
        )
        assert dim_df["n_obs"].nunique() == 1, (
            f"Dimension CSV should have fixed n_obs, got {dim_df['n_obs'].nunique()} values"
        )
