#!/usr/bin/env python3
"""
Build automation for Matheron=EnKF paper using doit.

This file defines the build pipeline:
1. Generate timing data (observation and dimension sweeps)
2. Create timing plots from CSV data
3. Generate posterior comparison plot
4. Build final PDF with LaTeX

Usage:
    doit pdf           # Full pipeline: data → plots → PDF
    doit figures       # Just generate all figures
    doit timing_data   # Just generate timing CSVs
    doit clean         # Clean all generated files
    doit list          # Show all available tasks

Dependencies are managed by uv/pip and should be installed before running doit.
"""

import os
from pathlib import Path

from doit.tools import config_changed

# Configuration
DOIT_CONFIG = {
    "default_tasks": ["pdf"],
    "verbosity": 2,
    "reporter": "executed-only",  # Only show executed tasks
}

# Paths
FIGURES_DIR = Path("figures")
DATA_DIR = Path("data")
SCRIPTS_DIR = Path("da_gp/scripts")

# Timing parameters - matching the README workflow
OBS_SWEEP = [100, 500, 1000, 2000, 5000]
DIM_SWEEP = [250, 500, 1000, 2000, 4000]
BACKENDS = ["sklearn", "dapper_enkf", "dapper_letkf"]
REPEATS = 5

# Logging configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "WARNING")


def task_setup_dirs():
    """Create necessary output directories."""
    return {
        "actions": [
            f"mkdir -p {FIGURES_DIR}",
            f"mkdir -p {DATA_DIR}",
        ],
        "targets": [str(FIGURES_DIR / ".gitkeep"), str(DATA_DIR / ".gitkeep")],
        "uptodate": [False],  # Always run to ensure dirs exist
    }


def task_timing_obs_csv():
    """Generate observation scaling timing data."""
    csv_file = DATA_DIR / "timing_obs.csv"
    obs_args = " ".join(map(str, OBS_SWEEP))
    backends_args = " ".join(BACKENDS)
    params = dict(
        obs_sweep=OBS_SWEEP,
        backends=BACKENDS,
        repeats=REPEATS,
        grid_size_fixed=2000,
    )

    return {
        "actions": [
            f"uv run python {SCRIPTS_DIR}/bench.py "
            f"--n_obs_grid {obs_args} "
            f"--grid_size_fixed 2000 "
            f"--backends {backends_args} "
            f"--csv {csv_file} "
            f"--repeats {REPEATS} "
            f"--log-level {LOG_LEVEL}"
        ],
        "targets": [str(csv_file)],
        "file_dep": [
            str(SCRIPTS_DIR / "bench.py"),
            "da_gp/src/gp_common.py",
            "da_gp/src/gp_sklearn.py",
            "da_gp/src/gp_dapper.py",
        ],
        "verbosity": 2,
        "uptodate": [config_changed(params)],
        "clean": True,
    }


def task_timing_dim_csv():
    """Generate dimension scaling timing data."""
    csv_file = DATA_DIR / "timing_dim.csv"
    dim_args = " ".join(map(str, DIM_SWEEP))
    backends_args = " ".join(BACKENDS)
    params = dict(
        dim_sweep=DIM_SWEEP,
        backends=BACKENDS,
        repeats=REPEATS,
        n_obs_fixed=1000,
    )

    return {
        "actions": [
            f"uv run python {SCRIPTS_DIR}/bench.py "
            f"--dim_grid {dim_args} "
            f"--n_obs_fixed 1000 "
            f"--backends {backends_args} "
            f"--csv {csv_file} "
            f"--repeats {REPEATS} "
            f"--log-level {LOG_LEVEL}"
        ],
        "targets": [str(csv_file)],
        "file_dep": [
            str(SCRIPTS_DIR / "bench.py"),
            "da_gp/src/gp_common.py",
            "da_gp/src/gp_sklearn.py",
            "da_gp/src/gp_dapper.py",
        ],
        "verbosity": 2,
        "uptodate": [config_changed(params)],
        "clean": True,
    }


def task_timing_data():
    """Generate all timing data (meta-task)."""
    return {
        "actions": None,
        "task_dep": ["timing_obs_csv", "timing_dim_csv"],
    }


def task_timing_vs_observations():
    """Generate timing vs observations plot."""
    csv_file = DATA_DIR / "timing_obs.csv"
    pdf_file = FIGURES_DIR / "timing_vs_observations.pdf"

    return {
        "actions": [
            f"uv run python {SCRIPTS_DIR}/plot_timing.py {csv_file} --output-dir {FIGURES_DIR} --log-level {LOG_LEVEL}"
        ],
        "targets": [str(pdf_file)],
        "file_dep": [
            str(csv_file),
            str(SCRIPTS_DIR / "plot_timing.py"),
            "da_gp/figstyle.py",
        ],
        "task_dep": ["timing_obs_csv"],
        "clean": True,
    }


def task_timing_vs_dimensions():
    """Generate timing vs dimensions plot."""
    csv_file = DATA_DIR / "timing_dim.csv"
    pdf_file = FIGURES_DIR / "timing_vs_dimensions.pdf"

    return {
        "actions": [
            f"uv run python {SCRIPTS_DIR}/plot_timing.py {csv_file} --output-dir {FIGURES_DIR} --log-level {LOG_LEVEL}"
        ],
        "targets": [str(pdf_file)],
        "file_dep": [
            str(csv_file),
            str(SCRIPTS_DIR / "plot_timing.py"),
            "da_gp/figstyle.py",
        ],
        "task_dep": ["timing_dim_csv"],
        "clean": True,
    }


def task_posterior_samples():
    """Generate posterior comparison plot."""
    pdf_file = FIGURES_DIR / "posterior_samples.pdf"

    return {
        "actions": [
            f"uv run python {SCRIPTS_DIR}/plot_posterior.py --n_obs 50 --grid_size 1000 --log-level {LOG_LEVEL}"
        ],
        "targets": [str(pdf_file)],
        "file_dep": [
            str(SCRIPTS_DIR / "plot_posterior.py"),
            "da_gp/src/gp_common.py",
            "da_gp/src/gp_sklearn.py",
            "da_gp/src/gp_dapper.py",
            "da_gp/figstyle.py",
        ],
        "task_dep": ["setup_dirs"],
        "clean": True,
    }


def task_figures():
    """Generate all figures (meta-task)."""
    return {
        "actions": None,
        "task_dep": [
            "timing_vs_observations",
            "timing_vs_dimensions",
            "posterior_samples",
        ],
    }


def task_pdf():
    """Build main.pdf using latexmk."""
    return {
        "actions": ["latexmk -pdf main.tex"],
        "targets": ["main.pdf"],
        "file_dep": [
            "main.tex",
            "refs.bib",
            str(FIGURES_DIR / "timing_vs_observations.pdf"),
            str(FIGURES_DIR / "timing_vs_dimensions.pdf"),
            str(FIGURES_DIR / "posterior_samples.pdf"),
        ],
        "task_dep": ["figures"],
        "clean": ["latexmk -C main.tex"],
    }


def task_test():
    """Run the test suite."""
    return {
        "actions": ["uv run pytest da_gp/tests/ -v"],
        "verbosity": 2,
    }


# Clean tasks
def task_clean_data():
    """Clean generated CSV data files."""
    return {
        "actions": [f"rm -f {DATA_DIR}/*.csv"],
        "verbosity": 2,
    }


def task_clean_figures():
    """Clean generated figure files."""
    return {
        "actions": [f"rm -f {FIGURES_DIR}/*.pdf"],
        "verbosity": 2,
    }


def task_clean_latex():
    """Clean LaTeX auxiliary files."""
    return {
        "actions": ["latexmk -C main.tex"],
        "verbosity": 2,
    }


def task_clean_all():
    """Clean all generated files."""
    return {
        "actions": None,
        "task_dep": ["clean_data", "clean_figures", "clean_latex"],
    }
