"""
Shared Utilities
================

Helper functions for logging, plotting, and file I/O used across
the project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure project-wide logging with a clean format.

    Parameters
    ----------
    level : int
        Logging level. Default ``logging.INFO``.

    Returns
    -------
    logging.Logger
        Root logger instance.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    return root_logger


# ---------------------------------------------------------------------------
# Plotting Helpers
# ---------------------------------------------------------------------------

# Consistent style for all plots
PLOT_STYLE = {
    "figure.figsize": (12, 6),
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Color palette for real vs fake
LABEL_COLORS = {"Real": "#2ecc71", "Fake": "#e74c3c"}
PALETTE_BINARY = ["#2ecc71", "#e74c3c"]  # [real, fake]


def apply_plot_style():
    """Apply the project's consistent matplotlib style."""
    plt.rcParams.update(PLOT_STYLE)
    sns.set_style("whitegrid")


def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: Optional[str] = None,
    dpi: int = 150,
):
    """
    Save a matplotlib figure to the reports/figures directory.

    Parameters
    ----------
    fig : plt.Figure
        The figure to save.
    filename : str
        Output filename (e.g., ``"label_distribution.png"``).
    output_dir : str, optional
        Output directory. Defaults to ``reports/figures/``.
    dpi : int
        Resolution. Default 150.
    """
    if output_dir is None:
        project_root = Path(__file__).resolve().parents[1]
        output_dir = project_root / "reports" / "figures"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Figure saved: {filepath}")


# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parents[1]


def get_data_dir(stage: str = "raw") -> Path:
    """
    Return path to a data subdirectory.

    Parameters
    ----------
    stage : str
        One of ``"raw"``, ``"interim"``, ``"processed"``.

    Returns
    -------
    Path
        Absolute path to the data directory.
    """
    data_dir = get_project_root() / "data" / stage
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
