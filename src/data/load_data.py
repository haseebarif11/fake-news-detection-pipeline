"""
Unified Data Loading Module
============================

Provides consistent interfaces for loading all project datasets into
pandas DataFrames with standardized column names:
    - `text`:  The statement, article body, or tweet text
    - `label`: The ground-truth label (original)
    - `label_binary`: Binarized label — 0 = REAL, 1 = FAKE
    - `source`: Which dataset the row came from

Usage:
    >>> from src.data.load_data import load_liar
    >>> df = load_liar()
    >>> df.head()
"""

import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Project root is two levels up from this file (src/data/load_data.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Label mapping constants
LIAR_FAKE_LABELS = {"pants-fire", "false", "barely-true"}
LIAR_REAL_LABELS = {"half-true", "mostly-true", "true"}

TWITTER_FAKE_LABELS = {"true", "false"}  # true rumor & false rumor = misinfo
TWITTER_REAL_LABELS = {"non-rumor"}


# ---------------------------------------------------------------------------
# LIAR Dataset
# ---------------------------------------------------------------------------

def load_liar(data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load the LIAR dataset from original TSV files.
    If not present, downloads the dataset from the original source.

    The LIAR dataset contains ~12.8K political statements rated on a 6-point
    scale by PolitiFact. Labels are binarized into REAL (0) and FAKE (1).

    Parameters
    ----------
    data_dir : str, optional
        Directory containing the LIAR TSV files. Defaults to
        ``data/raw/liar/``.

    Returns
    -------
    pd.DataFrame
        Columns: text, label, label_binary, speaker, party, subject,
        context, source
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR / "liar"
    else:
        data_dir = Path(data_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if files exist, if not download
    required_files = ["train.tsv", "valid.tsv", "test.tsv"]
    missing = [f for f in required_files if not (data_dir / f).exists()]
    
    if missing:
        logger.info("Downloading LIAR dataset from original source...")
        import requests
        import zipfile
        import io
        
        url = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
        try:
            r = requests.get(url, allow_redirects=True, timeout=15)
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(data_dir)
            logger.info("LIAR dataset downloaded and extracted.")
        except Exception as e:
            raise RuntimeError(f"Failed to download LIAR dataset: {e}")

    logger.info("Loading LIAR dataset from local TSV files...")

    col_names = [
        "id", "label", "text", "subject", "speaker", "job_title", 
        "state_info", "party", "barely_true", "false_counts", 
        "half_true", "mostly_true", "pants_on_fire", "context"
    ]

    frames = []
    file_map = {
        "train": "train.tsv",
        "validation": "valid.tsv",
        "test": "test.tsv",
    }

    for split_name, filename in file_map.items():
        filepath = data_dir / filename
        if filepath.exists():
            df_split = pd.read_csv(filepath, sep="\t", header=None, names=col_names, quoting=3) # quoting=3 is QUOTE_NONE
            df_split["split"] = split_name
            frames.append(df_split)

    if not frames:
        raise FileNotFoundError(f"LIAR dataset not found at {data_dir}")

    df = pd.concat(frames, ignore_index=True)

    # Normalize labels
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    # Binarize: fake=1, real=0
    df["label_binary"] = df["label"].apply(
        lambda x: 1 if x in LIAR_FAKE_LABELS else 0
    )

    df["source"] = "liar"

    keep_cols = [
        "text", "label", "label_binary",
        "speaker", "party", "subject", "context", "source", "split",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    logger.info(f"LIAR dataset loaded: {len(df)} rows")
    return df[keep_cols].copy()



# ---------------------------------------------------------------------------
# ISOT Fake News Dataset
# ---------------------------------------------------------------------------

def load_isot(data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load the ISOT Fake News dataset from local CSV files.

    The ISOT dataset contains ~44K news articles (21K fake, 23K real)
    collected from Reuters (real) and various unreliable sources (fake).

    Parameters
    ----------
    data_dir : str, optional
        Path to the directory containing ``Fake.csv`` and ``True.csv``.
        Defaults to ``data/raw/isot/``.

    Returns
    -------
    pd.DataFrame
        Columns: text, title, label, label_binary, subject, date, source

    Raises
    ------
    FileNotFoundError
        If the CSV files are not found in the specified directory.

    Examples
    --------
    >>> df = load_isot()
    >>> df["label_binary"].value_counts()
    1    23481
    0    21417
    Name: label_binary, dtype: int64
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR / "isot"
    else:
        data_dir = Path(data_dir)

    fake_path = data_dir / "Fake.csv"
    true_path = data_dir / "True.csv"

    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError(
            f"ISOT dataset not found at {data_dir}.\n"
            f"Download from: https://www.kaggle.com/datasets/"
            f"clmentbisaillon/fake-and-real-news-dataset\n"
            f"Place Fake.csv and True.csv in: {data_dir}"
        )

    logger.info("Loading ISOT Fake News dataset...")

    df_fake = pd.read_csv(fake_path)
    df_fake["label"] = "fake"
    df_fake["label_binary"] = 1

    df_true = pd.read_csv(true_path)
    df_true["label"] = "real"
    df_true["label_binary"] = 0

    df = pd.concat([df_fake, df_true], ignore_index=True)
    df["source"] = "isot"

    # Standardize: combine title + text for full article content
    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["text"] = df["text"].str.strip()

    keep_cols = ["text", "title", "label", "label_binary", "subject", "date", "source"]
    keep_cols = [c for c in keep_cols if c in df.columns]

    logger.info(f"ISOT dataset loaded: {len(df)} rows")
    return df[keep_cols].copy()


# ---------------------------------------------------------------------------
# COVID-19 Fake Tweets Dataset
# ---------------------------------------------------------------------------

def load_covid_tweets(data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load the COVID-19 Fake News / Constraint dataset from local CSVs.

    Contains ~10K tweets labeled as ``real`` or ``fake`` related to
    COVID-19 misinformation.

    Parameters
    ----------
    data_dir : str, optional
        Path to directory containing ``Constraint_Train.csv``,
        ``Constraint_Val.csv``, ``Constraint_Test.csv``.
        Defaults to ``data/raw/covid_tweets/``.

    Returns
    -------
    pd.DataFrame
        Columns: text, label, label_binary, source, split

    Raises
    ------
    FileNotFoundError
        If no CSV files are found in the specified directory.

    Examples
    --------
    >>> df = load_covid_tweets()
    >>> df["label"].value_counts()
    real    5600
    fake    5100
    Name: label, dtype: int64
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR / "covid_tweets"
    else:
        data_dir = Path(data_dir)

    file_map = {
        "train": "Constraint_Train.csv",
        "validation": "Constraint_Val.csv",
        "test": "Constraint_Test.csv",
    }

    frames = []
    found_any = False

    for split_name, filename in file_map.items():
        filepath = data_dir / filename
        if filepath.exists():
            df_split = pd.read_csv(filepath)
            df_split["split"] = split_name
            frames.append(df_split)
            found_any = True
            logger.info(f"  Loaded {filename}: {len(df_split)} rows")

    if not found_any:
        # Try loading any CSV in the directory
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            for f in csv_files:
                df_split = pd.read_csv(f)
                df_split["split"] = f.stem
                frames.append(df_split)
            found_any = True

    if not found_any:
        raise FileNotFoundError(
            f"COVID-19 tweets dataset not found at {data_dir}.\n"
            f"Download from: https://www.kaggle.com/datasets/"
            f"elvinagammed/covid19-fake-news-dataset-nlp\n"
            f"Place CSV files in: {data_dir}"
        )

    df = pd.concat(frames, ignore_index=True)

    # Standardize column names (datasets vary)
    text_col = _find_column(df, ["tweet", "text", "headline", "headlines"])
    label_col = _find_column(df, ["label", "outcome", "class"])

    if text_col:
        df = df.rename(columns={text_col: "text"})
    if label_col and label_col != "label":
        df = df.rename(columns={label_col: "label"})

    # Normalize labels
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    # Binarize
    df["label_binary"] = df["label"].apply(
        lambda x: 1 if x in ("fake", "0", "false") else 0
    )

    df["source"] = "covid_tweets"

    keep_cols = ["text", "label", "label_binary", "source", "split"]
    keep_cols = [c for c in keep_cols if c in df.columns]

    logger.info(f"COVID-19 tweets loaded: {len(df)} rows")
    return df[keep_cols].copy()


# ---------------------------------------------------------------------------
# Twitter15 / Twitter16 Rumor Dataset
# ---------------------------------------------------------------------------

def load_twitter15_16(data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load Twitter15/16 rumor detection datasets from Rumor_RvNN format.

    Reads ``Twitter15_label_All.txt`` and ``Twitter16_label_All.txt``.
    These are tab-separated files with columns:
        label, event_id, tweet_id, num_replies, max_depth,
        max_breadth, tree_size, source_type, credibility_score

    Labels are binarized:
        - FAKE (1): true rumor, false rumor
        - REAL (0): non-rumor
        - Unverified rumor rows are kept with ``label_binary = -1``

    Parameters
    ----------
    data_dir : str, optional
        Path to directory containing the label files.
        Defaults to ``data/raw/twitter15_16/``.

    Returns
    -------
    pd.DataFrame
        Columns: text, label, label_binary, event_id, tweet_id,
        dataset_version, source

    Raises
    ------
    FileNotFoundError
        If label files are not found.
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR / "twitter15_16"
    else:
        data_dir = Path(data_dir)

    frames = []

    # Column names for the tab-separated files
    col_names = [
        "label", "event_id", "tweet_id",
        "num_replies", "max_depth", "max_breadth",
        "tree_size", "source_type", "credibility_score",
    ]

    # Try both naming conventions
    file_patterns = [
        ("twitter15", "Twitter15_label_All.txt"),
        ("twitter16", "Twitter16_label_All.txt"),
        ("twitter15", "twitter15_label.txt"),
        ("twitter16", "twitter16_label.txt"),
    ]

    seen_versions = set()
    for version, filename in file_patterns:
        if version in seen_versions:
            continue
        label_file = data_dir / filename
        if not label_file.exists():
            continue

        seen_versions.add(version)

        try:
            df_version = pd.read_csv(
                label_file, sep="\t", header=None,
                names=col_names, encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Error reading {filename}: {e}")
            continue

        df_version["label"] = df_version["label"].astype(str).str.lower().str.strip()
        df_version["dataset_version"] = version

        # Build a descriptive text from event_id (this dataset has
        # metadata but not full tweet text — the event_id is the
        # topic/event descriptor e.g. "germanwings", "ottawashooting")
        df_version["text"] = (
            df_version["event_id"].astype(str) + " "
            + df_version["source_type"].astype(str)
        )

        frames.append(df_version)
        logger.info(f"  Loaded {version} ({filename}): {len(df_version)} rows")

    if not frames:
        raise FileNotFoundError(
            f"Twitter15/16 dataset not found at {data_dir}.\n"
            f"Expected files: Twitter15_label_All.txt, Twitter16_label_All.txt\n"
            f"Download from: https://github.com/majingCUHK/Rumor_RvNN\n"
            f"Place label files in: {data_dir}"
        )

    df = pd.concat(frames, ignore_index=True)

    # Binarize labels
    df["label_binary"] = df["label"].map({
        "true": 1,        # true rumor -> misinformation
        "false": 1,       # false rumor -> misinformation
        "non-rumor": 0,   # legitimate
        "unverified": -1, # uncertain -- kept for analysis
    })

    df["source"] = "twitter15_16"

    keep_cols = [
        "text", "label", "label_binary", "event_id", "tweet_id",
        "num_replies", "max_depth", "max_breadth", "tree_size",
        "source_type", "credibility_score",
        "dataset_version", "source",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    logger.info(f"Twitter15/16 loaded: {len(df)} rows")
    return df[keep_cols].copy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Find the first matching column name from a list of candidates."""
    df_cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in df_cols_lower:
            return df_cols_lower[candidate.lower()]
    return None


def load_all_news() -> pd.DataFrame:
    """
    Load and combine all news datasets (LIAR + ISOT).

    Returns a unified DataFrame with standardized columns.
    Each row includes a ``source`` column indicating the origin dataset.

    Returns
    -------
    pd.DataFrame
        Combined news dataset.
    """
    frames = []

    try:
        frames.append(load_liar())
        logger.info("✓ LIAR loaded")
    except Exception as e:
        logger.warning(f"✗ LIAR failed: {e}")

    try:
        frames.append(load_isot())
        logger.info("✓ ISOT loaded")
    except Exception as e:
        logger.warning(f"✗ ISOT failed: {e}")

    if not frames:
        raise RuntimeError("No news datasets could be loaded.")

    # Keep only common columns
    common_cols = ["text", "label", "label_binary", "source"]
    df = pd.concat(
        [f[common_cols] for f in frames],
        ignore_index=True,
    )

    logger.info(f"Combined news dataset: {len(df)} rows")
    return df


def load_all_tweets() -> pd.DataFrame:
    """
    Load and combine all tweet datasets (COVID-19 + Twitter15/16).

    Returns a unified DataFrame with standardized columns.

    Returns
    -------
    pd.DataFrame
        Combined tweet dataset.
    """
    frames = []

    try:
        frames.append(load_covid_tweets())
        logger.info("✓ COVID-19 tweets loaded")
    except Exception as e:
        logger.warning(f"✗ COVID-19 tweets failed: {e}")

    try:
        df_tw = load_twitter15_16()
        # Exclude unverified (-1) for combined dataset
        df_tw = df_tw[df_tw["label_binary"] != -1].copy()
        frames.append(df_tw)
        logger.info("✓ Twitter15/16 loaded")
    except Exception as e:
        logger.warning(f"✗ Twitter15/16 failed: {e}")

    if not frames:
        raise RuntimeError("No tweet datasets could be loaded.")

    common_cols = ["text", "label", "label_binary", "source"]
    df = pd.concat(
        [f[common_cols] for f in frames],
        ignore_index=True,
    )

    logger.info(f"Combined tweet dataset: {len(df)} rows")
    return df
