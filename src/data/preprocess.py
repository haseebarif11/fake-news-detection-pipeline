"""
Text Preprocessing Pipeline
============================

Configurable text cleaning for news articles and tweets.

Usage:
    >>> from src.data.preprocess import clean_text, preprocess_pipeline
    >>> clean_text("<p>Check http://example.com @user #trending!</p>")
    'check trending'
"""

import re
import logging
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_STOPWORDS: Optional[set] = None
_LEMMATIZER = None


def _get_stopwords() -> set:
    global _STOPWORDS
    if _STOPWORDS is None:
        import nltk
        try:
            from nltk.corpus import stopwords
            _STOPWORDS = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


def _get_lemmatizer():
    global _LEMMATIZER
    if _LEMMATIZER is None:
        import nltk
        try:
            from nltk.stem import WordNetLemmatizer
            _LEMMATIZER = WordNetLemmatizer()
            _LEMMATIZER.lemmatize("test")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            from nltk.stem import WordNetLemmatizer
            _LEMMATIZER = WordNetLemmatizer()
    return _LEMMATIZER


# --- Individual Cleaning Steps ---

def strip_html(text: str) -> str:
    """Remove HTML tags using BeautifulSoup."""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


def normalize_urls(text: str, replacement: str = "") -> str:
    """Replace URLs with a token or remove them."""
    pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|"
        r"[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        re.IGNORECASE,
    )
    return pattern.sub(replacement, text)


def remove_mentions(text: str, replacement: str = "") -> str:
    """Remove Twitter @mentions."""
    return re.sub(r"@\w+", replacement, text)


def normalize_hashtags(text: str) -> str:
    """Remove '#' symbol but keep the hashtag word."""
    return re.sub(r"#(\w+)", r"\1", text)


def remove_source_tags(text: str) -> str:
    """Remove common news agency source tags like 'WASHINGTON (Reuters) - '."""
    # Pattern for "LOCATION (Agency) - " or "AGENCY - "
    text = re.sub(r'^[A-Z, ]+\s+\([A-Z, ]+\)\s+-\s+', '', text)
    text = re.sub(r'^[A-Z, ]+-[A-Z, ]+\s+-\s+', '', text)
    text = re.sub(r'\(Reuters\)', '', text, flags=re.IGNORECASE)
    return text


def remove_special_chars(text: str) -> str:
    """Remove non-alphanumeric characters (except spaces)."""
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text)


def collapse_whitespace(text: str) -> str:
    """Replace multiple whitespace with a single space."""
    return re.sub(r"\s+", " ", text).strip()


def remove_stopwords(text: str) -> str:
    """Remove English stopwords."""
    stops = _get_stopwords()
    return " ".join(w for w in text.split() if w not in stops)


def lemmatize_text(text: str) -> str:
    """Apply WordNet lemmatization."""
    lem = _get_lemmatizer()
    return " ".join(lem.lemmatize(w) for w in text.split())


# --- Master Cleaning Function ---

def clean_text(
    text: str,
    do_html: bool = True,
    do_urls: bool = True,
    do_mentions: bool = True,
    do_hashtags: bool = True,
    do_special_chars: bool = True,
    do_lowercase: bool = True,
    do_stopwords: bool = False,
    do_lemmatize: bool = False,
) -> str:
    """
    Apply a configurable text cleaning pipeline.

    Stopword removal and lemmatization are off by default (task-dependent).

    Examples
    --------
    >>> clean_text("<p>BREAKING: Check http://t.co/abc @CNN #FakeNews!!!</p>")
    'breaking check fakenews'
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    if do_html:
        text = strip_html(text)
        text = remove_source_tags(text)
    if do_urls:
        text = normalize_urls(text)
    if do_mentions:
        text = remove_mentions(text)
    if do_hashtags:
        text = normalize_hashtags(text)
    if do_special_chars:
        text = remove_special_chars(text)
    if do_lowercase:
        text = text.lower()
    if do_stopwords:
        text = remove_stopwords(text)
    if do_lemmatize:
        text = lemmatize_text(text)

    text = collapse_whitespace(text)
    return text


# --- DataFrame-level Pipeline ---

def remove_duplicates(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Remove rows with duplicate text content."""
    n_before = len(df)
    df = df.drop_duplicates(subset=[text_col], keep="first")
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info(f"Removed {n_removed} duplicate rows ({n_removed/n_before:.1%})")
    return df.reset_index(drop=True)


def handle_missing(
    df: pd.DataFrame, text_col: str = "text", label_col: str = "label_binary"
) -> pd.DataFrame:
    """Drop rows with missing or empty text/label values."""
    n_before = len(df)
    df = df.dropna(subset=[text_col])
    df = df[df[text_col].astype(str).str.strip().str.len() > 0]
    if label_col in df.columns:
        df = df.dropna(subset=[label_col])
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info(f"Removed {n_removed} missing-value rows ({n_removed/n_before:.1%})")
    return df.reset_index(drop=True)


def preprocess_pipeline(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label_binary",
    clean_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline: missing values → duplicates → clean text.

    Creates a new ``text_clean`` column with the cleaned text.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    text_col : str
        Column containing raw text.
    label_col : str
        Column containing labels.
    clean_kwargs : dict, optional
        Keyword arguments forwarded to ``clean_text()``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with ``text_clean`` column.
    """
    if clean_kwargs is None:
        clean_kwargs = {}

    logger.info(f"Starting preprocessing ({len(df)} rows)...")

    df = handle_missing(df, text_col=text_col, label_col=label_col)
    df = remove_duplicates(df, text_col=text_col)

    from tqdm import tqdm
    tqdm.pandas(desc="Cleaning text")
    df["text_clean"] = df[text_col].progress_apply(
        lambda x: clean_text(str(x), **clean_kwargs)
    )

    df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)
    logger.info(f"Preprocessing complete: {len(df)} rows remaining")
    return df
