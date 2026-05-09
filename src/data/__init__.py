"""
Data loading and preprocessing module.

Provides unified interfaces for loading multiple fake news and
misinformation datasets, along with a robust text cleaning pipeline.
"""

from .load_data import load_liar, load_isot, load_covid_tweets, load_twitter15_16
from .preprocess import clean_text, preprocess_pipeline

__all__ = [
    "load_liar",
    "load_isot",
    "load_covid_tweets",
    "load_twitter15_16",
    "clean_text",
    "preprocess_pipeline",
]
