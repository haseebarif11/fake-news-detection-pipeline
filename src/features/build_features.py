"""
Feature Engineering Module
===========================

Transforms cleaned text into numerical feature representations.
Supports TF-IDF vectorization and dense NLP features like sentiment,
readability, and stylometrics.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from textblob import TextBlob
import textstat
import re
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dense Feature Extractors (Custom Transformers)
# ---------------------------------------------------------------------------

class DenseFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts dense numeric features from text:
    - Sentiment (polarity, subjectivity)
    - Readability (Flesch-Kincaid)
    - Stylometrics (all-caps ratio, punctuation ratio)
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Ensure X is a list or pandas Series of strings
        if isinstance(X, pd.Series):
            X = X.tolist()
            
        features = []
        for text in X:
            text_str = str(text)
            if not text_str.strip():
                # Fallback for empty strings
                features.append([0.0, 0.0, 0.0, 0.0, 0.0])
                continue

            # 1. Sentiment
            blob = TextBlob(text_str)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # 2. Readability (Flesch-Kincaid Grade)
            try:
                readability = textstat.flesch_kincaid_grade(text_str)
            except Exception:
                readability = 0.0

            # 3. Stylometrics
            words = text_str.split()
            word_count = len(words)
            if word_count > 0:
                caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / word_count
                punct_count = len(re.findall(r'[!?.,;:]', text_str))
                punct_ratio = punct_count / word_count
            else:
                caps_ratio = 0.0
                punct_ratio = 0.0

            features.append([
                polarity,
                subjectivity,
                readability,
                caps_ratio,
                punct_ratio
            ])
            
        return np.array(features)

class TweetFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts tweet-specific features:
    - Hashtag count
    - URL presence (binary)
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.Series):
            X = X.tolist()
            
        features = []
        for text in X:
            text_str = str(text)
            
            # Hashtags
            hashtag_count = len(re.findall(r'#\w+', text_str))
            
            # URL presence (using a simple http check)
            has_url = 1.0 if 'http' in text_str.lower() else 0.0
            
            features.append([hashtag_count, has_url])
            
        return np.array(features)

# ---------------------------------------------------------------------------
# Feature Pipelines
# ---------------------------------------------------------------------------

def build_advanced_features_pipeline(
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    is_tweet: bool = False
) -> FeatureUnion:
    """
    Build a FeatureUnion pipeline combining TF-IDF and dense features.

    Parameters
    ----------
    max_features : int
        Max features for TF-IDF.
    ngram_range : tuple
        N-gram range for TF-IDF.
    is_tweet : bool
        If True, includes tweet-specific features (hashtags, URLs).

    Returns
    -------
    FeatureUnion
        A scikit-learn FeatureUnion object combining all features.
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        max_df=0.95,
        min_df=2,
        strip_accents="unicode",
        sublinear_tf=True,
    )
    
    dense_features = DenseFeatureExtractor()
    
    transformers = [
        ('tfidf', tfidf),
        ('dense', dense_features)
    ]
    
    if is_tweet:
        transformers.append(('tweet', TweetFeatureExtractor()))
        
    return FeatureUnion(transformers)


def build_tfidf_features(
    train_texts: pd.Series,
    test_texts: Optional[pd.Series] = None,
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    max_df: float = 0.95,
    min_df: int = 2,
) -> tuple:
    """
    Build TF-IDF feature matrices from text data (Legacy/Basic).

    Parameters
    ----------
    train_texts : pd.Series
        Training text data.
    test_texts : pd.Series, optional
        Test text data to transform (not fit).

    Returns
    -------
    tuple
        (X_train, X_test, vectorizer)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        strip_accents="unicode",
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(train_texts)

    X_test = None
    if test_texts is not None:
        X_test = vectorizer.transform(test_texts)

    return X_train, X_test, vectorizer
