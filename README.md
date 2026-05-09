# 🔍 Fake News & Tweet Misinformation Detector

A machine learning project that classifies **news articles** as real or fake and detects **misinformation in tweets**.

---

## 📋 Project Overview

| Task | Description | Datasets |
|------|-------------|----------|
| **News Classification** | Binary classification of news articles (real vs. fake) | LIAR, ISOT Fake News |
| **Tweet Misinformation** | Detect misinformation in tweets | COVID-19 Fake Tweets, Twitter15/16 |

---

## 🗂️ Project Structure

```
fake project/
├── data/
│   ├── raw/                  # Original downloaded datasets
│   ├── interim/              # Intermediate cleaning outputs
│   └── processed/            # Final cleaned datasets
├── notebooks/
│   ├── 01_eda_news.ipynb     # EDA for news datasets
│   ├── 02_eda_tweets.ipynb   # EDA for tweet datasets
│   └── 03_preprocessing.ipynb # Data cleaning walkthrough
├── src/
│   ├── data/
│   │   ├── load_data.py      # Unified data loading
│   │   └── preprocess.py     # Cleaning pipeline
│   ├── features/
│   │   └── build_features.py # Feature engineering (future)
│   ├── models/
│   │   └── train.py          # Model training (future)
│   └── utils.py              # Helpers
├── reports/figures/           # Saved EDA plots
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('punkt_tab')"
```

### 3. Download datasets

#### LIAR Dataset (auto-downloaded)
The LIAR dataset is loaded automatically via HuggingFace `datasets` library. No manual download needed.

#### ISOT Fake News Dataset
1. Go to: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Download and extract into `data/raw/isot/`
3. You should have:
   - `data/raw/isot/Fake.csv`
   - `data/raw/isot/True.csv`

#### COVID-19 Fake Tweets
1. Go to: https://www.kaggle.com/datasets/elvinagammed/covid19-fake-news-dataset-nlp
2. Download and extract into `data/raw/covid_tweets/`
3. You should have:
   - `data/raw/covid_tweets/Constraint_Train.csv`
   - `data/raw/covid_tweets/Constraint_Val.csv`
   - `data/raw/covid_tweets/Constraint_Test.csv`

#### Twitter15/16 Rumor Dataset
1. Go to: https://github.com/majingCUHK/Rumor_RvNN (or similar preprocessed repos)
2. Download and place into `data/raw/twitter15_16/`
3. You should have:
   - `data/raw/twitter15_16/twitter15_label.txt`
   - `data/raw/twitter15_16/twitter16_label.txt`
   - `data/raw/twitter15_16/twitter15_source_tweets.txt`
   - `data/raw/twitter15_16/twitter16_source_tweets.txt`

### 4. Run notebooks
```bash
jupyter notebook
```
Open notebooks in order: `01_eda_news.ipynb` → `02_eda_tweets.ipynb` → `03_preprocessing.ipynb`

---

## 📊 Datasets Summary

| Dataset | Type | Labels | Size |
|---------|------|--------|------|
| LIAR | News statements | 6-class → binarized | ~12.8K |
| ISOT | News articles | fake / real | ~44K |
| COVID-19 Tweets | Tweets | fake / real | ~10K |
| Twitter15/16 | Tweets | 4-class → binarized | ~1.5K |

---

## 📊 Final Model Comparison

Our pipeline was benchmarked across traditional Machine Learning baselines and state-of-the-art Deep Learning models.

| Model | Dataset | Accuracy | F1-Score | Note |
| :--- | :--- | :--- | :--- | :--- |
| **BERT** | News (ISOT) | **100%** | **1.00** | Exceptional on long-form text |
| **RoBERTa** | Tweets (COVID) | **85.5%** | **0.82** | Native Slang/Emoji support |
| **Random Forest** | News (ISOT) | 99.4% | 0.99 | Best balance for production |
| **Random Forest** | Tweets (COVID) | 91.4% | 0.91 | TF-IDF + Stylometrics |
| **Logistic Regression**| News (ISOT) | 98.1% | 0.98 | Reliable baseline |

### 🔍 Interpretability with LIME
We don't just predict; we explain. Using **LIME (Local Interpretable Model-agnostic Explanations)**, we can see exactly which words trigger a "Fake" classification. Sensationalist words like "BREAKING" or "SECRET" often carry the most weight in misinformation.

---

## 🚀 How to Run

### 1. Web Application (Streamlit)
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 2. Training Pipelines
All phases are documented in Jupyter Notebooks:
- `01_eda_news.ipynb`: Exploratory Data Analysis for News.
- `04_feature_engineering_and_modeling.ipynb`: Classical ML Baselines.
- `05_deep_learning_models.ipynb`: BERT & RoBERTa Fine-tuning.
- `06_interpretability_and_comparison.ipynb`: LIME Explanations.

---

## 🛠️ Tech Stack
- **Core:** Python, Pandas, Scikit-Learn
- **NLP:** Transformers (HuggingFace), NLTK, TextBlob, TextStat
- **Deep Learning:** PyTorch, BERT, RoBERTa
- **Explainability:** LIME
- **Deployment:** Streamlit

---
**Author:** Haseeb Arif
**Project:** Fake News & Misinformation Detection Pipeline
