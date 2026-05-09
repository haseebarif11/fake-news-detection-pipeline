import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
from src.data.preprocess import clean_text
import os

print("Building Master Multi-Topic Model...")

# 1. Load ISOT (Politics)
df_fake = pd.read_csv('data/raw/isot/Fake.csv').sample(3000, random_state=42)
df_true = pd.read_csv('data/raw/isot/True.csv').sample(3000, random_state=42)
df_fake['label_binary'] = 1
df_true['label_binary'] = 0

# 2. Load COVID Tweets (Health/Science)
df_covid = pd.read_csv('data/processed/covid_tweets_cleaned.csv').sample(3000, random_state=42)
# Ensure covid labels are mapped to our 0=Real, 1=Fake
# (Assuming covid_tweets_cleaned already has label_binary)

# Combine
df_master = pd.concat([df_fake, df_true, df_covid[['text_clean', 'label_binary']]])

# 3. Clean
print("Re-cleaning master dataset...")
df_master['text_clean'] = df_master.apply(
    lambda row: clean_text(str(row['text'])) if 'text' in row else str(row['text_clean']), axis=1
)

# 4. Train High-Capacity Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=8000, ngram_range=(1,2))),
    ('rf', RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42))
])

print("Training...")
pipeline.fit(df_master['text_clean'], df_master['label_binary'])

# 5. Save
joblib.dump(pipeline, 'models/random_forest_news_model.joblib')
print("Master Model Ready!")
