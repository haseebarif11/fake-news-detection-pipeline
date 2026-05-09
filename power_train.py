import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
from src.data.preprocess import clean_text

print("Building High-Performance Power Model...")

# 1. Load full datasets (not just small samples)
df_fake = pd.read_csv('data/raw/isot/Fake.csv').sample(8000, random_state=42)
df_true = pd.read_csv('data/raw/isot/True.csv').sample(8000, random_state=42)
df_fake['label_binary'] = 1
df_true['label_binary'] = 0
df_isot = pd.concat([df_fake, df_true])

df_covid = pd.read_csv('data/processed/covid_tweets_cleaned.csv')

# Combine
df_master = pd.concat([
    df_isot[['text', 'label_binary']].rename(columns={'text': 'text_clean'}),
    df_covid[['text_clean', 'label_binary']]
])

# 2. Clean
print("Processing text...")
df_master['text_clean'] = df_master['text_clean'].apply(lambda x: clean_text(str(x), do_stopwords=True))

# 3. UNLIMITED DEPTH Training
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42))
])

print("Training high-capacity ensemble...")
pipeline.fit(df_master['text_clean'], df_master['label_binary'])

# 4. Save
joblib.dump(pipeline, 'models/random_forest_news_model.joblib')
print("Power Model deployed!")
