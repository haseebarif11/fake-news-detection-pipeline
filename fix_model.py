import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
from src.data.preprocess import clean_text

# Load small sample of ISOT data
df_fake = pd.read_csv('data/raw/isot/Fake.csv').sample(3000, random_state=42)
df_true = pd.read_csv('data/raw/isot/True.csv').sample(3000, random_state=42)
df_fake['label_binary'] = 1
df_true['label_binary'] = 0
df = pd.concat([df_fake, df_true])

# Re-clean using new logic
df['text_clean'] = df['text'].apply(lambda x: clean_text(str(x), do_stopwords=True, do_lemmatize=True))

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(df['text_clean'], df['label_binary'])
joblib.dump(pipeline, 'models/random_forest_news_model.joblib')
