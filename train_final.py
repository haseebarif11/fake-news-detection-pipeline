import os
import joblib
import pandas as pd
import logging
from src.utils import setup_logging
from src.features.build_features import build_advanced_features_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

setup_logging(logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading ISOT dataset...")
    # Load processed ISOT data
    try:
        df = pd.read_csv('data/processed/isot_cleaned.csv')
    except FileNotFoundError:
        logger.error("Processed data not found. Please run the preprocessing notebook first.")
        return

    # To keep model size manageable for GitHub/Portfolio and fast for UI, we train on a balanced sample of 20k
    df_sample = df.sample(n=min(20000, len(df)), random_state=42).reset_index(drop=True)
    
    logger.info("Building features pipeline...")
    # Using 5000 features is a good balance between accuracy and model size
    pipeline = build_advanced_features_pipeline(max_features=5000, is_tweet=False)
    
    X = pipeline.fit_transform(df_sample['text_clean'])
    y = df_sample['label_binary'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logger.info("Training Final Random Forest Model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Quick eval
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logger.info(f"Final Model Validation Accuracy: {acc:.4f}")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    logger.info("Saving model and pipeline to models/ ...")
    joblib.dump(pipeline, 'models/feature_pipeline.joblib')
    joblib.dump(model, 'models/rf_model.joblib')
    
    logger.info("✅ Final model saved successfully!")

if __name__ == "__main__":
    main()
