import joblib
import pandas as pd
import numpy as np
from lime.lime_text import LimeTextExplainer
from src.data.preprocess import clean_text
import os

class FakeNewsDetector:
    def __init__(self, model_path='models/random_forest_news_model.joblib'):
        """Initialize with a trained Scikit-Learn Pipeline."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        
        self.pipeline = joblib.load(model_path)
        self.explainer = LimeTextExplainer(class_names=['Real', 'Fake'])

    def predict(self, text):
        """Clean text and predict probability/class."""
        # Clean text using our project's standard preprocessor
        cleaned_text = clean_text(text, do_stopwords=True, do_lemmatize=True)
        
        # Predict
        prob = self.pipeline.predict_proba([cleaned_text])[0]
        prediction = self.pipeline.predict([cleaned_text])[0]
        
        return {
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': prob[prediction],
            'probabilities': {
                'REAL': prob[0],
                'FAKE': prob[1]
            }
        }

    def explain(self, text, num_features=10):
        """Generate LIME explanation for a single prediction."""
        # Use the pipeline's predict_proba directly for LIME
        # Note: LIME needs raw text because the pipeline handles vectorization
        exp = self.explainer.explain_instance(
            text, 
            self.pipeline.predict_proba, 
            num_features=num_features
        )
        return exp
