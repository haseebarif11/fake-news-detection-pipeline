import streamlit as st
from src.models.predict import FakeNewsDetector
import textstat
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Fake News & Misinformation Detector",
    page_icon="🕵️",
    layout="centered"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .prediction-box-fake {
        background-color: #ffcccc;
        border-left: 5px solid #ff0000;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .prediction-box-real {
        background-color: #ccffcc;
        border-left: 5px solid #00cc00;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the model only once and cache it for performance."""
    return FakeNewsDetector()

def analyze_linguistics(text):
    """Extract linguistic metrics for display."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    try:
        readability = textstat.flesch_kincaid_grade(text)
    except Exception:
        readability = 0.0
        
    return polarity, subjectivity, readability

def main():
    st.title("🕵️ Misinformation Detector")
    st.markdown("""
    Welcome to the Fake News & Misinformation Detection engine. 
    Paste any news article or tweet below, and our Machine Learning model (Random Forest + TF-IDF + Linguistic Features) will analyze it.
    """)
    
    try:
        detector = load_model()
    except Exception as e:
        st.error(f"Error loading model. Did you run Phase 3 training? {e}")
        return

    # Text Input
    user_input = st.text_area("Paste an article or tweet here:", height=200, placeholder="Breaking News: ...")
    
    if st.button("🔍 Analyze Text"):
        if not user_input.strip():
            st.warning("Please enter some text to analyze.")
            return
            
        with st.spinner("Analyzing text and extracting NLP features..."):
            # 1. Get Prediction
            result = detector.predict(user_input)
            
            # 2. Get Linguistic Stats
            polarity, subjectivity, readability = analyze_linguistics(user_input)
            
            # --- Display Results ---
            st.markdown("### Results")
            
            # Prediction Box
            if result['prediction'] == "FAKE":
                st.markdown(
                    f'<div class="prediction-box-fake">'
                    f'<h2>🚨 Predicted: FAKE NEWS</h2>'
                    f'<p><strong>Confidence:</strong> {result["confidence"]*100:.1f}%</p>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="prediction-box-real">'
                    f'<h2>✅ Predicted: REAL NEWS</h2>'
                    f'<p><strong>Confidence:</strong> {result["confidence"]*100:.1f}%</p>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
                
            st.divider()
            
            # Linguistic Insights
            st.markdown("### Linguistic Insights")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sentiment_label = "Neutral"
                if polarity > 0.1: sentiment_label = "Positive"
                elif polarity < -0.1: sentiment_label = "Negative"
                st.metric("Sentiment Polarity", f"{polarity:.2f}", sentiment_label)
                
            with col2:
                st.metric("Subjectivity", f"{subjectivity:.2f}", "0=Objective, 1=Opinionated")
                
            with col3:
                st.metric("Readability Grade", f"{readability:.1f}", "Flesch-Kincaid Level")
                
            # Explainability
            st.info("""
            **How does this work?**
            The model translates your text into a numerical matrix (TF-IDF) identifying the most important vocabulary. 
            It also calculates the dense NLP features you see above. 
            A Random Forest ensemble algorithm then compares these features against tens of thousands of known fake and real articles to make a determination.
            """)

if __name__ == "__main__":
    main()
