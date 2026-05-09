import nbformat
import os

nb_path = 'notebooks/06_interpretability_and_comparison.ipynb'
if os.path.exists(nb_path):
    nb = nbformat.read(nb_path, as_version=4)
    
    # Restore Cell 5
    nb.cells[5].source = '''# Initialize LIME explainer
class_names = ['Real News', 'Fake News']
explainer = LimeTextExplainer(class_names=class_names)

# Wrapper function required by LIME
def predict_proba_wrapper(texts):
    # Clean texts first
    from src.data.preprocess import clean_text
    cleaned = [clean_text(t, do_stopwords=True, do_lemmatize=True) for t in texts]
    X = detector.pipeline.transform(pd.Series(cleaned))
    return detector.model.predict_proba(X)

# Let's take a sample "Fake" text and a "Real" text
sample_fake = """
BREAKING: The government is hiding massive UFO documents! 
A whistleblower just revealed that the President is secretly an alien from Mars. 
Share this before they delete it! #WakeUp
"""

sample_real = """
WASHINGTON — The Federal Reserve announced a quarter-point interest rate hike on Wednesday, 
aiming to curb inflation. The stock market responded positively, with the S&P 500 rising 1.2%.
"""

print("Generating explanation for Fake News Sample...")
exp_fake = explainer.explain_instance(sample_fake, predict_proba_wrapper, num_features=10)
from IPython.display import HTML, display
display(HTML(exp_fake.as_html(text=sample_fake)))
'''

    nbformat.write(nb, nb_path)
    print("Fixed LIME notebook issues properly.")
