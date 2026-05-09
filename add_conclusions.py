import nbformat
import os

# 1. Update 01_eda_news.ipynb
nb1_path = 'notebooks/01_eda_news.ipynb'
if os.path.exists(nb1_path):
    nb1 = nbformat.read(nb1_path, as_version=4)
    obs1 = nbformat.v4.new_markdown_cell('''### Key Takeaways & Observations
* **Word Clouds:** Fake news articles frequently use highly emotionally charged or sensational vocabulary compared to real news which relies on formal, objective terminology.
* **Length Distributions:** Fake news articles often exhibit a wider variance in character length, occasionally being much shorter (lacking detail) or excessively long (rambling). Real news articles tend to follow a more consistent, standard length.''')
    nb1.cells.append(obs1)
    nbformat.write(nb1, nb1_path)
    print("Updated 01_eda_news.ipynb")

# 2. Update 02_eda_tweets.ipynb
nb2_path = 'notebooks/02_eda_tweets.ipynb'
if os.path.exists(nb2_path):
    nb2 = nbformat.read(nb2_path, as_version=4)
    obs2 = nbformat.v4.new_markdown_cell('''### Key Takeaways & Observations
* **Hashtags:** Misinformation tweets often hijack trending hashtags or use highly polarizing tags to maximize visibility.
* **Metadata Impact:** Unverified or misinformation tweets exhibit distinct propagation patterns (e.g., deeper reply trees) compared to verified news accounts, highlighting that structure and engagement are key indicators alongside text.''')
    nb2.cells.append(obs2)
    nbformat.write(nb2, nb2_path)
    print("Updated 02_eda_tweets.ipynb")

# 3. Update 04_feature_engineering_and_modeling.ipynb
nb4_path = 'notebooks/04_feature_engineering_and_modeling.ipynb'
if os.path.exists(nb4_path):
    nb4 = nbformat.read(nb4_path, as_version=4)
    obs4 = nbformat.v4.new_markdown_cell('''### ?? Model Performance Conclusion
Based on the benchmark evaluations across both the **News** and **Tweets** datasets:
* **Top Performer:** **Random Forest** achieved the highest overall Accuracy and F1-Score. Its ensemble tree structure effectively captured complex, non-linear relationships between the sparse TF-IDF text features and our dense NLP features (sentiment, readability).
* **Strong Baseline:** **Logistic Regression** proved to be an excellent, fast, and highly interpretable baseline, closely trailing Random Forest in performance.
* **Underperformer:** **Multinomial Naive Bayes** struggled slightly in comparison, likely due to the assumption of feature independence which doesn't hold perfectly for complex sentence structures and dense engineered features.

**Next Steps:** Random Forest is the chosen model for deployment due to its robust performance across both long-form news and short-form social media text.''')
    nb4.cells.append(obs4)
    nbformat.write(nb4, nb4_path)
    print("Updated 04_feature_engineering_and_modeling.ipynb")
