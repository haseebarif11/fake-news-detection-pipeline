import nbformat
import os

nb_path = 'notebooks/04_feature_engineering_and_modeling.ipynb'
if os.path.exists(nb_path):
    nb = nbformat.read(nb_path, as_version=4)
    
    # Check if the last cell is already a conclusion, and if it's the mangled one, remove it
    if len(nb.cells) > 0 and nb.cells[-1].cell_type == 'markdown' and 'Model Performance Conclusion' in nb.cells[-1].source:
        nb.cells.pop()
        
    obs = nbformat.v4.new_markdown_cell('''### Model Performance Conclusion

Based on the benchmark evaluations across both the **News** and **Tweets** datasets:

* **Top Performer:** **Random Forest** achieved the highest overall Accuracy and F1-Score. Its ensemble tree structure effectively captured complex, non-linear relationships between the sparse TF-IDF text features and our dense NLP features (sentiment, readability).
* **Strong Baseline:** **Logistic Regression** proved to be an excellent, fast, and highly interpretable baseline, closely trailing Random Forest in performance.
* **Underperformer:** **Multinomial Naive Bayes** struggled slightly in comparison, likely due to the assumption of feature independence which doesn't hold perfectly for complex sentence structures and dense engineered features.

**Next Steps:** Random Forest is the chosen model for deployment due to its robust performance across both long-form news and short-form social media text.''')
    nb.cells.append(obs)
    nbformat.write(nb, nb_path)
    print("Added conclusion cell successfully!")
else:
    print("Notebook not found!")
