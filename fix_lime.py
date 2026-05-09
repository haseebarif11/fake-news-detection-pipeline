import nbformat
import os

nb_path = 'notebooks/06_interpretability_and_comparison.ipynb'
if os.path.exists(nb_path):
    nb = nbformat.read(nb_path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'show_in_notebook' in cell.source:
            cell.source = cell.source.replace('.show_in_notebook(text=sample_fake)', '.as_html(text=sample_fake)')
            cell.source = cell.source.replace('.show_in_notebook(text=sample_real)', '.as_html(text=sample_real)')
            cell.source = cell.source.replace('exp_fake.as_html', 'display(HTML(exp_fake.as_html') + '))'
            cell.source = cell.source.replace('exp_real.as_html', 'display(HTML(exp_real.as_html') + '))'
            
    # Quick fix for cell 5
    nb.cells[5].source = '''
print("Generating explanation for Fake News Sample...")
exp_fake = explainer.explain_instance(sample_fake, predict_proba_wrapper, num_features=10)
from IPython.display import HTML, display
display(HTML(exp_fake.as_html(text=sample_fake)))
'''
    nb.cells[6].source = '''
print("Generating explanation for Real News Sample...")
exp_real = explainer.explain_instance(sample_real, predict_proba_wrapper, num_features=10)
from IPython.display import HTML, display
display(HTML(exp_real.as_html(text=sample_real)))
'''

    nbformat.write(nb, nb_path)
    print("Fixed LIME notebook issues.")
