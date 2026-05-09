import nbformat
import os

nb_path = 'notebooks/05_deep_learning_models.ipynb'
if os.path.exists(nb_path):
    nb = nbformat.read(nb_path, as_version=4)
    
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if 'evaluation_strategy=' in cell.source:
                cell.source = cell.source.replace('evaluation_strategy=', 'eval_strategy=')
                
    nbformat.write(nb, nb_path)
    print("Fixed TrainingArguments compatibility in Notebook 05.")
