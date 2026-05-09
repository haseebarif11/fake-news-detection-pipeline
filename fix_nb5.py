import nbformat
import os

nb_path = 'notebooks/05_deep_learning_models.ipynb'
if os.path.exists(nb_path):
    nb = nbformat.read(nb_path, as_version=4)
    
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if '# news_trainer.train()' in cell.source:
                cell.source = cell.source.replace('# news_trainer.train()', 'news_trainer.train()')
                cell.source = cell.source.replace('# print(news_trainer.evaluate())', 'print(news_trainer.evaluate())')
            if '# tweet_trainer.train()' in cell.source:
                cell.source = cell.source.replace('# tweet_trainer.train()', 'tweet_trainer.train()')
                cell.source = cell.source.replace('# print(tweet_trainer.evaluate())', 'print(tweet_trainer.evaluate())')
                
    nbformat.write(nb, nb_path)
    print("Uncommented training lines in Notebook 05.")
