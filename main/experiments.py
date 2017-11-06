from simple_lstm import SimpleImdbLstm
import model_scanner
from hyperopt import hp
import numpy as np


hyperparams = {
    'top_words': 5000,
    'max_word_length': 500,
    'batch_size': hp.choice('batch_size', np.arange(1, 100, dtype=int))
}
simple_imdb_lstm = SimpleImdbLstm(hyperparams)
results = model_scanner.find_best_model(simple_imdb_lstm)
print(results)
