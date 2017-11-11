import numpy as np
from hyperopt import hp
from math import log

import model_scanner
import data_utils
from simple_lstm import SimpleImdbLstm

data_utils.set_seed(11)

hyperparams = {
    'top_words': hp.qloguniform('top_words', log(100), log(10000), 1),
    'max_word_length': hp.qloguniform('max_word_length', log(50), log(500), 1),
    'batch_size': hp.choice('batch_size', np.arange(1, 100, dtype=int)),
    'data_fraction': 1.0
}
simple_imdb_lstm = SimpleImdbLstm(hyperparams)
trials = model_scanner.find_best_model(simple_imdb_lstm, 'exp1', max_evals=10)

data_utils.extract_results(trials).to_html('../simple_imdb_lstm_results.html')
