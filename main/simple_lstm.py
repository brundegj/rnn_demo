from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import load_model
from imdb_data import load_data
from hyperopt import STATUS_OK


class SimpleImdbLstm():
    """A holder for a trainable keras model with configurable hyperparameters.
    """

    def __init__(self, hyperparameters=None):
        # Set default values. Can be overridden.
        self.hyperparams = {
            'seed': 11,
            'top_words': 5000,
            'max_word_length': 500,
            'embedded_vector_length': 32,
            'lstm_units': 100,
            'batch_size': 64,
            'epochs': 1,
            'activation': 'sigmoid',
            'optimizer': 'adam',
            'loss': 'binary_crossentropy',
            'monitor': 'val_loss',
            'model_file': './simple_imdb_lstm_model.hdf5'
        }

        # override default hyperparams for any new values specified
        if hyperparameters is not None:
            self.hyperparams.update(hyperparameters)

        self.set_seed()

    def set_seed(self):
        """ Set the random seed for both tensorflow and theano backends
            See: https://github.com/fchollet/keras/issues/2280 for details and limitations
            also see: https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
        """
        if 'seed' in self.hyperparams:
            seed = self.hyperparams['seed']
            import numpy as np
            np.random.seed(seed)
            import tensorflow as tf
            tf.set_random_seed(seed)
            import random as rn
            rn.seed(seed)
            import os
            os.environ['PYTHONHASHSEED'] = str(seed)
            os.environ['HYPEROPT_FMIN_SEED'] = str(seed)
            if 'single_thread' in self.hyperparams and self.hyperparams['single_thread'] is True:
                session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
                from keras import backend as K
                sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
                K.set_session(sess)

    def create_model(self, hyperparams):
        model = Sequential()
        model.add(Embedding(hyperparams['top_words'],
                            hyperparams['embedded_vector_length'],
                            input_length=hyperparams['max_word_length']))
        model.add(LSTM(hyperparams['lstm_units']))
        model.add(Dense(1, activation=hyperparams['activation']))
        model.compile(optimizer=hyperparams['optimizer'],
                      loss=hyperparams['loss'],
                      metrics=['accuracy'])
        return model

    def objective(self, space):
        data = load_data(space['top_words'], space['max_word_length'], fraction=0.01)
        model = self.create_model(space)

        early_stopping = EarlyStopping(monitor=space['monitor'], patience=1)
        model_checkpoint = ModelCheckpoint(filepath=space['model_file'],
                                           monitor=space['monitor'],
                                           save_best_only=True,
                                           verbose=1)
        model.fit(data['train_data'], data['train_labels'],
                  validation_data=(data['valid_data'], data['valid_labels']),
                  batch_size=space['batch_size'],
                  epochs=space['epochs'],
                  callbacks=[model_checkpoint, early_stopping])

        best_model = load_model(space['model_file'])
        print(best_model.metrics_names)
        results = best_model.evaluate(data['valid_data'], data['valid_labels'], batch_size=2500)
        loss = results[0]
        acc = results[1]
        return {'loss': loss, 'acc': acc, 'status': STATUS_OK}

