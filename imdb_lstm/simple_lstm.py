from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import load_model
from imdb_data import load_data
from hyperopt import STATUS_OK


class SimpleImdbLstm:
    """A holder for a trainable keras model with configurable hyperparameters.
    """

    def __init__(self, hyperparameters=None):
        # Set default values. Can be overridden.
        self.hyperparams = {
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
            'model_file': './simple_imdb_lstm_model.hdf5',
            'data_fraction': 1.0
        }

        # override default hyperparams for any new values specified
        if hyperparameters is not None:
            self.hyperparams.update(hyperparameters)

    def create_model(self, hyperparams):
        model = Sequential()
        model.add(Embedding(int(hyperparams['top_words']),
                            hyperparams['embedded_vector_length'],
                            input_length=hyperparams['max_word_length']))
        model.add(LSTM(hyperparams['lstm_units']))
        model.add(Dense(1, activation=hyperparams['activation']))
        model.compile(optimizer=hyperparams['optimizer'],
                      loss=hyperparams['loss'],
                      metrics=['accuracy'])
        return model

    def objective(self, space):
        data = load_data(int(space['top_words']), int(space['max_word_length']), fraction=space['data_fraction'])
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
