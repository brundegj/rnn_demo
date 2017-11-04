from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.datasets import imdb
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing import sequence

from data_utils import shuffle_split


def get_hyperparams():
    return {
        'seed': 11,
        'top_words': 5000,
        'max_word_length': 500,
        'embedded_vector_length': 32,
        'lstm_units': 100,
        'batch_size': 64,
        'epochs': 10,
        'model_file': './imdb_lstm_model.hdf5'
    }


def load_data(hyperparams, fraction=1.0):
    (x_train_original, y_train_original), (x_test_original, y_test_original) \
        = imdb.load_data(num_words=hyperparams['top_words'])

    # reduce size of training set for rapid prototyping
    if fraction < 1.0:
        (x_train, y_train), (x_extra, y_extra) = shuffle_split(x_train_original, y_train_original, fraction=fraction)
    else:
        (x_train, y_train) = (x_train_original, y_train_original)

    (x_valid, y_valid), (x_test, y_test) = shuffle_split(x_test_original, y_test_original, fraction=0.5)
    max_word_length = hyperparams['max_word_length']
    x_train = sequence.pad_sequences(x_train, maxlen=max_word_length)
    x_valid = sequence.pad_sequences(x_valid, maxlen=max_word_length)
    x_test = sequence.pad_sequences(x_test, maxlen=max_word_length)
    return {'train_data': x_train,
            'train_labels': y_train,
            'valid_data': x_valid,
            'valid_labels': y_valid,
            'test_data': x_test,
            'test_labels': y_test}


def create_model(hyperparams):
    model = Sequential()
    model.add(Embedding(hyperparams['top_words'],
                        hyperparams['embedded_vector_length'],
                        input_length=hyperparams['max_word_length']))
    model.add(LSTM(hyperparams['lstm_units']))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(hyperparams, model, data):
    monitor = 'val_loss'
    early_stopping = EarlyStopping(monitor=monitor, patience=1)
    model_checkpoint = ModelCheckpoint(filepath=hyperparams['model_file'],
                                       monitor=monitor,
                                       save_best_only=True,
                                       verbose=1)
    model.fit(data['train_data'], data['train_labels'],
              validation_data=(data['valid_data'], data['valid_labels']),
              batch_size=hyperparams['batch_size'],
              epochs=hyperparams['epochs'],
              callbacks=[model_checkpoint, early_stopping])
    return load_model(hyperparams['model_file'])


def run_exp():
    hyperparams = get_hyperparams()
    data = load_data(hyperparams)
    model = create_model(hyperparams)
    trained_model = train_model(hyperparams, model, data)
    scores = trained_model.evaluate(data['test_data'], data['test_labels'], verbose=0)
    print('Accuracy: {}'.format(scores[1]*100))


run_exp()
