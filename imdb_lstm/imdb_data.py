from keras.datasets import imdb
from keras.preprocessing import sequence
from data_utils import shuffle_split


# noinspection PyUnusedLocal
def load_data(num_words, max_length, fraction=1.0):
    """Loads the IMDB movie review dataset from keras.datasets.
        The dataset is encoded using word-embedding, with each review padded.

    # Arguments
        num_words: int 
            The number of total words in the dataset to encode.
        max_length: int
            The max number of words in a review to use. Reviews whose length
            exceeds this number will be truncated.
        fraction: float between 0 and 1.0 (Default 1.0) 
            Optional fraction of the data to use. Affects all returned datasets.

    # Returns
        A dict containing 6 word-embedded ndarrays:
        train_data: the IMDB training set of 25000 instances 
            (or less if fraction is set)
        train_labels: the labels for train_data.
        valid_data: half of the instances from the IMDB test set (or less if 
            fraction is set).
        valid_labels: the labels for valid_data.
        test_data: the other half of the instances from the IMDB test set (or 
            less if fraction is set). No instances overlap with valid_data.
        test_labels: the labels for test_data.

    # Raises
        ValueError: If the fraction is not between 0 and 1.
    """
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError('fraction must be 0-1. Was {}'.format(fraction))

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

    # reduce size of training set for rapid prototyping
    if fraction < 1.0:
        (x_train, y_train), (x_extra, y_extra) = shuffle_split(x_train, y_train, fraction=fraction)
        (x_test, y_test), (x_extra, y_extra) = shuffle_split(x_test, y_test, fraction=fraction)

    (x_valid, y_valid), (x_test, y_test) = shuffle_split(x_test, y_test, fraction=0.5)
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_valid = sequence.pad_sequences(x_valid, maxlen=max_length)
    x_test = sequence.pad_sequences(x_test, maxlen=max_length)
    return {'train_data': x_train,
            'train_labels': y_train,
            'valid_data': x_valid,
            'valid_labels': y_valid,
            'test_data': x_test,
            'test_labels': y_test}
