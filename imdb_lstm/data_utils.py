import numpy as np
import pandas as pd


def shuffle_split(data, labels, fraction):
    """Shuffles the rows of the given data, and matches the order of the given labels.

    # Arguments
        data: ndarray
            The data matrix with each row representing an instance.
        labels: ndarray
            A vector with the labels for each row in the data. 
        fraction: float between 0 and 1.0
            Fraction of the data to use. Must be a number between 0 and 1.
            The fraction is the proportion of instances (rows) returned
            in the first tuple. Hence a value of 0.2 will return 20% of the 
            instances in the first tuple, and the remaining 80% of the instances
            in the second tuple.

    # Returns
        A pair of tuples split from the original data and labels.
        Each tuple contains a data and label ndarray. The first tuple contains
        the proportion of instances given in fraction, and the second tuple 
        contains the remainder.
        The data will be shuffled before splitting, so random instances will 
        occur in each tuple.

    # Raises
        ValueError: If the fraction is not between 0 and 1.
    """
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError('fraction must be 0-1. Was {}'.format(fraction))

    # shuffle
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    # split
    idx = int(len(indices) * (1.0-fraction))
    data1 = data[idx:]
    data2 = data[:idx]
    labels1 = labels[idx:]
    labels2 = labels[:idx]

    return (data1, labels1), (data2, labels2)


def extract_results(trials):
    """Given a Hyperopt Trials object, extract a DataFrame of result metrics for each individual trial
    
    # Arguments
        trials: A Hyperopt Trials object that has previously been operated on via a call to hyperopt.fmin()
        
    # Returns
        A Pandas DataFrame. The columns are the metrics of the fit during the trial
        (as determined by the return value of the objective function), and the value of each hyperparameter
        that was varied during the minimization. Each row is a separate trial.
    """
    results = []
    for i in range(len(trials.trials)):
        result = {}
        result.update(trials.results[i])
        result.update(trials.miscs[i]['vals'])
        results.append(result)
    df = pd.DataFrame(results)
    columns = df.columns.values.tolist()
    columns.remove('status')
    columns.remove('loss')
    columns.remove('acc')
    columns.sort()
    df = df.reindex(['status', 'loss', 'acc'] + columns, axis=1)
    df.sort_values(by='loss', ascending=True, inplace=True)
    return df


def set_seed(seed, single_thread=False):
    """ Set the random seed for both tensorflow and theano backends
        See: https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
        Also see: https://github.com/fchollet/keras/issues/2280 for details and limitations
        
    # Arguments
        seed: int
            A random seed number
        single_thread: bool default False
            Whether to run experiments in single_thread mode. In some cases this necessary to guarentee 
            reproducible results, though performance will suffer.
    """
    np.random.seed(seed)
    import tensorflow as tf
    tf.set_random_seed(seed)
    import random as rn
    rn.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['HYPEROPT_FMIN_SEED'] = str(seed)
    if single_thread is True:
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        from keras import backend as K
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
