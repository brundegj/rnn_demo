import numpy as np


def shuffle_split(data, labels, fraction):
    """Shuffles the rows of the given data, and matches the order of the given labels.

    # Arguments
        data: A numpy ndarray with each row representing an instance.
        labels: A numpy ndarray with the labels of each row in the data. 
        fraction: Fraction of the data to use. Must be a number between 0 and 1.
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
