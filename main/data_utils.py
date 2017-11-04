import numpy as np


def shuffle_split(data, labels, fraction):
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
