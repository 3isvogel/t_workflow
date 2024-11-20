import numpy as np
def get_numerosity(labels: np.ndarray) -> np.ndarray:
    """Get numerosity from a one-hot encoded numpy array with two dimensions (including one-hot)

    Args:
        labels (np.ndarray): labels to count

    Returns:
        np.ndarray: occurrencies of elements, element coded with 1 in position k is in the k'th position of the array
    """
    # using axis as below, should account for any shape, assuming labels occurs in last position,
    # once per data point
    axis=tuple(i for i in range(labels.ndim - 1))
    return np.sum(labels, axis=axis)

# Compute class weight
def _get_class_weight(labels):
    class_num = get_numerosity(labels)
    print(f'Count: {class_num}')
    total_samples = np.sum(class_num)
    # Previous weight was wrong
    class_weight = {i: total_samples / (len(class_num) * class_num[i]) for i in range(len(class_num))}
    print(f'Normalized class weight: {class_weight}')
    return class_weight

#   for each vector, for the love of God use labels as the last argument
from sklearn.model_selection import train_test_split
def _set_split(data, split):
    sets = []
    for i in range(len(split)):
        data = train_test_split(*data, stratify=data[-1], train_size = split[i], shuffle = True)
        setk = data[ ::2]
        data = data[1::2]
        sets.append(setk)
    sets.append(data)
    return sets

def _to_remove(array, occurrencies):
    occurrencies = occurrencies.astype(int)
    mask = np.zeros(len(array), dtype=int)
    for i in range(len(occurrencies)):
        indices = np.argwhere(array[..., i] == 1)[..., 0]
        indices = np.unique(indices)
        indices = np.random.choice(indices, occurrencies[i], replace = False)
        mask[indices] = 1
    return mask

def _downsample(sets):
    # TODO: shuffle the set somewhere
    y_cnt = get_numerosity(sets[-1])
    y_diff = y_cnt - np.min(y_cnt)  # remove the minimum from all of them, obtaining the exact number of values to eremove
    remove_mask = _to_remove(sets[-1], y_diff)
    keep_mask = 1 - remove_mask
    
    keep_set   = list(map(lambda x: x[keep_mask], sets))
    remove_set = list(map(lambda x: x[remove_mask], sets))

    return keep_set, remove_set

def prepare_sets(*data, labels: np.ndarray, splits: list[float], downsample: bool = False) -> tuple[list[np.ndarray], list[np.ndarray], dict[float]]:
    """Prepare sets for training, splitting the setsaccording to the values in `split`

    Args:
        labels (np.ndarray): one-hot encoded numpy array, must be bidimensional (including the one-hot encoding)
        splits (list[float]): values between 0 and 1, these are used as percentages to split the data, note that the process is iterative: the first value is the percentage of the data, the second is the value of the remaining data, and so on, (don't use tuples, or python become stupid)
        downsample (bool, optional): balance data downsampling it. Defaults to False.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray], dict[float]]: `sets, remainder, class_weight` the first element is the actual split, remainder is the remaining data from downsampling, `class_weight` is the inverse of numersoity, as required by `Model.fit` for reducing overfit, according to the value of `downsample` either `remainder` or `class_weight` is None
    """
    class_weight = None
    remainder = None
    data = list(data)
    data.append(labels)    # ensures that labels is in the correct position, also helps visualizing labels data
    if downsample:
        data, remainder = _downsample(data)
    else:
        class_weight = _get_class_weight(data[-1])
    sets = _set_split(data, splits)
    return sets, remainder, class_weight