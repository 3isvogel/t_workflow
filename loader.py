from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import Callable
from collections.abc import Iterable
import os

DRIVE_DIR = 'DRIVE_DIRECTORY_NOT_SET'
CACHE_DIR = 'CACHE_DIRECTORY_NOT_SET'

def drive_mount(path: str = '/content/drive') -> None:
    """Mount Google Drive to the selected path

    Args:
        drive (str): Path to mount the drive to

    Raises:
        ValueError: If either import of `google.colab.drive` or `drive.mount` fails
    """
    try:
        from google.colab import drive
        drive.mount(path)
        global DRIVE_DIR
        DRIVE_DIR = path
    except Exception as e:
        raise ValueError(f'Failed to mount Google drive: {e}')

def set_dataset_cache(directory: str) -> None:
    """Set the cache directory for dataset loading, this may be different from the `transform cache` directory

    Args:
        directory (str): Dataset_cache_directory
    """
    global CACHE_DIR
    CACHE_DIR = directory

import _pickle as cPickle
def _default_load_pickle(name: str) -> dict:
    t = cPickle.load(
        open(name), 'rb',
        encoding='latin1')
    print('‾',end='')
    return t

def _load_parallel_dataset(file_list: Iterable, open_function: Callable) -> Iterable[dict]:
    dataset = [i for i in range(len(file_list))]; i = 0
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(open_function, name) for name in file_list]
        for future in as_completed(futures):
            dataset[i] = future.result()
            i += 1
    return dataset

import numpy as np
from transformer import np_map, hashtag
DATA_CACHE = '' 
LABL_CACHE = ''
def load_pickle(file_list: list[str], open_function: Callable = _default_load_pickle) -> tuple[np.ndarray, np.ndarray]:
    """Load pickle dataset

    Args:
        file_list (list[str]): List of files to load
        open_function (Callable, optional): function to open the files. Defaults to _default_load_pickle.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tensors containing the data and the labels
    """
    global DATA_CACHE
    global LABL_CACHE
    ht = hashtag(str(file_list))
    DATA_CACHE = os.path.join(CACHE_DIR, f'_data-{ht}.npy')
    LABL_CACHE = os.path.join(CACHE_DIR, f'_labl{ht}.npy')
    t_data = None; t_labels = None
    # If already saved as numpy array: load the numpy array instead
    if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)
    if os.path.isfile(DATA_CACHE) and os.path.isfile(LABL_CACHE):
        t_data   = np.load(DATA_CACHE)
        t_labels = np.load(LABL_CACHE)
    else:
        print(f'{"_" * len(file_list)}')
        dataset = _load_parallel_dataset(file_list, open_function); print()

        if not os.path.isfile(DATA_CACHE):
            t_data    = np_map(lambda x: x.get('data'),   dataset)
            np.save(DATA_CACHE, t_data)

        if not os.path.isfile(LABL_CACHE):
            t_labels  = np_map(lambda x: x.get('labels'), dataset)
            np.save(LABL_CACHE, t_labels)

    print(f'Loaded data:   {t_data.shape}')
    print(f'Loaded labels: {t_labels.shape}')
    return t_data, t_labels