import numpy                as _np
import os                   as _os
import time                 as _time
import multiprocessing      as _mp
import matplotlib.pyplot    as _plt
import hashlib              as _hl

from typing import Callable as _Callable, Iterable as _Iterable

def l_map(func: _Callable, array: _Iterable) -> list:
    """Compact map function

    Args:
        func (Callable): the function to apply
        array (Iterable): data to map

    Returns:
        list: result of applying map
    """
    return list(map(func, array))

def np_map(func: _Callable, array: _Iterable) -> _np.ndarray:
    """Compact map funciton, converts output into np.array

    Args:
        func (Callable): the function to apply
        array (Iterable): data to map

    Returns:
        np.ndarray: result of applying map, transformed into array
    """
    return _np.array(l_map(func, array))

def _hms(time: float) -> str:
    """Stringify time in m:s:ms fromat

    Args:
        time (float): seconds elapsed

    Returns:
        str: formatted in m:s:ms
    """
    h = time//3600
    time -= h*3600
    m = time//60
    time -= m*60
    s, ms = divmod(time, 1)
    return f'{int(m)}:{int(s)}:{int(ms*1000)}'

def _guard(array) -> _np.ndarray:
    """Set NaNs and infinite values to 0

    Args:
        array (np.ndarray): input array

    Returns:
        np.ndarray: array with substitutions
    """
    if not _np.isfinite(array).all(): print(f'\033[91mFound not-finite values, replacing with 0s\033[0m')
    array = _np.nan_to_num(array, nan=0, posinf=0, neginf=0)
    return array

def _pipe_transform_apply(steps: _Iterable[_Callable], array: _np.ndarray, verbose: bool = True) -> _np.ndarray:
    """transform the input array and outputs statistics of the computation

    Args:
        steps (Iterable[Callable]): steps to apply
        array (np.ndarray): input array
        verbose (bool, optional): print tensor shapes for each step. Defaults to True.

    Returns:
        np.ndarray: reshaped array
    """
    from datetime import datetime
    t = []
    if verbose: print(f'    Initial shape: \033[34m{array.shape}\033[0m')
    start = datetime.now()
    for p in steps:
        if verbose: print(f'\033[32m{p.__name__}\033[0m')
        ti = _time.time()
        array = _guard(p(array))
        dt = _time.time() - ti
        t.append(dt)
        if verbose: print(f'    \033[34m{array.shape}\033[0m')
    print(f'Total elapsed computation time: {datetime.now() - start}')
    return array, (steps, t)

def _unpacking_apply_along_axis(all_args):
    """Unpacks args for `np.apply_along_axis`"""
    (func1d, axis, arr, args, kwargs) = all_args
    return _np.apply_along_axis(func1d, axis, arr, *args, **kwargs)

def _parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """Multithreaded variant of `np.apply_along_axis`"""
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis: arr = arr.swapaxes(axis, effective_axis)
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs) for sub_arr in _np.array_split(arr, _mp.cpu_count())]
    pool = _mp.Pool()
    individual_results = pool.map(_unpacking_apply_along_axis, chunks)
    pool.close(); pool.join()
    arr = _np.concatenate(individual_results)
    # In case of functions returning multiple values, preserve order of dimensions
    if effective_axis != axis: arr = arr.swapaxes(axis, effective_axis)
    return arr

def to_ax(function: _Callable, axis: int = -1, threadsafe: bool = False) -> _Callable:
    """Returns a lambda, applying the desired function along the axis of the input tensor for using in `Transformer`

    Args:
        function (Callable): the transform to apply
        axis (int, optional): the axis along which the transform must be applied. Defaults to -1.
        threadsafe (bool, optional): disable multithreaded version of function, for use with lambdas. Defaults to False.

    Returns:
        Callable: lambda to transform the tensor
    """
    x = None
    if threadsafe: x = lambda x: _np.apply_along_axis(function, axis, x)
    else:    x = lambda x: _parallel_apply_along_axis(function, axis, x)
    x.__name__ = f'to_ax[{axis}]: {function.__name__}'
    return x

def moveaxis(from_axis: int, to_axis: int) -> _Callable:
    """Move the axis of a tensor from position `from_axis` to position `to_axis`, for uing in `Transformer`

    Args:
        from_axis (int): index of the origin axis
        to_axis (int): index of the destination axis

    Returns:
        Callable: lambda to transform the tensor
    """
    t = lambda x: _np.moveaxis(x, from_axis, to_axis)
    t.__name__ = f"moveaxis[{from_axis}, {to_axis}]"
    return t

def _plot_timing(pipe, val, name, size: tuple = (12,9)):
    _plt.figure(figsize=size)
    _plt.bar(range(len(pipe)), val)
    _plt.xticks(range(len(pipe)),
            map(lambda x: x.__name__, pipe),
            rotation=30,
            horizontalalignment='right')
    _plt.grid(True)
    _plt.xlabel(f'[{name}] Processing steps')
    _plt.ylabel(f'[{name}] Processing time [s]')
    _plt.title(name)
    _plt.show()

def _tensor_apply_transform(data: _np.ndarray, name: str = 'Unknown', transforms: _Iterable[_Callable] = [], verbose = True) -> _np.ndarray:
    """Toplevel function for transforming tensors

    Args:
        data (np.ndarray): the tensor to transform
        name (str, optional): transformation name. Defaults to 'Unknown'.
        transforms (Iterable[Callable], optional): list of transformations to apply to the tensor. Defaults to [].
        verbose (bool, optional): print tensor shape at each step and dosply timing report. Defaults to True.

    Returns:
        np.ndarray: transformed tensor
    """
    print(f'Transforming "{name}"')
    data, stats = _pipe_transform_apply(transforms, data, verbose)
    if verbose: _plot_timing(*stats, name = name)
    return data

class Transformer:
    def __init__(self, name: str, verbose: bool = True, transforms: _Iterable[_Callable] = []):
        """Create a new `Transformer` object

        Args:
            name (str): transformer name
            verbose (bool, optional): reports tensor shape and timing for each step. Defaults to True.
            transforms (Iterable[Callable], optional): list of transformations to apply. Defaults to [].
        """
        self.l = lambda x: _tensor_apply_transform(x, name, transforms, verbose)
        self.l.__name__ = f'transformer: {name} - {str(l_map(lambda x: x.__name__, transforms))}'
    def call(self, x):
        return self.l(x)

def value_repeat(times: int, axis: int = -1) -> _Callable:
    """Repeat the values on specified axis, for using in `Transformer`

    Args:
        times (int): number of times to repeat the values
        expand_axis (int, optional): axis to expand to repeat the valuess. Defaults to -1.

    Returns:
        Callable: lambda to transform the tensor
    """
    l = lambda x: _np.repeat(_np.expand_dims(x, axis=axis), times, axis=axis)
    l.__name__ = f'repeat[{times}, {axis}, {axis}]'
    return l

# Segment channels
def split(segments_n: int) -> _Callable:
    """Split array, for using in `Transformer`

    Args:
        segments_n (int): number of segments

    Returns:
        Callable: lambda to transform the tensor
    """
    l = lambda x: _np.split(x, segments_n)
    l.__name__ = f'segment[{segments_n}]'
    return l

def _onehot_back(x,n):
    if n is None: _, n = _np.unique(x, return_counts=True)
    return _np.eye(n)[x] 

def onehot(n = None) -> _Callable:
    """Encode the vector using one-hot encoding, for use in `Transform`

    Args:
        n (integer, optional): number of classes, auto-determine if None. Defaults to None.

    Returns:
        Callable: lambda to transform the tensor
    """
    return lambda x: _onehot_back(x,n)

def flatten(x: _np.ndarray) -> _np.ndarray:
    """Flatten, for use in `Transform`

    Args:
        x (np.ndarray): input tensor

    Returns:
        np.ndarray: transformed tensor
    """
    return x.flatten()

# Deterministic hash, for caching
def _det_hash(input_string: str):
    encoded_input = input_string.encode('utf-8')
    hash_object = _hl.sha256()
    hash_object.update(encoded_input)
    return hash_object.hexdigest()

def _hashtag(x: str): return _det_hash(str(x))[2:10].upper().zfill(8)

def cached_transform(input: _np.ndarray, transformer: Transformer, name: str, path: str = './', force: bool = False) -> _np.ndarray:
    """if a cache file "<`path`><`name`>_<`Tensor hash`>.npy" exists, load the tensor from it, otherwise apply the tensor to the specified input

    Args:
        input (np.ndarray): input tensor
        transformer (Transformer): Transformer to applyt to the tensor
        name (str): transformation name
        path (str, optional): path to search for file name. Defaults to './'.
        force (bool, optional): force transformation even if cached file exists, overwrite the cache file. Defaults to False.

    Returns:
        np.ndarray: transformed tensor
    """
    transformer_pipe_name = '#'.join(transformer.__name__)
    name_hash = _hashtag(transformer_pipe_name)
    cache_file = f'{path}{name}_{name_hash}.npy'
    filename = cache_file.split('/')[-1]
    t = None
    if force or not _os.path.isfile(cache_file):
        if not _os.path.isfile(cache_file):
            print(f'\033[33m{filename} not found, generating\033[0m')
        else:
            print(f'\033[33mForced generation of {filename}\033[0m')
        t = transformer(input)
        _np.save(cache_file, t)
        print(f'\033[33mTensor saved to {filename}\033[0m')
    else:
        t = _np.load(cache_file)
        print(f'\033[33mTensor loaded from {filename}\033[0m')
        print(f'    {t.shape}')
    return t
