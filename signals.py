from typing import Callable

from scipy.signal import butter as _butter, filtfilt as _filtfilt
def pass_filter(band: tuple[float, float], sample_f:float, btype='band', order: int = 3) -> Callable:
    
    """Apply butter filter to a signal, for use with `Transformer`

    Args:
        band (tuple[float, float]) two ends of the band to use, low end is not used for low-pass, high end is not used for high-pass
        sample_f (float): sampling frequency of signal
        btype (str, optional): type of filter: "low", "high" and "band" pass. Defaults to 'band'.
        order (int, optional): butter fiter order. Defaults to 3.

    Raises:
        ValueError: Unknown btype 

    Returns:
        Callable: lambda for transforming tensor
    """
    low, high= band
    nyq  = 0.5 * sample_f
    a = None; b = None; l = None
    if btype == 'band':
        a, b = _butter(order, [low/nyq, high/nyq], btype=btype)
        l = lambda x: _filtfilt(a, b, x)
        l.__name__ = f'filter_band@[{low},{high}]'
    elif btype == 'low':
        a, b = _butter(order, high/nyq, btype=btype)
        l    = lambda x: _filtfilt(a, b, x)
        l.__name__ = f'filter_low@[{high}]'
    elif btype == 'high':
        a, b = _butter(order, low/nyq, btype=btype)
        l = lambda x: _filtfilt(a, b, x)
        l.__name__ = f'filter_high@[{low}]'
    else: raise ValueError(f'Unknown btype: {btype}')
    return l

import numpy as np
def remove_baseline(baseline_samples: int, method: str = 'median') -> Callable:
    """Subtract baseline from a sample sequence using the specified method and remove the baseline samples

    Args:
        baseline_samples (int): length of the baseline
        method (str, optional): method to compute baseline, accetps "median" or "mean". Defaults to 'median'.

    Raises:
        ValueError: Unknown method

    Returns:
        Callable: lambda to transform tensor
    """
    l = None
    if method == 'median':
        l = lambda x: x[baseline_samples:] - np.median(x[:baseline_samples])
    elif method == 'mean':
        l = lambda x: x[baseline_samples:] - np.mean(x[:baseline_samples])
    else:
        raise ValueError(f'Unknown baseline-removal method: {method}')
    l.__name__ = f'remove_baseline[{method}]'
    return l

def diff_entropy(signal: np.ndarray) -> float:
    """Compute differential entropy

    Args:
        signal (np.ndarray): signal

    Returns:
        float: differential entropy
    """
    return np.log(2 * np.pi * np.e * np.var(signal, ddof=1)) / 2