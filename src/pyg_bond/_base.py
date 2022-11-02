from pyg_base import dt, df_reindex, is_date, is_ts
import pandas as pd

RATE_FMT = 100
_rate_formats = {'%' : 100, 'bp': 10000, 1: 1, 100: 100, 10000: 10000}

def rate_format(rate_fmt = None):
    if not rate_fmt:
        return RATE_FMT
    if rate_fmt not in _rate_formats:
        raise ValueError(f'rate format must be in {_rate_formats}')
    return _rate_formats[rate_fmt]

def annual_freq(freq):
    """
    >>> assert annual_freq('6M') == 2
    >>> assert annual_freq('360d') == 1

    Parameters
    ----------
    freq : either integer or '6m'

    """
    if isinstance(freq, str):
        freq = freq.lower()
        print('freq is %s'%freq)
        return int(round(dict(d = 365, m = 12, y = 1, q = 4, w = 52)[freq[-1]] / int(freq[:-1]),0))
    else:
        return freq

