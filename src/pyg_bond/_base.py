from pyg_base import dt, df_reindex, is_date, is_ts
import pandas as pd

RATE_FMT = 1
_rate_formats = {'%' : 100, 'bp': 10000, 1: 1, 100: 100, 10000: 10000}

def rate_format(rate_fmt = None):
    if not rate_fmt:
        return RATE_FMT
    if rate_fmt not in _rate_formats:
        raise ValueError(f'rate format must be in {_rate_formats}')
    return _rate_formats[rate_fmt]



def years_between(t0, t1):
    """
    whole years between two dates

    Example
    -------
    >>> t0 = dt(2000,1,1)
    >>> t1 = dt(2001,1,1)
    >>> assert years_between(t0,t1) == 1
    >>> assert years_between(dt(2000, 2, 1), t1) == 0

    
    """
    y = t1.year - t0.year
    if dt(t0.year + y, t0.month, t0.day) > t1:
        return y-1
    else:
        return y

def years_to_maturity(maturity, ts = None):
    """
    calculates years to maturity with part of the year calculated as ACT/365
    
    :Example:
    ---------
    >>> from pyg import *     
    >>> ts = pd.Series(range(1000), drange(-999))
    >>> maturity = dt('2Y')
    >>> bond_years_to_maturity(ts, maturity)
    
    If ts is not provided, we assume currently that maturity is given as an integer
    
    """
    if not is_date(maturity):
        return maturity
    if is_date(ts):
        y = years_between(ts, maturity)
        frac = (dt(maturity, f'-{y}y') - ts).days / 365.
        return y + frac
    elif isinstance(ts, list):
        return [years_to_maturity(maturity, t) for t in ts]             
    elif is_ts(ts):
        if len(ts) == 0:
            return ts
        t0 = ts.index[0]
        years = list(range(2+maturity.year-t0.year))[::-1]
        dates = [dt(maturity, f'-{y}y') for y in years]
        y = df_reindex(pd.Series(years, dates), ts, method = 'bfill')
        days = df_reindex(pd.Series(dates, dates), ts, method = 'bfill')
        frac = pd.Series((days.values - days.index).days / 365, ts.index)
        return y + frac
    else:
        raise ValueError(f'cannot calculate years_to_maturity for {maturity} and {ts}')
