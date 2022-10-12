import numpy as np
import pandas as pd
import datetime
from pyg_base import nona, is_num, pd2np, dt, df_reindex, loop, is_date, is_ts, ts_gap, mul_, add_
from pyg_timeseries import shift, diff
from pyg_bond._base import rate_format, years_to_maturity

__all__ = ['bond_pv', 'bond_yld', 'bond_duration', 'bond_yld_and_duration']

@pd2np
def _bond_pv_and_duration(yld, tenor, coupon = 0.06, freq = 2):
    """
    
    Given yield and cash flows (coupon, tenor and freq), we calculate pv and duration.
    We expects the yield and the coupons to be quoted as actual values rather than in percentages

    :Present Value calculation:
    --------------------------
    
    There are n = freq * tenor periods
    and a period discount factor f is (1 + y/freq) i.e.   
    f = 1/(1 + y/freq) [so that paying a coupon of y/freq at end of period, would keep value constant at 1]
    r = 1/(1-f)
    so...
    
    coupons_pv = c f + c * f^2 + ... c * f ^ (freq * tenor)  = c * f * (1+f...+f^(n-1)) = c * f * (1 - f^n) / (1 - f)  = c * f * (1-f^n) * r
    notional_pv = f^n
    
    if yld == 0 and df == 1 then...
    pv = 1 + c * n # n coupons + notional
    
    :Duration calculation:
    --------------------------
    df/dy = - (1 + y/freq)^-2 * 1/freq = f^2 / freq
    dr/dy = r^2 df/dy
    
    - dnotional/dy =  n f ^ (n-1) df/dy 
    - dcoupons/dy = c *  df/dy * [(1-f^n)*r - f * n f^n-1 *r + f * (1-f^n) * r^2]  # using the product rule
                  = c * df/dy * r [(1-f^n) - n * f^n + f(1-f^n)*r]    

    if yld == 0 and f == 1 then..
    
    dnotional_dy = tenor
    coupons_pv = c f + c * f^2 + ... c * f ^ (freq * tenor)  = c * f * (1+f...+f^(n-1)) = c * f * (1 - f^n) / (1 - f)  = c * f * (1-f^n) * r
    dcoupon_dy/c = df/dy ( 1 + 2f + 3 f^2 ... + nf^(n-1)) 
                 = 1/freq * (1 + 2 +... n) ## since f = 1
                 = n(n+1)/(2 * freq)
                 
    """
    n = tenor * freq
    c = coupon / freq
    if is_num(yld) and yld == 0:
        pv = 1 + n * c
        duration = tenor + c*n*(n+1)/(2*freq)
        return pv, duration
    f = 1/(1 + yld/freq)
    dfy = f**2 / freq ## we ignore the negative sign
    fn1 = f ** (n-1)    
    r = 1 / (1 - f)
    notional_pv = fn = fn1 * f
    dnotional_dy = n * fn1 * dfy
    coupon_pv = c * f * (1 - fn) * r
    pv = notional_pv + coupon_pv
    dcoupon_dy = c * dfy * r * ((1 - fn)  - n * fn  + f * (1-fn) * r)
    duration =  dnotional_dy + dcoupon_dy
    if isinstance(yld, (pd.Series, pd.DataFrame, np.ndarray)):
        mask = yld == 0
        pv0 = 1 + n * c
        duration0 = tenor + c*n*(n+1)/(2*freq)
        pv[mask] = pv0 if is_num(pv0) else pv0[mask]
        duration[mask] = duration0 if is_num(duration0) else duration0[mask]
    return pv, duration

def bond_pv(yld, tenor, coupon = 0.06, freq = 2, rate_fmt = None):
    """
    
    Calculates the bond present value given yield and coupon.
    Returns par value as 1.
    
    :Example:
    ---------
    >>> assert abs(bond_pv(yld = 0.06, tenor = 10, coupon = 0.06, freq = 2) - 1) < 1e-6

    Parameters
    ----------
    yld : float
        yield in market.
    tenor : int
        maturity of bond, e.g. tenor = 10 for a 10-year bond.
    coupon : float, optional
        Bond coupon. The default is 0.06.
    freq : int, optional
        number of coupon payments in a year. The default is 2.
    rate_fmt : int, optional
        is coupon/yield data provided as actual or as a %. The default is None, actual

    Returns
    -------
    pv : float
        Bond present value.

    """
    rate_fmt = rate_format(rate_fmt)
    pv, duration = _bond_pv_and_duration(yld / rate_fmt, tenor, coupon = coupon / rate_fmt, freq = freq)
    return pv

bond_pv.__doc__ = _bond_pv_and_duration.__doc__
        
def bond_duration(yld, tenor, coupon = 0.06, freq = 2, rate_fmt = None):
    """
	
	bond_duration calculates duration (sensitivity to yield change).
	
    Parameters
    ----------
    yld: float/array
        yield of bond
    tenor : int
        tenor of a bond.
    coupon : float, optional
        coupon of a bond. The default is 0.06.
    freq : int, optional
        number of coupon payments per year. The default is 2.
    rate_fmt: int
        how is coupon/yield quoted. 1 = actual (e.g. 0.06) while 100 is market convention (6 represents 6%)

    Returns
    -------
    duration: number/array
        the duration of the bond
    """
    rate_fmt = rate_format(rate_fmt)
    tenor = years_to_maturity(tenor, yld)
    pv, duration = _bond_pv_and_duration(yld/rate_fmt, tenor = tenor, coupon = coupon/rate_fmt, freq = freq)    
    return duration

bond_duration.__doc__ = _bond_pv_and_duration.__doc__


    
def _bond_yld_and_duration(price, tenor, coupon = 0.06, freq = 2, iters = 5):
    """
	
	bond_yld_and_duration calculates yield from price iteratively using Newton Raphson gradient descent.
	
    We expect price to be quoted as per usual in market, i.e. 100 being par value. However, coupon and yield should be in fed actual values.

    Parameters
    ----------
    price : float/array
        price of bond
    tenor : int
        tenor of a bond.
    coupon : float, optional
        coupon of a bond. The default is 0.06.
    freq : int, optional
        number of coupon payments per year. The default is 2.
    iters : int, optional
        Number of iterations to find yield. The default is 5.

    Returns
    -------
	returns a dict of the following keys:
	
    yld : number/array
        the yield of the bond
	duration: number/array 
		the duration of the bond. Note that this is POSITIVE even though the dPrice/dYield is negative
    """
    px = price/100
    yld = ((1+tenor*coupon) - px)/tenor
    for _ in range(iters):
        pv, duration = _bond_pv_and_duration(yld, tenor, coupon = coupon, freq = freq)
        yld = yld + (pv - px) / duration
    return dict(yld = yld, duration = duration)

_bond_yld_and_duration.output = ['yld', 'duration']


_bond_yld_and_duration_ = loop(pd.DataFrame, pd.Series)(_bond_yld_and_duration)


def bond_yld_and_duration(price, tenor, coupon, freq = 2, iters = 5, rate_fmt = None):
    """
    calculates both yield and duration from a maturity date or a tenor

    Parameters
    ----------
    price : float/array
        price of bond
    tenor: int, date, array
        if a date, will calculate 
    coupon : float, optional
        coupon of a bond. The default is 0.06.
    freq : int, optional
        number of coupon payments per year. The default is 2.
    iters : int, optional
        Number of iterations to find yield. The default is 5.

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    """
    rate_fmt = rate_format(rate_fmt)
    tenor = years_to_maturity(tenor, price)
    if rate_fmt == 1:        
        return _bond_yld_and_duration_(price, tenor = tenor, coupon = coupon, freq = freq, iters = iters)
    else:
        res = _bond_yld_and_duration_(price, tenor = tenor, coupon = coupon/rate_fmt, freq = freq, iters = iters)
        res['yld'] *= rate_fmt
        return res


bond_yld_and_duration.output = _bond_yld_and_duration.output
    

def bond_yld(price, tenor, coupon = None, freq = 2, iters = 5, rate_fmt = None):
    """
	
	bond_yld calculates yield from price iteratively using Newton Raphson gradient descent.
	
    We expect price to be quoted as per usual in market, i.e. 100 being par value. However, coupon and yield should be in fed actual values.

    Parameters
    ----------
    price : float/array
        price of bond
    tenor : int
        tenor of a bond.
    coupon : float, optional
        coupon of a bond. The default is 0.06.
    freq : int, optional
        number of coupon payments per year. The default is 2.
    iters : int, optional
        Number of iterations to find yield. The default is 5.
    rate_fmt: how you prefer to quote rates: 1 = 6% is represented as 0.06, 100 = 6% is represented as 6.

    Returns
    -------
    yld : number/array
        the yield of the bond
    """

    rate_fmt = rate_format(rate_fmt)
    if coupon is None:
        coupon = 0.06 * rate_fmt 
    return bond_yld_and_duration(price, tenor, coupon = coupon, freq = freq, iters = iters, rate_fmt = rate_fmt)['yld']


    
def bond_total_return(price, coupon, funding, rate_fmt = 100):
    """
    price = pd.Series([1,np.nan,np.nan,2], [dt(-100),dt(-99),dt(-88), dt(0)])
    """
    rate_fmt = rate_format(rate_fmt)
    prc = nona(price)
    dcf = ts_gap(prc)/365. ## day count fraction, forward looking
    funding = df_reindex(funding, prc, method = ['ffil', 'bfill'])
    carry = df_reindex(shift(mul_(coupon - funding, dcf)), price) ## accruals less funding costs
    rtn = diff(price)
    return add_([rtn, (100/rate_fmt) * carry])

