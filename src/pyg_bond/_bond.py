import numpy as np
import pandas as pd
import datetime
from pyg_base import is_num, pd2np

__all__ = ['aus_bill_pv', 'bond_pv', 'bond_yld', 'bond_duration', 'aus_bond_pv', 'bond_yld_and_duration']

def aus_bill_pv(quote, tenor = 90, facevalue = 100, daycount = 365):
    """
	converts ausralian accepted bank bills quote as (100-yld) into a price.
    See https://www.asx.com.au/documents/products/ird-pricing-guide.pdf for full implementaion
    """
    yld = 1 - quote/100
    discount_factor = 1/(1 + tenor * yld / daycount)
    pv = facevalue * discount_factor
    return pv 

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

def bond_pv(yld, tenor, coupon = 0.06, freq = 2):
    pv, duration = _bond_pv_and_duration(yld, tenor, coupon = coupon, freq = freq)
    return pv

bond_pv.__doc__ = _bond_pv_and_duration.__doc__

def _tenor(tenor, yld):
    if isinstance(tenor, datetime.datetime) and isinstance(yld, (pd.Series, pd.DataFrame)) and isinstance(yld.index, pd.DatetimeIndex):
        return pd.Series((tenor - yld.index).days/365.25, yld.index)
    else:
        return tenor
        
def bond_duration(yld, tenor, coupon = 0.06, freq = 2):
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

    Returns
    -------
    duration: number/array
        the duration of the bond
    """
    pv, duration = _bond_pv_and_duration(yld, _tenor(tenor,yld), coupon = coupon, freq = freq)
    return duration

bond_duration.__doc__ = _bond_pv_and_duration.__doc__


def aus_bond_pv(quote, tenor, coupon = 0.06, freq = 2, facevalue = 100):
    """
    
    Australian bond futures are quoted as 100-yield. Here we calculate their actual price. See:
    https://www.asx.com.au/documents/products/ird-pricing-guide.pdf
    
    :Parameters:
    ------------
    quote: float/timeseries
        quote of aus bond future
    
    tenor: int
        usually 3 or 10-year bonds
    
    coupon: float
        bond future coupon, default is 6%

    freq: int
        aussie bonds pay twice a year        
    
    :Examples:
    -----------
    >>> assert aus_bond_pv(100, 10) == 160 # yld = 0 so price is notional + 10 6% coupons
    >>> assert abs(aus_bond_pv(98, 10, coupon = 0.02) - 100)<1e-6 # Here yield and coupon the same

    >>> quote = 95.505
    >>> assert round(aus_bond_pv(quote, 3),5) ==  104.18009    
    >>> assert round(aus_bond_pv(95.500, 10),5) == 111.97278

    
    """
    yld = 1 - quote / 100
    return facevalue * bond_pv(yld, tenor = tenor, coupon = coupon, freq = freq)

def bond_yld_and_duration(price, tenor, coupon = 0.06, freq = 2, iters = 5):
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

bond_yld_and_duration.output = ['yld', 'duration']

def bond_yld(price, tenor, coupon = 0.06, freq = 2, iters = 5):
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

    Returns
    -------
    yld : number/array
        the yield of the bond
    """
    return bond_yld_and_duration(price, tenor, coupon = coupon, freq = freq, iters = iters)['yld']


def bond_par_conversion_factor(yld, tenor, coupon = 0.06, freq = 2, invert = False):
    """
    This is an approximation, calculating the conversion factor for a par bond.
    We are given a yield curve (yld) and we construct a par bond. 
    The conversion factor is given by the value of the bond if interest rates were at 6% (coupon)
    :Parameters:
    ------------
    yld: float/array/DataFrame
        The yield of a bond
    tenor: int
        The maturity of the bond
        
    
    :Example: simple calculation
    ----------------------------
    >>> import numpy as np; import pandas as pd
    >>> from pyg_base import * 
    >>> from pyg_bond import *

    >>> yld = 0.023
    >>> tenor = 7
    >>> coupon = 0.06
    >>> freq = 2
    
    :Example: working with a pandas object
    --------------------------------------
    
    >>> yld = pd.Series(np.random.uniform(0,0.1,100), drange(-99))
    >>> bond_par_conversion_factor(yld, 10)

    :Example: working with a pandas objects with multiple expiries
    -------------------------------------------------------------
    >>> tenors = [7,8,9,10,11]
    >>> yld_curve = pd.DataFrame(np.random.uniform(0,0.1,(100,5)), drange(-99), tenors)
    >>> res = loop(pd.DataFrame)(bond_par_conversion_factor)(yld_curve, tenors)
 
    :Example: the cheapest to deliver tenor for a flat yield curve:
    --------------------------------------------------------------
    >>> tenor =  [7,8,9,10,11]
    >>> yld = [0.02] * 5
    >>> loop(list)(bond_par_conversion_factor)(yld, tenor, invert = True)
    >>> [1.2918585801399018, 1.3355093954613233, 1.3794440302478643, 1.423587848577121,1.4678647692438134]
    >>> print('if yields are lower than 6% cheapest to deliver is the 7-year par bond')

    >>> yld = [0.08] * 5
    >>> loop(list)(bond_par_conversion_factor)(yld, tenor, invert = True)
    >>> [0.8985042974046984, 0.8884063695192392, 0.8790937290011396, 0.8704926716179342, 0.862538032755245]
    >>> print('if yields are higher than 6% cheapest to deliver is the 11-year par bond')
    """
    res = bond_pv(coupon, tenor, coupon = yld, freq = freq)
    return 1/res if invert else res

def bond_ctd(tenor2yld, coupon = 0.06, freq = 2):
    """
    returns yld, tenor and future price of a CTD future with multiple yields
    
    Parameters
    ----------
    tenor2yld : dict
        mapping from tenor in years (int) to yield timeseries
    coupon : float, optional
        The coupon for the future. The default is 0.06.
    freq : int, optional
        payment frequency per year. The default is 2.

    Returns
    -------
    pd.DataFrame 
        with three columns: tenor, yld, price

    :Example:
    ---------
    >>> y7 = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.071], drange(-7))
    >>> y10 = pd.Series([0.02, 0.03, 0.04, 0.05, 0.06, 0.07], drange(-6,-1))
    >>> tenor2yld = {7 : y7, 10: y10}

    >>>              yld  tenor     price
    >>> 2022-01-11  0.01      7  1.393538
    >>> 2022-01-12  0.02      7  1.291859
    >>> 2022-01-13  0.03      7  1.204009
    >>> 2022-01-14  0.04      7  1.127346
    >>> 2022-01-15  0.05      7  1.059861
    >>> 2022-01-16  0.06     10  1.000000
    >>> 2022-01-17  0.07     10  0.930763
    >>> 2022-01-18  0.07     10  0.930763

    """
    tenors = list(tenor2yld.keys())
    ylds = pd.concat(tenor2yld.values(), axis = 1).ffill()
    cfs = [bond_par_conversion_factor(yld, tenor, coupon, freq, invert = True) for tenor, yld in tenor2yld.items()]
    df = pd.concat(cfs, axis=1).ffill()
    m = df.min(axis=1).values
    mask = df.values == np.array([m]*len(cfs)).T
    t = np.array([tenors] * len(df)) * mask
    
    ylds[~mask] = np.nan
    df[~mask] = np.nan
    yld = ylds.mean(axis=1)
    tenor = pd.Series(np.amax(t, axis = 1), df.index)
    res = dict(yld = yld, tenor = tenor, price = df.mean(axis=1))
    return pd.DataFrame(res)
    
    
    