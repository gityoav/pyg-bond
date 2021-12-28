import numpy as np
import pandas as pd

__all__ = ['aus_bill_pv', 'bond_pv', 'bond_yld', 'bond_duration', 'aus_bond_pv', 'bond_yld_and_duration']

def is_num(value):
	return isinstance(value, (int, np.int64, np.int32, np.int16, np.int8, float, np.float16, np.float32, np.float64))

def aus_bill_pv(quote, tenor = 90, facevalue = 100, daycount = 365):
    """
	converts ausralian accepted bank bills quote of (100-yld) into a price
    See https://www.asx.com.au/documents/products/ird-pricing-guide.pdf
    """
    yld = 1 - quote/100
    discount_factor = 1/(1 + tenor * yld / daycount)
    pv = facevalue * discount_factor
    return pv 


def _bond_pv_and_duration(yld, tenor, coupon = 0.05, freq = 2):
    """
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
        return 1 + n * c
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
        pv[mask] = 1 + n * c
        duration[mask] = tenor + c*n*(n+1)/(2*freq)
    return pv, duration

def bond_pv(yld, tenor, coupon = 0.06, freq = 2):
    pv, duration = _bond_pv_and_duration(yld, tenor, coupon = coupon, freq = freq)
    return pv

bond_pv.__doc__ = _bond_pv_and_duration.__doc__

def bond_duration(yld, tenor, coupon = 0.06, freq = 2):
    pv, duration = _bond_pv_and_duration(yld, tenor, coupon = coupon, freq = freq)
    return duration

bond_duration.__doc__ = _bond_pv_and_duration.__doc__


def aus_bond_pv(quote, tenor, coupon = 0.06, freq = 2, facevalue = 100):
    """
    https://www.asx.com.au/documents/products/ird-pricing-guide.pdf
    
    assert aus_bond_pv(100, 10) == 100
    assert aus_bond_pv(98, 10) == 1.6e5
    quote = 95.505
    assert round(aus_bond_pv(quote, 3),2) ==  104180.09
    
    assert round(aus_bond_pv(95.500, 10),2) == 111972.78

    
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
    yld : number/array
        the yield of the bond
    """
    return bond_yld_and_duration(price, tenor, coupon = coupon, freq = freq, iters = iters)['yld']

