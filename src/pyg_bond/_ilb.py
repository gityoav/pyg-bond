# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pyg_bond._base import rate_format
from pyg_base import dt, drange, ts_gap, years_to_maturity, df_reindex, mul_, add_, pd2np, is_num, loop, is_ts, is_arr, calendar, df_sync
from pyg_timeseries import shift, diff


def cpi_reindexed(cpi, ts):
    """
    august cpi is published in september and that return materializes over october to november
    interestingly, this future growth is KNOWN so potentially, future back adjusted calculation is possible
    """
    if is_ts(cpi):
        cpi_eom = cpi.resample('m').last()
        dates = [dt(dt(eom, 1), '2m') for eom in cpi_eom.index]
        if isinstance(cpi, pd.DataFrame):
            res = pd.DataFrame(cpi_eom.values, index = dates, columns = cpi_eom.columns)
        else:
            res = pd.Series(cpi_eom.values, index = dates)
        t0 = ts.index[0]
        t1 = ts.index[-1]
        extended_dates = drange(dt(t0.year, t0.month, 0), dt(t1.year, t1.month+1,0))
        rtn = df_reindex(df_reindex(res, extended_dates, method = 'linear'), ts)
        return rtn
    else:
        return cpi

def ilb_ratio(cpi, base_cpi = 1, floor = 1):
    ratio = cpi/base_cpi
    if floor:
        ratio = np.maximum(floor, ratio)
    return ratio
    
def ilb_total_return(price, coupon, funding, base_cpi, cpi, floor = 1, rate_fmt = 100, 
                     freq = 2, dirty_correction = True):
    """
    inflation linked bond clean price is quoted prior to notional multiplication and accrual
    
    So:
        notional = cpi / base_cpi
        carry = daily_accrual - daily_funding
        MTM = notional * dirty price
        change(dirty_price) = change(clean_price) + carry

    Using the product rule:
        
        change(MTM) = change(notional * clean_price) + notional * carry + change(notional) * (dirty-clean)

    We actually approximate it a little... as
        change(MTM) = change(notional * clean_price) + notional * carry + change(notional) * AVG(dirty-clean)
    since
        AVG(dirty-clean) = 0.5 * (coupon / freq) (it grows from 0 to coupon/freq before dropping back to 0)
    
    :Example:
    ---------
    >>> from pyg import * 
    >>> coupon = 3
    >>> funding = 1
    >>> price = pd.Series([80, 80, np.nan] * 87 + [80], drange(2001,2002,'1b')) ##  
    >>> rate_fmt = 100    
    >>> base_cpi = floor = 1
    >>> cpi = pd.Series(np.arange(1,2+1/261,1./261), drange(2001,2002,'1b'))
    >>> tri = ilb_total_return(price, coupon, funding, base_cpi, cpi, floor = 1, rate_fmt = 100, dirty_correction = False)
    
    The total return in MTM is due to notional doubling due to inflation. 
    Price remain constant so MTM going up from 80 to 160.
    
    Carry should be almost exactly: (3 - 0.8) * cpi.mean() == 3.3 ## 3% less funding of 80 at 1%
    
    >>>> assert (tri.sum() - 83.3)<1e-2
    
    
    """
    rate_fmt = rate_format(rate_fmt)
    mask = np.isnan(price)
    prc = price[~mask]
    dcf = ts_gap(prc)/365 ## day count fraction, forward looking
    notional = df_reindex(cpi / base_cpi, price, method = 'ffill')
    finance = (prc/100) * df_reindex(funding, prc, method = ['ffill', 'bfill'])
    notional[mask] = np.nan
    if floor:
        notional = np.maximum(floor, notional)
    carry = df_reindex(shift(mul_([coupon - finance, dcf, notional])), price) ## ## accruals less funding costs on notional
    pv = mul_(price, notional)
    rtn = diff(pv)
    if dirty_correction:
        dirty_change_in_notional = diff(notional) * (coupon / (2 * freq))
        return add_([rtn, (100/rate_fmt) * carry, dirty_change_in_notional])
    else:
        return add_([rtn, (100/rate_fmt) * carry])
    

@pd2np
def _ilb_pv_and_durations(nominal_yld, cpi_yld, tenor, coupon, freq = 2):
    """
    
    Given 
    - yld by which we discount all cash flows,
    - cpi_yld: the growth rate of cpi
    and the usual tenor, coupon, freq defining the cash flows,
    can we determine the pv of an ilb and its derivative wrt both yld and cpi_yld
    

    :Present Value calculation:
    --------------------------
    
    There are n = freq * tenor periods
    and a period discount factor, i.e.   

    d = (1 + yld/freq) [so that paying a coupon of y/freq at end of period, would keep value constant at 1]

    On the other hand, there is growth factor g = (1 + cpi_yld/freq) since we get paid based on growth of cpi

    g = (1+cpi_yld/freq)

    Let f = g / d

    and let r = 1/(1-f)

    just like a normal bond:
        
    coupons_pv = c f + c * f^2 + ... c * f ^ (freq * tenor)  
               = c f * (1+f...+f^(n-1)) 
               = c f * (1 - f^n) / (1 - f)  = c * f * (1-f^n) * r
    notional_pv = f^n
    
    if yld == cpi_yld and f == 1 then...
    pv = 1 + c * n # n coupons + notional
    
    :duration calculation:
    --------------------------
    we denote p = cpi_yld
    df/dy = - 1/freq * g/d^2 = - f^2 / (freq * g)
    df/dp = = 1/(freq * d) = f / (freq * g) 
    
    dr/dy = r^2 df/dy
    dr/dp = r^2 df/dp
    
    
    yield duration
    ---------------
    - dnotional/dy =  n f ^ (n-1) df/dy 
    - dcoupons/dy = c * df/dy * [(1-f^n)*r - f * n f^n-1 *r + f * (1-f^n) * r^2]  # using the product rule
                  = c * df/dy * r [(1-f^n) - n * f^n + f(1-f^n)*r]    

    if yld == cpi_yld and f == 1 then..
    
    dnotional_dy = tenor
    coupons_pv = c f + c * f^2 + ... c * f ^ (freq * tenor)  = c * f * (1+f...+f^(n-1)) 
    dcoupon_dy/c = df/dy ( 1 + 2f + 3 f^2 ... + nf^(n-1)) 
                 = df/fy (1+...n) # since f = 1
                 = (1/g * freq) n(n+1)/2

    cpi duration
    ------------
    The formula is identical, except we replace df/dy with df/dp so we just need to divide by -f
    
    
    Example: ilb calculations match normal bond when cpi_yld = 0
    ---------
    >>> tenor = 10; coupon = 0.02; yld = 0.05; cpi_yld = 0.03; freq = 2
    
    >>> _ilb_pv_and_durations(yld = yld, cpi_yld = 0.00, tenor = tenor, coupon = coupon, freq = freq)
    >>> (0.7661625657152991, 6.857403925710587, 6.690150171424962)
    
    >>> _bond_pv_and_duration(yld = yld, tenor = tenor, coupon = coupon, freq = freq)
    >>> (0.7661625657152991, 6.690150171424962)

    Example: ilb calculated duration is same as empirical one
    ---------
    >>> pv3, cpi3, yld3 = _ilb_pv_and_durations(yld = yld, cpi_yld = 0.03, tenor = tenor, coupon = coupon, freq = freq)
    >>> pv301, cpi301, yld301 = _ilb_pv_and_durations(yld = yld, cpi_yld = 0.0301, tenor = tenor, coupon = coupon, freq = freq)
    >>> 1e4 * (pv301 - pv3), 0.5*(cpi301 + cpi3)


    """
    n = tenor * freq
    c = coupon / freq
    if is_arr(nominal_yld):
        nominal_yld[nominal_yld<=-freq] = np.nan
    elif nominal_yld<=-freq:
        nominal_yld = np.nan
    if is_arr(cpi_yld):
        cpi_yld[cpi_yld<=-freq] = np.nan
    elif cpi_yld<=-freq:
        cpi_yld= np.nan
    d = (1 + nominal_yld / freq)
    g = (1 + cpi_yld / freq)
    if is_num(nominal_yld) and is_num(cpi_yld) and nominal_yld == cpi_yld:        
        pv = 1 + n * c
        yld_duration = n * (n + 1) / (2 * freq * g)
        cpi_duration = yld_duration
    f = g / d
    dfy = f**2 / (g * freq) ## we ignore the negative sign
    dfp = f / (g * freq)
    fn1 = f ** (n-1)    
    r = 1 / (1 - f)
    notional_pv = fn = fn1 * f
    dnotional_dy = n * fn1 * dfy
    dnotional_dp = n * fn1 * dfp
    coupon_pv = c * f * (1 - fn) * r
    pv = notional_pv + coupon_pv
    dcoupon_dy = c * dfy * r * ((1 - fn)  - n * fn  + f * (1-fn) * r)
    dcoupon_dp = c * dfp * r * ((1 - fn)  - n * fn  + f * (1-fn) * r)
    yld_duration = dnotional_dy + dcoupon_dy
    cpi_duration = dnotional_dp + dcoupon_dp
    if isinstance(nominal_yld, (pd.Series, pd.DataFrame, np.ndarray)):
        mask = f == 1
        pv0 = 1 + n * c
        duration0 = tenor + c*n*(n+1)/(2*freq*g)
        pv[mask] = pv0 if is_num(pv0) else pv0[mask]
        yld_duration[mask] = duration0 if is_num(duration0) else duration0[mask]
        cpi_duration[mask] = duration0 if is_num(duration0) else duration0[mask]
    return pv, cpi_duration, yld_duration


def ilb_pv(nominal_yld, cpi_yld, tenor, coupon, freq = 2, rate_fmt = None):
    """
    Given 
    - nominal_yld by which we discount all cash flows,
    - cpi_yld: the growth rate of cpi
    
    and the usual tenor, coupon, freq defining the cash flows,
    can we determine the pv of an ilb and its derivative wrt both yld and cpi_yld
    
    
    Example:
    --------
    cpi_yld = ilb_cpi_yld(100, )
        

    :Present Value calculation:
    --------------------------
    
    There are n = freq * tenor periods
    and a period discount factor, i.e.   

    d = (1 + nominal_yld/freq) [so that paying a coupon of y/freq at end of period, would keep value constant at 1]

    On the other hand, there is growth factor g = (1 + cpi_yld/freq) since we get paid based on growth of cpi

    g = (1+cpi_yld/freq)

    Let f = g / d

    and let r = 1/(1-f)

    just like a normal bond:
        
    coupons_pv = c f + c * f^2 + ... c * f ^ (freq * tenor)  
               = c f * (1+f...+f^(n-1)) 
               = c f * (1 - f^n) / (1 - f)  = c * f * (1-f^n) * r
    notional_pv = f^n
    
    if nominal_yld == cpi_yld and f == 1 then...
    pv = 1 + c * n # n coupons + notional
    
    :duration calculation:
    ----------------------
    we denote p = cpi_yld and y = yld = nominal_yld
    
    f = g/d = (1+p/freq)/(1+y/freq)
    df/dy = - 1/freq * g/d^2 = - f^2 / (freq * g)
    df/dp =   1/(freq * d) = f / (freq * g) 
    
    dr/dy = r^2 df/dy
    dr/dp = r^2 df/dp
    
    nominal yield duration
    ---------------
    - dnotional/dy =  n f ^ (n-1) df/dy 
    - dcoupons/dy = c * df/dy * [(1-f^n)*r - f * n f^n-1 *r + f * (1-f^n) * r^2]  # using the product rule
                  = c * df/dy * r [(1-f^n) - n * f^n + f(1-f^n)*r]    

    if yld == cpi_yld and f == 1 then..
    
    dnotional_dy = tenor
    coupons_pv = c f + c * f^2 + ... c * f ^ (freq * tenor)  = c * f * (1+f...+f^(n-1)) 
    dcoupon_dy/c = df/dy ( 1 + 2f + 3 f^2 ... + nf^(n-1)) 
                 = df/fy (1+...n) # since f = 1
                 = (1/g * freq) n(n+1)/2

    cpi duration
    ------------
    The formula is identical, except we replace df/dy with df/dp so we just need to divide by -f
    
    Example: ilb calculations match normal bond when cpi_yld = 0
    ---------
    >>> tenor = 10; coupon = 0.02; yld = 0.05; cpi_yld = 0.03; freq = 2
    
    >>> _ilb_pv_and_durations(yld = yld, cpi_yld = 0.00, tenor = tenor, coupon = coupon, freq = freq)
    >>> (0.7661625657152991, 6.857403925710587, 6.690150171424962)
    
    >>> _bond_pv_and_duration(yld = yld, tenor = tenor, coupon = coupon, freq = freq)
    >>> (0.7661625657152991, 6.690150171424962)

    Example: ilb calculated duration is same as empirical one
    ---------
    >>> pv3, cpi3, yld3 = _ilb_pv_and_durations(yld = yld, cpi_yld = 0.03, tenor = tenor, coupon = coupon, freq = freq)
    >>> pv301, cpi301, yld301 = _ilb_pv_and_durations(yld = yld, cpi_yld = 0.0301, tenor = tenor, coupon = coupon, freq = freq)
    >>> 1e4 * (pv301 - pv3), 0.5*(cpi301 + cpi3)


    """
    rate_fmt = rate_format(rate_fmt)
    nominal_yld, cpi_yld = df_sync([nominal_yld, cpi_yld])
    if rate_fmt!=1:
        nominal_yld, cpi_yld, coupon = nominal_yld/rate_fmt, cpi_yld/rate_fmt, coupon/rate_fmt 
    tenor = years_to_maturity(tenor, cpi_yld)
    pv, cpi_duration, yld_duration = _ilb_pv_and_durations(nominal_yld, cpi_yld, tenor = tenor, coupon = coupon, freq = freq)
    px = pv * 100
    return px
    
def ilb_yld_duration(nominal_yld, cpi_yld, tenor, coupon, freq = 2, rate_fmt = None):
    rate_fmt = rate_format(rate_fmt)
    nominal_yld, cpi_yld = df_sync([nominal_yld, cpi_yld])
    if rate_fmt!=1:
        nominal_yld, cpi_yld, coupon = nominal_yld/rate_fmt, cpi_yld/rate_fmt, coupon/rate_fmt 
    tenor = years_to_maturity(tenor, cpi_yld)
    pv, cpi_duration, yld_duration = _ilb_pv_and_durations(nominal_yld, cpi_yld, tenor = tenor, coupon = coupon, freq = freq)
    return yld_duration
    

def ilb_cpi_duration(nominal_yld, cpi_yld, tenor, coupon, freq = 2, rate_fmt = None):
    rate_fmt = rate_format(rate_fmt)
    nominal_yld, cpi_yld = df_sync([nominal_yld, cpi_yld])
    if rate_fmt!=1:
        nominal_yld, cpi_yld, coupon = nominal_yld/rate_fmt, cpi_yld/rate_fmt, coupon/rate_fmt 
    tenor = years_to_maturity(tenor, cpi_yld)
    pv, cpi_duration, yld_duration = _ilb_pv_and_durations(nominal_yld, cpi_yld, tenor = tenor, coupon = coupon, freq = freq)
    return cpi_duration


def _ilb_cpi_yld_and_duration(price, nominal_yld, tenor, coupon, freq = 2, iters = 5):
    """
	
    We calculate break-even yield for a bond, given its price, the yield of a normal government bond and tenor and coupons...	
    We expect price to be quoted as per usual in market, i.e. 100 being par value. However, coupon and yield should be in fed actual values.

    Parameters
    ----------
    price : float/array
        clean price of an inflation linked bond
    nominal_yld: float/array
        The yield of a vanilla government bond, used as a reference for discounting cash flows
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
    px = price /100
    cpi_yld = 0
    for _ in range(1+iters):
        pv, cpi_duration, yld_duration = _ilb_pv_and_durations(nominal_yld, cpi_yld, tenor, coupon, freq = freq)
        cpi_yld = cpi_yld + (px - pv) / cpi_duration
    return dict(cpi_yld = cpi_yld, cpi_duration = cpi_duration, yld_duration = yld_duration)

_ilb_cpi_yld_and_duration.output = ['cpi_yld', 'cpi_duration', 'yld_duration']

_ilb_cpi_yld_and_duration_ = loop(pd.DataFrame, pd.Series)(_ilb_cpi_yld_and_duration)


def ilb_cpi_yld_and_duration(price, nominal_yld, tenor, coupon, freq = 2, iters = 5, rate_fmt = None):
    """
    calculates both cpi_yield and cpi_duration from a maturity date or a tenor.
    cpi_yld is the breakeven yield inflation that matches the prices with vanilla bond.

    Parameters
    ----------
    price : float/array
        price of bond
    nominal_yld: float/array
        yield of a NOMINAL bond with similar maturity
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
    res : dict
        cpi_yld and cpi_duration.
        
    Example:
    --------
    >>> cpi_yld = ilb_cpi_yld(84, nominal_yld = 0.04, tenor = 10, coupon = 0.01, cpi = 1.3, base_cpi = 1)
    >>> px = ilb_pv(nominal_yld = 0.04, cpi_yld = res['cpi_yld'], tenor = 10, coupon = 0.01)
    >>> assert abs(px-84)<1e-6
    """
    rate_fmt = rate_format(rate_fmt)
    price, nominal_yld = df_sync([price, nominal_yld])
    tenor = years_to_maturity(tenor, price)
    if rate_fmt == 1:        
        return _ilb_cpi_yld_and_duration_(price, nominal_yld, tenor, coupon, freq = freq, iters = iters)
    else:
        res = _ilb_cpi_yld_and_duration_(price = price, 
                                        nominal_yld = nominal_yld/rate_fmt, tenor = tenor, coupon = coupon/rate_fmt, 
                                        freq = freq, iters = iters)
        res['cpi_yld'] *= rate_fmt
        return res

ilb_cpi_yld_and_duration.output = _ilb_cpi_yld_and_duration.output 


def ilb_cpi_yld(price, nominal_yld, tenor, coupon, freq = 2, iters = 5, rate_fmt = None):
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

    return ilb_cpi_yld_and_duration(price = price, nominal_yld = nominal_yld, 
                                    tenor = tenor, coupon = coupon, freq = freq, iters = iters, 
                                    rate_fmt = rate_fmt)['cpi_yld']




