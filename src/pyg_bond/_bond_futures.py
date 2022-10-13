from pyg_bond._bond import bond_pv
from pyg_bond._base import rate_format
from pyg_base import cache
import pandas as pd
import numpy as np

__all__ = ['aus_bill_pv', 'aus_bond_pv']

def aus_bill_pv(quote, tenor = 90, facevalue = 100, daycount = 365):
    """
	converts ausralian accepted bank bills quote as (100-yld) into a price.
    See https://www.asx.com.au/documents/products/ird-pricing-guide.pdf for full implementaion
    """
    yld = 1 - quote/100
    discount_factor = 1/(1 + tenor * yld / daycount)
    pv = facevalue * discount_factor
    return pv 



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
    return facevalue * bond_pv(yld, tenor = tenor, coupon = coupon, freq = freq, rate_fmt = 1)



def bond_par_conversion_factor(yld, tenor, coupon = None, freq = 2, invert = False, rate_fmt = 1):
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
    coupon: float
        bond coupon. defaults to 6%
    freq: int
        number of coupon payments per year
    
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
    >>> loop(list)(bond_par_conversion_factor)([y * 100 for y in yld], tenor, invert = True, coupon = 6, rate_fmt=100)
    >>> [1.2918585801399018, 1.3355093954613233, 1.3794440302478643, 1.423587848577121,1.4678647692438134]
    >>> print('if yields are lower than 6% cheapest to deliver is the 7-year par bond')

    >>> yld = [0.08] * 5
    >>> loop(list)(bond_par_conversion_factor)(yld, tenor, invert = True)
    >>> [0.8985042974046984, 0.8884063695192392, 0.8790937290011396, 0.8704926716179342, 0.862538032755245]
    >>> print('if yields are higher than 6% cheapest to deliver is the 11-year par bond')
    """
    rate_fmt = rate_format(rate_fmt)
    if coupon is None:
        coupon = 0.06 * rate_fmt 
    res = bond_pv(coupon, tenor, coupon = yld, freq = freq, rate_fmt = rate_fmt)
    return 1/res if invert else res


def bond_ctd(tenor2yld, coupon = None, freq = 2, rate_fmt = 1):
    """
    returns yld, tenor and future price of a CTD future with multiple yields
    
    Parameters
    ----------
    tenor2yld : dict
        mapping from tenor in years (int) to yield timeseries
    coupon : float, optional
        The coupon for the future. The default is 6%
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
    rate_fmt = rate_format(rate_fmt)
    if coupon is None:
        coupon = 0.06 * rate_fmt 
    tenors = list(tenor2yld.keys())
    ylds = pd.concat(tenor2yld.values(), axis = 1).ffill()
    cfs = [bond_par_conversion_factor(yld, tenor, coupon, freq, invert = True, rate_fmt = rate_fmt) for tenor, yld in tenor2yld.items()]
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


def us_bond_quote(price, frac = 32, small_digits = 1):
    """
    converts price of a bond to the quote in the market place

    >>> assert bond_quote(114.125) == "114'4"
    >>> assert bond_quote(114.5) == "114'16"
    """
    whole = int(price)
    remainder = (price - whole)*frac
    numerator = int(remainder)
    final = '' if remainder == numerator else str(remainder - numerator)[:small_digits]
    res = f"{whole}'{numerator}{final}"
    return res

@cache
def _small_frac_map(small_frac, digits = 1):
    """
    assert  _small_frac_map(2,1) = {'0': 0.0, '5': 0.5, '': 0}
    assert  _small_frac_map(4,1) = {'0': 0.0, '2': 0.25, '5': 0.5, '7': 0.75, '': 0}
    assert  _small_frac_map(8,1) =  {'0': 0.0,
                                     '1': 0.125,
                                     '2': 0.25,
                                     '3': 0.375,
                                     '5': 0.5,
                                     '6': 0.625,
                                     '7': 0.75,
                                     '8': 0.875,
                                     '': 0}
    """
    res =  {str(i/small_frac).split('.')[1][:digits] :  i/small_frac for i in range(small_frac)}
    res[''] = 0
    return res

def us_bond_price(quote, frac = 32, small_frac = 4):
    """
    converts quoted price into actual price
    
    >>> assert us_bond_quote(114.125) == "114'4"
    >>> assert us_bond_quote(114.5) == "114'16"
    >>> assert us_bond_price(quote = "114'4") == 114.125
    >>> assert us_bond_price(quote = "114'16") == 114.5
    >>> assert us_bond_price(quote = "114'162") == 114.5078125
    """
    if "'" not in quote:
        return int(quote)
    whole, fraction = quote.split("'")
    large_digits = 1 if frac<10 else 2 if frac < 100 else 3
    numerator = int(fraction[:large_digits])
    remainder = fraction[large_digits:]
    res = int(whole) + numerator / frac + _small_frac_map(small_frac, len(remainder))[remainder] / frac
    return res
    
    