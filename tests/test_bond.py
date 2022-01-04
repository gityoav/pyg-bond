
from pyg_bond._bond import aus_bill_pv, bond_pv, bond_yld, bond_duration, aus_bond_pv, bond_yld_and_duration
import numpy as np
import pandas as pd

_aus_vals = [100, 100.24718483932983, 99.75403115605357]

def test_aus_bill_pv():
    assert aus_bill_pv(100) == _aus_vals[0]
    assert aus_bill_pv(101) == _aus_vals[1]
    assert aus_bill_pv(99)  == _aus_vals[2]
    assert aus_bill_pv(101, tenor = 90, daycount = 360) == 100.25062656641603
    assert aus_bill_pv(101, tenor = 30, daycount = 360) == 100.0834028356964
    assert aus_bill_pv(100, facevalue = 540) == 540


def test_aus_bill_pv_series():
    quote = np.array([100,101,99])
    assert list(aus_bill_pv(quote)) ==  _aus_vals
    quote = pd.Series(quote)
    assert list(aus_bill_pv(quote).values) == _aus_vals
    quote = pd.DataFrame([100,101,99])
    assert list(aus_bill_pv(quote).values[:,0]) == _aus_vals

def test_bond_pv_coupon_same_as_yield():
    for tenor in [1,2,5,10,20]:
        for coupon in [0, 0.01, 0.02, 0.1]:
            for freq in [1,2,4]:                
                assert abs(bond_pv(coupon, tenor = tenor, freq = freq, coupon = coupon)-1)<1e-10

def test_bond_pv_coupon_near_yield():
    for tenor in [1,2,5,10,20]:
        for coupon in [0, 0.01, 0.02, 0.1]:
            for freq in [1,2,4]:                
                res = bond_pv(yld = coupon+0.01, tenor = tenor, freq = freq, coupon = coupon)
                assert res<1 and res>1-tenor*0.01
                

def test_bond_yld():
    for tenor in [1,2,5,10,20]:
        for coupon in [0, 0.01, 0.02, 0.1]:
            for yld in [0, 0.01, 0.02, 0.1]:
                for freq in [1,2,4]:                
                    px = bond_pv(yld, tenor, coupon, freq)
                    estimated_yld = bond_yld(px*100, tenor, coupon, freq, 10)
                    assert abs(estimated_yld-yld)<1e-10

def test_bond_duration():
    epsilon = 1e-6
    for tenor in [1,2,5,10,20]:
        for coupon in [0, 0.01, 0.02, 0.1]:
            for yld in [0, 0.01, 0.02, 0.1]:
                for freq in [1,2,4]:                
                    px = bond_pv(yld, tenor, coupon, freq)
                    px1 = bond_pv(yld + epsilon, tenor, coupon, freq)
                    estimated_yld = bond_yld(px*100, tenor, coupon, freq, 10)
                    estimated_duration = bond_duration(estimated_yld, tenor, coupon, freq)
                    empirical_duration = (px-px1)/epsilon
                    assert abs(empirical_duration-estimated_duration)<1e-2
    
