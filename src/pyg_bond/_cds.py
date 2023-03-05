from pyg_bond._base import rate_format, annual_freq

def cds_survival_probability(spread, yld=0.0, recovery_rate = 0.4, freq = 4, yld_format = 100, spread_format = 10000):
    """

    Parameters
    ----------
    spread : float/timeseries
        e.g. 0.02
    yld : flaot/timeseries
        e.g. 0.05
    recovery_rate : float, optional
        The recovery assumption on the bond. The seller of the protection pays (1-recovery_rate). The default is 0.4.
    freq : int, optional
        coupon frequency for a CDS. The default is 4.


    Maths
    ------
    Suppose you are selling protection on a bond and coupon frequency is quarterly.
    Suppose further payout upon default can only happen at the end of the coupon period.
    
    We assume there exists a constant survival probability for the underlying bond.
    at each coupon payment, you will get your coupon, provided the bond has not defaulted

    df = 1/(1 + yld/freq)   ## period discount factor
    n = tenor * freq        ## number of periods
    
    We let 
    
    sp = period survival probability < 1
    
    expected_coupons_pv = (spread/freq) * \sum(1<=i<=n) df^i * sp^i 
                        = (spread/freq) * df * sp * (1-(df*sp)^n)/(1 - df*sp)


    The probability of a payout on the ith period is like a geometric distribution so payout on ith period has prob sp^(i-1) * (1-sp)
    
    expected_payout_pv = (1-rr) * \sum(1<=i<n) df^i * (1-sp) * sp^(i-1)
                       = (1-rr) * (1-sp) * df * (1-(df*sp)^n)/(1-df*sp)
     
    We ley pay = 1-recover_rate.
    
    The two balance out when:
        
    payout_pv = coupons_pv
    
    pay * (1-sp) = (spread/freq) * sp

    pay = sp(pay + spread/freq)
    sp = pay/(pay+spread/freq)

    We assumed payouts on defaults only occur at the END of the coupon period.
    That undervalues the payout flow slightly. Given that a default happened sometime during the period, we expect it to have occured "in the middle"
    
    intrapperiod_payout_pv = expected_payout_pv / df^(0.5) which is ever so slightly bigger
    We now let  
    
    pay = (1-rr)/df^0.5      # represent the modified payout
    sp = pay/(pay+spread/freq)

    Note:
    -----
    Interestingly, the expression:
    uncertain_cash_flows_pv = sum(i<=n) (df*sp)^(i-1) = (1-(df*sp)^n)/(1 - df*sp)
    does not feature in the formula as it is common both for the payout and the coupons stream. It will feature once we need to value    
    Similarly, the discount factor hardly features in the calculation, only changing the payout calculation slightly. 
    
    Example:
    -------
    >>> assert cds_survival_probability(spread = 200, freq = 1, recovery_rate = 0) == 0.9803921568627451 ## very close to 2% decay, since recovery is 0
    >>> assert cds_survival_probability(spread = 200, freq = 1, recovery_rate = 0.4) == 0.9677419354838709 ## prob survival lower, to make defaults more likely and balance the book, when we cosider we only get paid 60c.

    """
    spread_format = rate_format(spread_format)
    yld_format = rate_format(yld_format)
    df = 1 / (1 + (yld/yld_format) / freq)
    pay = (1 - recovery_rate) / (df**0.5)    
    sp = pay/(pay+(spread/spread_format)/freq)
    return sp    
    

def cds_pv_and_duration(spread, yld, coupon, tenor, recovery_rate = 0.4, freq = 4, yld_format = 100, spread_format = 10000):
    """
    see cds_survival_probability for a full calculations

    Parameters
    ----------
    spread : float/timeseries
        e.g. 0.02
    yld : flaot/timeseries
        e.g. 0.05
    coupon: float
        The CDX series is usually issued with a running_coupon. When we trade the CDX, we pay a cash payment to represent the different between current spreads and the original coupon agreed.
    tenor: float
        The period left in the life of the cds
    recovery_rate : float, optional
        The recovery assumption on the bond. The seller of the protection pays (1-recovery_rate). The default is 0.4.
    freq : int, optional
        coupon frequency for a CDS. The default is 4.

    Parameters
    ----------
    spread : float/timeseries
        DESCRIPTION.
    yld : TYPE
        DESCRIPTION.
    coupon : TYPE
        DESCRIPTION.
    tenor : TYPE
        DESCRIPTION.
    recovery_rate : TYPE, optional
        DESCRIPTION. The default is 0.4.
    freq : TYPE, optional
        DESCRIPTION. The default is 4.


    Example:
    -------
    >>> coupon = 300
    >>> spread = 200

    """
    spread_format = rate_format(spread_format)
    yld_format = rate_format(yld_format)
    yld = yld/yld_format
    sprd = spread / spread_format
    cpn = coupon / spread_format

    df = 1 / (1 + yld / freq)
    pay = (1 - recovery_rate) / (df**0.5)    
    sp = pay/(pay+sprd/freq)

    freq = annual_freq(freq)
    n = tenor * freq
    dfsp = df * sp
    duration = (dfsp * (1 - dfsp**n)/(1-dfsp))/freq
    pv = (cpn- sprd)*duration
    return dict(pv = pv, duration = duration)

cds_pv_and_duration.output = ['pv', 'duration']


def cds_duration(spread, yld, coupon, tenor, recovery_rate = 0.4, freq = 4, yld_format = 100, spread_format = 10000):
    """
    see cds_pv_and_duration
    """
    return cds_pv_and_duration(spread, yld, coupon, tenor, recovery_rate = recovery_rate , freq = freq, yld_format = yld_format , spread_format = spread_format )['duration']


def cds_pv(spread, yld, coupon, tenor, recovery_rate = 0.4, freq = 4, yld_format = 100, spread_format = 10000):
    """
    see cds_pv_and_duration
    """
    return cds_pv_and_duration(spread, yld, coupon, tenor, recovery_rate = recovery_rate , freq = freq, yld_format = yld_format , spread_format = spread_format )['pv']



    
    ###
    ###



