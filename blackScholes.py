# -*- coding: utf-8 -*-
def blackScholes (S0, r, T, sigma, K):
    import math 
    from scipy.stats import norm
    ''' Black Scholes formula for pricing EU-Call-options where, 
    S0 is the stock-value at time 0;
    r is the interest rate
    T is the strike-date
    sigma is the volatilty of the stock
    K is the strike.price'''

    d1 = (1/(sigma*math.sqrt(T))) * ( math.log(S0/K) + (r + 0.5*sigma**2) * T)
    
    d2 = d1 - sigma * math.sqrt(T)

    P = S0 * norm.cdf(d1) - math.exp(-r*T) * K * norm.cdf(d2)

    return P
