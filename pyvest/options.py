# -*- coding: utf-8 -*-




import numpy as np
from scipy.stats import norm




def callpremium(spot, strike, volatility, ttm, risk_free = 0.02, dividend_yield = 0):
    
    """
    Calculates the premium of a European-style call option using the Black-Scholes option pricing
    model
    
    spot = Spot price of the underlying asset
    strike =  Strike price of the option
    volatility =  Implied volatility of the underlying asset price, defined as the annualized standard deviation of the asset returns
    ttm = Time to maturity in years
    risk_free =  Annual continuously-compounded risk-free rate
    dividend_yield = Annual continuously-compounded dividend yield
    """
    
    
    if ttm == 0:
        ttm = 1e-200
        
    d1 = ((np.log(spot/strike)+((risk_free - dividend_yield+(volatility**2/2))*ttm)))/(volatility*np.sqrt(ttm))
    d2 = d1 - volatility * np.sqrt(ttm)
    
    return spot* np.exp(-dividend_yield*ttm)* norm.cdf(d1)-strike*np.exp(-risk_free*ttm)*norm.cdf(d2)


def putpremium(spot, strike, volatility, ttm, risk_free = 0.02, dividend_yield = 0):
    
    """
    Calculates the premium of a European-style put option using the Black-Scholes option pricing
    model
    
    spot = Spot price of the underlying asset
    strike =  Strike price of the option
    volatility =  Implied volatility of the underlying asset price, defined as the annualized standard deviation of the asset returns
    ttm = Time to maturity in years
    risk_free =  Annual continuously-compounded risk-free rate
    dividend_yield = Annual continuously-compounded dividend yield
    """
    
    if ttm == 0:
        ttm = 1e-200
    
    
    d1 = ((np.log(spot/strike)+((risk_free - dividend_yield+(volatility**2/2))*ttm)))/(volatility*np.sqrt(ttm))
    d2 = d1 - volatility * np.sqrt(ttm)
    
    return strike* np.exp(-risk_free * ttm) * norm.cdf(-d2) - spot * np.exp(-dividend_yield * ttm) * norm.cdf(-d1)



def calldelta(spot, strike, volatility, ttm, risk_free = 0.02, dividend_yield = 0):
    
    """
    Calculates the delta of a European-style call option using the Black-Scholes option pricing
    model
    
    spot = Spot price of the underlying asset
    strike =  Strike price of the option
    volatility =  Implied volatility of the underlying asset price, defined as the annualized standard deviation of the asset returns
    ttm = Time to maturity in years
    risk_free =  Annual continuously-compounded risk-free rate
    dividend_yield = Annual continuously-compounded dividend yield
    """
    
    if ttm == 0:
        ttm = 1e-200
        
    d1 = ((np.log(spot/strike)+((risk_free - dividend_yield+(volatility**2/2))*ttm)))/(volatility*np.sqrt(ttm))
    
    return np.exp(-dividend_yield * ttm) * norm.cdf(d1)



def putdelta(spot, strike, volatility, ttm, risk_free = 0.02, dividend_yield = 0):
    
    
    """
    Calculates the delta of a European-style put option using the Black-Scholes option pricing
    model
    
    spot = Spot price of the underlying asset
    strike =  Strike price of the option
    volatility =  Implied volatility of the underlying asset price, defined as the annualized standard deviation of the asset returns
    ttm = Time to maturity in years
    risk_free =  Annual continuously-compounded risk-free rate
    dividend_yield = Annual continuously-compounded dividend yield
    """
    
    if ttm == 0:
        ttm = 1e-200
        
    d1 = ((np.log(spot/strike)+((risk_free - dividend_yield+(volatility**2/2))*ttm)))/(volatility*np.sqrt(ttm))
    
    return np.exp(-dividend_yield * ttm) * (norm.cdf(d1) - 1)



def calltheta(spot, strike, volatility, ttm, risk_free = 0.02, dividend_yield = 0):
    
    """
    Calculates the theta of a European-style call option using the Black-Scholes option pricing
    model
    
    spot = Spot price of the underlying asset
    strike =  Strike price of the option
    volatility =  Implied volatility of the underlying asset price, defined as the annualized standard deviation of the asset returns
    ttm = Time to maturity in years
    risk_free =  Annual continuously-compounded risk-free rate
    dividend_yield = Annual continuously-compounded dividend yield
    """
    
    if ttm == 0:
        
        return 0
    
    theta = callpremium(spot = spot, strike = strike, volatility  = volatility, ttm = ttm, risk_free = risk_free, dividend_yield = dividend_yield) - callpremium(spot = spot, strike = strike, volatility  = volatility, ttm = ttm - (1/365), risk_free = risk_free, dividend_yield = dividend_yield)
    
    return theta


def puttheta(spot, strike, volatility, ttm, risk_free = 0.02, dividend_yield = 0):
    
    """
    Calculates the theta of a European-style put option using the Black-Scholes option pricing
    model
    
    spot = Spot price of the underlying asset
    strike =  Strike price of the option
    volatility =  Implied volatility of the underlying asset price, defined as the annualized standard deviation of the asset returns
    ttm = Time to maturity in years
    risk_free =  Annual continuously-compounded risk-free rate
    dividend_yield = Annual continuously-compounded dividend yield
    """
    
    if ttm == 0:
        
        return 0
    
    theta = putpremium(spot = spot, strike = strike, volatility  = volatility, ttm = ttm, risk_free = risk_free, dividend_yield = dividend_yield) - putpremium(spot = spot, strike = strike, volatility  = volatility, ttm = ttm - (1/365), risk_free = risk_free, dividend_yield = dividend_yield)
    
    return theta


def callrho(spot, strike, volatility, ttm, risk_free = 0.02, dividend_yield = 0):
    
    """
    Calculates the rho of a European-style call option using the Black-Scholes option pricing
    model
    
    spot = Spot price of the underlying asset
    strike =  Strike price of the option
    volatility =  Implied volatility of the underlying asset price, defined as the annualized standard deviation of the asset returns
    ttm = Time to maturity in years
    risk_free =  Annual continuously-compounded risk-free rate
    dividend_yield = Annual continuously-compounded dividend yield
    """
    
    if ttm == 0:
        ttm = 1e-200
        
    s = spot
    x = strike
    sigma = volatility
    t = ttm
    r = risk_free
    d = dividend_yield

    d1 = ((np.log(spot/strike)+((risk_free - dividend_yield+(volatility**2/2))*ttm)))/(volatility*np.sqrt(ttm))
    d2 = d1 - volatility * np.sqrt(ttm)
    
    return (x*(t)*np.exp(-r*(t))*norm.cdf(d2))/100


def putrho(spot, strike, volatility, ttm, risk_free = 0.02, dividend_yield = 0):
    
    """
    Calculates the rho of a European-style put option using the Black-Scholes option pricing
    model
    
    spot = Spot price of the underlying asset
    strike =  Strike price of the option
    volatility =  Implied volatility of the underlying asset price, defined as the annualized standard deviation of the asset returns
    ttm = Time to maturity in years
    risk_free =  Annual continuously-compounded risk-free rate
    dividend_yield = Annual continuously-compounded dividend yield
    """
    
    if ttm == 0:
        ttm = 1e-200
        
    s = spot
    x = strike
    sigma = volatility
    t = ttm
    r = risk_free
    d = dividend_yield

    d1 = ((np.log(spot/strike)+((risk_free - dividend_yield+(volatility**2/2))*ttm)))/(volatility*np.sqrt(ttm))
    d2 = d1 - volatility * np.sqrt(ttm)
    
    return (-1/100)*(x*(t)*np.exp(-r*(t))*norm.cdf(-d2))


def optiongamma(spot, strike, volatility, ttm, risk_free = 0.02, dividend_yield = 0):
    
    """
    Calculates the gamma of a European-style option using the Black-Scholes option pricing
    model
    
    spot = Spot price of the underlying asset
    strike =  Strike price of the option
    volatility =  Implied volatility of the underlying asset price, defined as the annualized standard deviation of the asset returns
    ttm = Time to maturity in years
    risk_free =  Annual continuously-compounded risk-free rate
    dividend_yield = Annual continuously-compounded dividend yield
    """
    
    if ttm == 0:
        ttm = 1e-200
        
    s = spot
    x = strike
    sigma = volatility
    t = ttm
    r = risk_free
    d = dividend_yield

    d1 = ((np.log(spot/strike)+((risk_free - dividend_yield+(volatility**2/2))*ttm)))/(volatility*np.sqrt(ttm))
    d2 = d1 - volatility * np.sqrt(ttm)
    
    return (np.exp(-d*t)/(s*sigma*np.sqrt(t)))*(1/np.sqrt(2*np.pi))*(np.exp((-d1**2)/2))


def optionvega(spot, strike, volatility, ttm, risk_free = 0.02, dividend_yield = 0):
    
    """
    Calculates the vega of a European-style option using the Black-Scholes option pricing
    model
    
    spot = Spot price of the underlying asset
    strike =  Strike price of the option
    volatility =  Implied volatility of the underlying asset price, defined as the annualized standard deviation of the asset returns
    ttm = Time to maturity in years
    risk_free =  Annual continuously-compounded risk-free rate
    dividend_yield = Annual continuously-compounded dividend yield
    """
    
    if ttm == 0:
        ttm = 1e-200
        
    s = spot
    x = strike
    sigma = volatility
    t = ttm
    r = risk_free
    d = dividend_yield

    d1 = ((np.log(spot/strike)+((risk_free - dividend_yield+(volatility**2/2))*ttm)))/(volatility*np.sqrt(ttm))
    d2 = d1 - volatility * np.sqrt(ttm)
    
    return (1/100)*(s*np.exp(-d*t)*np.sqrt(t)*(1/np.sqrt(2*np.pi))*np.exp((-d1**2)/2))


def impliedvol(premium, spot, strike, ttm, risk_free = 0, dividend_yield = 0, calls = True, error = 0.01, vol = None):
    
    """
    Calculates the implied volatility of a European-style option using the Black-Scholes option pricing
    model
    
    option = 'call' or 'put'
    price = Option premium
    spot = Spot price of the underlying asset
    strike =  Strike price of the option
    volatility =  Implied volatility of the underlying asset price, defined as the annualized standard deviation of the asset returns
    ttm = Time to maturity in years
    risk_free =  Annual continuously-compounded risk-free rate
    dividend_yield = Annual continuously-compounded dividend yield
    """
    if not vol:
        testvols = np.array([0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.15, 0.25, 0.50, 0.75])
        
    else:
        testvols = np.array([0.01, vol*75, vol*0.90, vol, vol*1.10, vol * 1.25])
    
    if calls:
        
        testvals = np.array([(premium - callpremium(spot = spot, strike = strike, volatility = x, ttm = ttm, risk_free = risk_free, dividend_yield = dividend_yield))/premium for x in testvols])
    
    else:
        
        testvals = np.array([(premium - putpremium(spot = spot, strike = strike, volatility = x, ttm = ttm, risk_free = risk_free, dividend_yield = dividend_yield))/premium for x in testvols])

    
    npmin = np.min(abs(testvals))

    npargwhere = np.argmin(abs(testvals))
    #return npargwhere, len(testvals)
    if npargwhere < len(testvals)-1:
        
        if npmin <= 0:
        
            point1x = testvols[npargwhere - 1]
            point1y = testvals[npargwhere - 1]
            point2x = testvols[npargwhere]
            point2y = testvals[npargwhere]
        
        else:
            
            point1x = testvols[npargwhere]
            point1y = testvals[npargwhere]
            point2x = testvols[npargwhere + 1]
            point2y = testvals[npargwhere + 1]
    else:
        
        if calls:
            return impliedvol(calls = True,premium = premium, spot = spot, strike = strike, ttm = ttm, risk_free = risk_free, dividend_yield = dividend_yield, vol = testvols[-1])
        else:
            return impliedvol(calls = False,premium = premium, spot = spot, strike = strike, ttm = ttm, risk_free = risk_free, dividend_yield = dividend_yield, vol = testvols[-1])
            
    slope = (point2y-point1y)/(point2x-point1x)
    
    testiv = point1x - (point1y/slope)
    
    #print(testiv)
    if calls:
        
        if np.isnan(callpremium(spot = spot, strike = strike, volatility = testiv, ttm = ttm, risk_free = risk_free, dividend_yield = dividend_yield)):
            return np.nan
        
    else:
        
        if np.isnan(putpremium(spot = spot, strike = strike, volatility = testiv, ttm = ttm, risk_free = risk_free, dividend_yield = dividend_yield)):
            return np.nan
    #return (premium - opt.callpremium(spot = spot, strike = strike, volatility = testiv, ttm = ttm, risk_free = risk_free, dividend_yield = dividend_yield))/premium
    
    if calls:
    
        if (premium - callpremium(spot = spot, strike = strike, volatility = testiv, ttm = ttm, risk_free = risk_free, dividend_yield = dividend_yield)) < error:
    
            return testiv
    
        else:
    
            return impliedvol(calls = True, premium = premium, spot = spot, strike = strike, ttm = ttm, risk_free = risk_free, dividend_yield = dividend_yield, vol = testiv)
    else:
        
        if (premium - putpremium(spot = spot, strike = strike, volatility = testiv, ttm = ttm, risk_free = risk_free, dividend_yield = dividend_yield)) < error:
    
            return testiv
    
        else:
    
            return impliedvol(calls = False, premium = premium, spot = spot, strike = strike, ttm = ttm, risk_free = risk_free, dividend_yield = dividend_yield, vol = testiv)



