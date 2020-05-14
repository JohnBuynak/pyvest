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



def impliedvol(option = 'call', price = 2.92, spot = 100, strike = 100, ttm = 45/365, risk_free = 0, dividend_yield = 0):
    
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
    
    vol = 1e-10
    
    if option == 'call':
    
        while vol < 100:
            
            diff1 = abs(price - callpremium(spot = spot, strike = strike, volatility = vol, ttm = ttm))
    
            vol = vol + 0.05
            
            diff2 = abs(price - callpremium(spot = spot, strike = strike, volatility = vol, ttm = ttm))
    
            if diff2 > diff1:
                
                vol = vol - 0.05
                
                difference = abs(price - callpremium(spot = spot, strike = strike, volatility = vol, ttm = ttm))
                
                while difference >= 0.005:
                    
                    difference = price - callpremium(spot = spot, strike = strike, volatility = vol, ttm = ttm)
                    
                    vol = vol + 0.0001
                
                return vol
    else:
        
        while vol < 100:
            
            diff1 = abs(price - putpremium(spot = spot, strike = strike, volatility = vol, ttm = ttm))
    
            vol = vol + 0.05
            
            diff2 = abs(price - putpremium(spot = spot, strike = strike, volatility = vol, ttm = ttm))
    
            if diff2 > diff1:
                
                vol = vol - 0.05
                
                difference = abs(price - putpremium(spot = spot, strike = strike, volatility = vol, ttm = ttm))
                
                while difference >= 0.005:
                    
                    difference = price - putpremium(spot = spot, strike = strike, volatility = vol, ttm = ttm)
                    
                    vol = vol + 0.0001
                
                return vol



