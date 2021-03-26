#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np

from scipy import stats


def black_76(days_to_expiry, futures_price, strike_price, implied_volatility,
             interest_rate, option_type="call"):
    
    if option_type not in ["call", "put"]:
        raise ValueError("Must provide 'call' for European call or 'put' for European put option.")

    F = futures_price
    K = strike_price
    sigma = implied_volatility
    r = interest_rate
    
    time_to_expiry = (days_to_expiry) / 365
    
    d_1 = (np.log(F / K) + 0.5 * (time_to_expiry) * sigma ** 2) / (sigma * np.sqrt(time_to_expiry))
    d_2 = d_1 - sigma * np.sqrt(time_to_expiry)
    
    if option_type == "call":
        C = np.exp(-r * (time_to_expiry)) * (F * stats.norm.cdf(d_1) - K * stats.norm.cdf(d_2))
        
        return C
    else:
        P = np.exp(-r * (time_to_expiry)) * (K * stats.norm.cdf(-d_2) - F * stats.norm.cdf(-d_1))
        
        return P


    
def tune(F_vect, delta=1/12):
    
    if type(F_vect) != np.ndarray:
        raise TypeError("Must provide numpy array.")
    
    n = len(F_vect) - 1
    
    x, y = F_vect[0:-1], F_vect[1:]
    
    s_x, s_y = np.sum(x), np.sum(y)
    
    s_xx = np.sum(x ** 2)
    s_xy = np.sum(x * y)
    s_yy = np.sum(y ** 2)
                  
    a = (n * s_xy - s_x * s_y) / (n * s_xx - s_x **2)
    b = (s_y - a * s_x) / n
    sd = np.sqrt((n * s_yy - s_y ** 2 - a * (n * s_xy - s_x * s_y)) / (n * (n-2)))
    
    lambda_par = - np.log(a) / delta
    mu = b / (1 - a)
    
    return mu, lambda_par
                  

                  
def gbm_path_gen(t, T, F_init, sigma, n=12):
    
    path = np.zeros(n)
    ttm = T - t
    dt = (ttm.days / (n * 365))
    
    path[0] = F_init
    
    for i in range(n - 1):
        path[i+1] = path[i] + path[i] * sigma * np.sqrt(dt) * np.random.randn()
                  
    return path


                  
def vasicek_path_gen(t, T, F_init, lambda_par, mu, sigma, n=12):
                  
    path = np.zeros(n)
    ttm = T - t
    dt = (ttm.days / (n * 365))

    path[0] = F_init
                  
    for i in range(n - 1):
        path[i+1] = path[i] + lambda_par * (mu - path[i]) * dt + sigma * np.sqrt(dt) * np.random.randn()
                  
    return path


                  
def garch_path_gen(t, T, F_init, lambda_par, mu, sigma, n=12):

    path = np.zeros(n)
    ttm = T - t
    dt = (ttm.days / (n * 365))
    
    path[0] = F_init
    
    for i in range(n - 1):
        path[i+1] = path[i] + lambda_par * (mu - path[i]) * dt + sigma * np.sqrt(dt) * path[i] * np.random.randn()
                  
    return path
                  


def monte_vasicek(t, T, F, K, sigma, r, lambda_par, mu, m=1e5, n=12):

    sim_start_date, sim_final_date = pd.to_datetime(t), pd.to_datetime(T)
    monte_mat = np.zeros((m, n))
    a_val = 0
                  
    for j in range(m):
        vasicek_mat[j, :] = vasicek_path_gen(pd.to_datetime(sim_start_date), pd.to_datetime(sim_final_date), F, 
                                             lambda_par, mu, sigma, n=n)
        a_val += max((vasicek_mat[j, -1] - K), 0)    
    
    price = (a_value / m) * (np.exp(-r * (10/365)))
                  
    return price
                  
                  

def monte_garch(t, T, T_e, F, K, sigma, r, lambda_par, mu, m=1000, n=12):

    sim_start_date, sim_final_date = pd.to_datetime(t), pd.to_datetime(T)
    monte_mat = np.zeros((m, n))
    a_val = 0
                  
    for j in range(m):
        monte_mat[j, :] = garch_path_gen(sim_start_date, sim_final_date, F, lambda_par, mu, sigma, n=n)
        
        a_val += max((monte_mat[j, -1] * (np.exp(-r * (10 / 365))) - K, 0))            
    
    price = (a_val / m) * (np.exp(-r * (T_e / 365)))
                  
    return price               

