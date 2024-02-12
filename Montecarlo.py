#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 16:29:06 2024

@author: rahulodedra
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# date
start_date = datetime(2018,1,1)
end_date = datetime.now().date()-timedelta(days=1)

# ticker = ['TSLA']
mkt_ticker = ['^GSPC']
# ticker = ['AAPL','NVDA','TSLA','LULU']

# download data
def get_stock_data(ticker,start_date=start_date,end_date=end_date):
    data = yf.download(ticker,start_date,end_date)
    return data['Adj Close']

def log_returns(data):
    return np.log(1+data.pct_change())

def simple_returns(data):
    return (data/data.shift(1))-1

# data = get_stock_data('AAPL', start_date, end_date)

# log_return = log_returns(data)
# simple_rt = simple_returns(data)

def bnchmark_combine(data,mkt_ticker,start_date=start_date,end_date=end_date):
    bnchmark_data = get_stock_data(mkt_ticker, start_date, end_date)
    bnchmark_return = log_returns(bnchmark_data).dropna()
    annual_return = np.exp(bnchmark_return.mean()*252)-1
    data = pd.merge(data,bnchmark_data,left_index=True,right_index=True)
    return data, annual_return

# x,y=bnchmark_combine(data,mkt_ticker,start_date, end_date)
    

def beta_sharpe(data,mkt_ticker,start_date=start_date,riskfree=0.04):
    #riskfree: the assumed risk free yield is 3%)
    mkt_data, ann_return = bnchmark_combine(data,mkt_ticker,start_date,end_date)
    log_rt = log_returns(mkt_data)
    covar = log_rt.cov()*252
    # covar = covar.iloc[0,1]
    covar = pd.DataFrame(covar.iloc[:-1,-1])
    mkt_var = log_rt.iloc[:,-1].var()*252
    beta = pd.DataFrame(index=[1],columns=['beta','Stdev','CAPM','Sharpe'])
    beta.at[1,'beta'] = covar.iloc[0,0]/mkt_var
        
    stdev_rt = (log_rt.std()*250**0.5)[:-1][0]
    # pd.merge(beta,stdev_rt, left_index=True, right_index=True)
    beta.at[1,'Stdev'] = stdev_rt
    # CAPM
    for i, row in beta.iterrows():
        beta.at[i,'CAPM'] = riskfree + (row['beta'] * (ann_return-riskfree))
        # beta['CAPM'] = 1
    # Sharpe
    for i, row in beta.iterrows():
        beta.at[i,'Sharpe'] = ((row['CAPM']-riskfree)/(row['Stdev']))
        return beta
    
# x=beta_sharpe(data,mkt_ticker)

def drift(data,return_type='log'):
    if return_type=='log':
        rts = log_returns(data)
    elif return_type=='simple':
        rts = simple_returns(data)
    mean = rts.mean()
    var = rts.var()
    drift = mean-(0.5*var)
    return drift
    
# x1=drift(data)
    
def daily_return(data,days,iteration,return_type='log'):
    dft = drift(data,return_type)
    if return_type=='log':
        std = log_returns(data).std()
    elif return_type=='simple':
        std = simple_returns(data).std()
    # cauchy distribution
    dr = np.exp(dft + std * norm.ppf(np.random.rand(days,iteration)))
    return dr

# dr = daily_return(data, 2, 3)
        
    
def prob_of_asset(predicted,higher,on='value'):
    if on == 'return':
        predicted0 = predicted.iloc[0,0]
        predicted = predicted.iloc[-1]
        predList = list(predicted)
        over = [(i*100)/predicted0 for i in predList if ((i-predicted0)*100)/predicted0 >= higher]
        less = [(i*100)/predicted0 for i in predList if ((i-predicted0)*100)/predicted0 < higher]
    elif on == 'value':
        predicted = predicted.iloc[-1]
        predList = list(predicted)
        over = [i for i in predList if i >= higher]
        less = [i for i in predList if i < higher]
    else:
        print("'on' must be either value or return")
    return round((len(over)/(len(over)+len(less))),2)
    
def simulate_mc(data,days,iterations,return_type='log',plot=True):
    # Generate daily returns
    returns = daily_return(data,days,iterations,return_type)
    # Create empty matrix
    price_list = np.zeros_like(returns)
    # Put the last actual price in the first row of matrix.
    price_list[0] = data.iloc[-1]
    # Calculate the price of each day
    for i in range(1,days):
        price_list[i] = price_list[i-1]*returns[i]
    # Plot Option
    if plot:
        x = pd.DataFrame(price_list).iloc[-1]
        fig, ax = plt.subplots(1,2, figsize=(14,4))
        sns.distplot(x, ax=ax[0])
        sns.distplot(x, hist_kws={'cumulative':True},kde_kws={'cumulative':True},ax=ax[1])
        plt.xlabel("Stock Price")
        plt.show()
    #CAPM and Sharpe Ratio
    # Printing information about stock
    
    print(ticker)
    print(f"Days: {days-1}")
    print(f"Expected Value: ${round(pd.DataFrame(price_list).iloc[-1].mean(),2)}")
    print(f"Return: {round(100*(pd.DataFrame(price_list).iloc[-1].mean()-price_list[0,1])/pd.DataFrame(price_list).iloc[-1].mean(),2)}%")
    print(f"Probability of Breakeven: {prob_of_asset(pd.DataFrame(price_list),0, on='return')}")
    return pd.DataFrame(price_list)   
        
# x3=simulate_mc(data, 252, 1000, 'log')
   

def monte_carlo(tickers,days_forecast,iterations,start_date,return_type='log', plotten=False):
    data = get_stock_data(ticker,start_date=start_date)
    inform = beta_sharpe(data,mkt_ticker,start_date=start_date)
    simulatedDF = []
    
    y = simulate_mc(data,(days_forecast+1),iterations,return_type)
    if plotten == True:
        forplot = y.iloc[:,0:10]
        forplot.plot(figsize=(15,4))
        
    print(f"Beta: {round(inform.iloc[0,inform.columns.get_loc('beta')],2)}")
    print(f"Sharpe: {round(inform.iloc[0,inform.columns.get_loc('Sharpe')],2)}") 
    print(f"CAPM Return: {round(100*inform.iloc[0,inform.columns.get_loc('CAPM')],2)}%")
    y['ticker'] = tickers
    cols = y.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    y = y[cols]
    simulatedDF.append(y)
    #simulatedDF = pd.concat(simulatedDF)
    return simulatedDF
    
ticker = 'NVDA'
start = "2018-1-1"
days_to_forecast= 252
simulation_trials= 10000
ret_sim_df = monte_carlo(ticker,days_forecast=days_to_forecast,iterations=simulation_trials,start_date=start,plotten=False)

    
    
    
    
    
    
    
    
    
