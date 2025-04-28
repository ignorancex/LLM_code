#!/usr/bin/env python
# coding: utf-8

# # Stock NeurIPS2018 Part 1. Data
# This series is a reproduction of paper *the process in the paper Practical Deep Reinforcement Learning Approach for Stock Trading*. 
# 
# This is the first part of the NeurIPS2018 series, introducing how to use FinRL to fetch and process data that we need for ML/RL trading.
# 
# Other demos can be found at the repo of [FinRL-Tutorials]((https://github.com/AI4Finance-Foundation/FinRL-Tutorials)).

# # Part 1. Install Packages

# In[ ]:


## install finrl library


#import subprocess

#already installed 

#subprocess.run(['pip', 'install', 'git+https://github.com/AI4Finance-Foundation/FinRL.git'])

#for ipython notebook:
#get_ipython().system('pip install git+https://github.com/AI4Finance-Foundation/FinRL.git')


# In[2]:


import pandas as pd
import numpy as np
import datetime
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config import INDICATORS

import itertools


# # Part 2. Fetch data

# [yfinance](https://github.com/ranaroussi/yfinance) is an open-source library that provides APIs fetching historical data form Yahoo Finance. In FinRL, we have a class called [YahooDownloader](https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/meta/preprocessor/yahoodownloader.py) that use yfinance to fetch data from Yahoo Finance.

# **OHLCV**: Data downloaded are in the form of OHLCV, corresponding to **open, high, low, close, volume,** respectively. OHLCV is important because they contain most of numerical information of a stock in time series. From OHLCV, traders can get further judgement and prediction like the momentum, people's interest, market trends, etc.


# ## Data for the chosen tickers

# In[8]:


# http://web.archive.org/web/20230913195340/https://en.wikipedia.org/wiki/Nasdaq-100

nasdaq_100_tickers_july_17_2023 = [
    "ADBE", "ADP", "ABNB", "ALGN", "GOOGL", "GOOG", "AMZN", "AMD", "AEP", "AMGN",
    "ADI", "ANSS", "AAPL", "AMAT", "ASML", "AZN", "TEAM", "ADSK", "BKR", "BIIB",
    "BKNG", "AVGO", "CDNS", "CHTR", "CTAS", "CSCO", "CTSH", "CMCSA", "CEG", "CPRT",
    "CSGP", "COST", "CRWD", "CSX", "DDOG", "DXCM", "FANG", "DLTR", "EBAY", "EA",
    "ENPH", "EXC", "FAST", "FTNT", "GEHC", "GILD", "GFS", "HON", "IDXX", "ILMN",
    "INTC", "INTU", "ISRG", "JD", "KDP", "KLAC", "KHC", "LRCX", "LCID", "LULU",
    "MAR", "MRVL", "MELI", "META", "MCHP", "MU", "MSFT", "MRNA", "MDLZ", "MNST",
    "NFLX", "NVDA", "NXPI", "ORLY", "ODFL", "ON", "PCAR", "PANW", "PAYX", "PYPL",
    "PDD", "PEP", "QCOM", "REGN", "ROST", "SGEN", "SIRI", "SBUX", "SNPS", "TMUS",
    "TSLA", "TXN", "TTD", "VRSK", "VRTX", "WBA", "WBD", "WDAY", "XEL", "ZM", "ZS"
]

position = nasdaq_100_tickers_july_17_2023.index("SGEN")
nasdaq_100_tickers_july_17_2023.remove("SGEN")


# In[9]:

#real period for the paper

TRAIN_START_DATE = '2013-01-01'
TRAIN_END_DATE = '2018-12-31'
TRADE_START_DATE = '2019-01-01'
TRADE_END_DATE = '2023-12-31'



#small period for testing code

#TRAIN_START_DATE = '2022-01-01'
#TRAIN_END_DATE = '2023-02-15'
#TRADE_START_DATE = '2023-02-16'
#TRADE_END_DATE = '2023-03-30'



# In[10]:


df_raw = YahooDownloader(start_date = TRAIN_START_DATE,
                     end_date = TRADE_END_DATE,
                     ticker_list = nasdaq_100_tickers_july_17_2023).fetch_data()


nasdaq_100_tickers_july_17_2023.insert(position, "SGEN")

# In[11]:


df_raw.head()


#bring SGEN data from elsewhere and insert it in df_raw

df_seagen=pd.read_csv('seagen.csv')

# Rename columns in df_seagen to match df_raw
df_seagen.rename(columns={
    'Date': 'date',
    'Price': 'close',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Vol.': 'volume',
    'Change %': 'change_percent'
}, inplace=True)

# Add the 'tic' column to df_seagen
df_seagen['tic'] = 'SGEN'

# Function to convert volume strings to numeric values
def convert_volume(vol_str):
    if vol_str.endswith('M'):
        return float(vol_str.replace('M', '')) * 1e6
    elif vol_str.endswith('K'):
        return float(vol_str.replace('K', '')) * 1e3
    else:
        return float(vol_str)

# Convert 'volume' to numeric
df_seagen['volume'] = df_seagen['volume'].apply(convert_volume)

# Convert 'date' to datetime format
df_seagen['date'] = pd.to_datetime(df_seagen['date'], format='%m/%d/%Y')
df_raw['date'] = pd.to_datetime(df_raw['date'])

# Drop the 'change_percent' column if not needed
df_seagen.drop(columns=['change_percent'], inplace=True)

# Concatenate df_seagen with df_raw
df_combined = pd.concat([df_raw, df_seagen], ignore_index=True)

# Sort the combined DataFrame by date
df_combined.sort_values(by='date', inplace=True)


# Restore 'date' column in df_combined to string type
df_combined['date'] = df_combined['date'].dt.strftime('%Y-%m-%d')


df_raw=df_combined



# # Part 3: Preprocess Data
# We need to check for missing data and do feature engineering to convert the data point into a state.
# * **Adding technical indicators**. In practical trading, various information needs to be taken into account, such as historical prices, current holding shares, technical indicators, etc. Here, we demonstrate two trend-following technical indicators: MACD and RSI.
# * **Adding turbulence index**. Risk-aversion reflects whether an investor prefers to protect the capital. It also influences one's trading strategy when facing different market volatility level. To control the risk in a worst-case scenario, such as financial crisis of 2007–2008, FinRL employs the turbulence index that measures extreme fluctuation of asset price.

# Hear let's take **MACD** as an example. Moving average convergence/divergence (MACD) is one of the most commonly used indicator showing bull and bear market. Its calculation is based on EMA (Exponential Moving Average indicator, measuring trend direction over a period of time.)

# In[14]:


fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list = INDICATORS,
                     use_vix=True,
                     use_turbulence=True,
                     user_defined_feature = False)

processed = fe.preprocess_data(df_raw)


# In[15]:


list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
combination = list(itertools.product(list_date,list_ticker))

processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date','tic'])

#processed_full = processed_full.fillna(0)
#processed_full = processed_full.fillna(method='ffill')

processed_full = processed_full.ffill()


# In[16]:


processed_full.head()


# # Part 4: Save the Data

# ### Split the data for training and trading

# In[17]:


train = data_split(processed_full, TRAIN_START_DATE,TRAIN_END_DATE)
trade = data_split(processed_full, TRADE_START_DATE,TRADE_END_DATE)
#print(len(train))
#print(len(trade))


# ### Save data to csv file

# For Colab users, you can open the virtual directory in colab and manually download the files.
# 
# For users running on your local environment, the csv files should be at the same directory of this notebook.

# In[18]:


train.to_csv('train_data_2013_2018.csv')
trade.to_csv('trade_data_2019_2023.csv')
