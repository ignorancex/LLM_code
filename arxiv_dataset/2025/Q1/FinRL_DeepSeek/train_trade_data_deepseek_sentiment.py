#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import datetime
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config import INDICATORS

import itertools


columns_to_load = ['Date', 'Stock_symbol', 'sentiment_deepseek']
sentiment = pd.read_csv('sentiment_deepseek.csv', usecols=columns_to_load)
#sentiment = pd.read_csv('sentiment_qwen_nasdaq_news_full.csv', usecols=columns_to_load)

print(sentiment.head())


#columns_risk = ['Date', 'Stock_symbol', 'risk_deepseek']

#risk = pd.read_csv('risk_deepseek.csv', usecols=columns_risk)

#risk = pd.read_csv('risk_qwen_nasdaq_news_full.csv', usecols=columns_risk)




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



# Step 1: Create a copy of sentiment_small to avoid warnings
def add_sentiment(train, sentiment, column_name='sentiment_deepseek'):
    sentiment_small = sentiment
    train_new = train
    train_new['date'] = pd.to_datetime(train_new['date'])

    # Step 2: Convert 'Date' to datetime without timezone
    sentiment_small['Date'] = pd.to_datetime(sentiment_small['Date']).dt.tz_localize(None)

    # Step 3: Rename 'Stock_symbol' to 'tic' and the sentiment column
    sentiment_small.rename(
        columns={'Stock_symbol': 'tic', column_name: 'llm_sentiment'},
        inplace=True
    )

    # Step 4: Merge train and sentiment_small into a new DataFrame
    train_new = train.merge(
        sentiment_small[['Date', 'tic', 'llm_sentiment']],
        left_on=['date', 'tic'],
        right_on=['Date', 'tic'],
        how='left'
    )

    # Step 5: Drop the additional 'Date' column
    train_new.drop(columns=['Date'], inplace=True)
    return train_new


def add_risk(train, risk, column_name='risk_deepseek'):
    train_new = train
    risk_new=risk
    train_new['date'] = pd.to_datetime(train_new['date'])

    # Step 2: Convert 'Date' to datetime without timezone
    risk_new['Date'] = pd.to_datetime(risk_new['Date']).dt.tz_localize(None)

    # Step 3: Rename 'Stock_symbol' to 'tic' and the sentiment column
    risk_new.rename(
        columns={'Stock_symbol': 'tic', column_name: 'llm_risk'},
        inplace=True
    )

    # Step 4: Merge train and sentiment_small into a new DataFrame
    train_new = train.merge(
        risk_new[['Date', 'tic', 'llm_risk']],
        left_on=['date', 'tic'],
        right_on=['Date', 'tic'],
        how='left'
    )

    # Step 5: Drop the additional 'Date' column
    train_new.drop(columns=['Date'], inplace=True)
    return train_new



stock_dimension = len(trade.tic.unique())
print("Stock Dimension:" + str(stock_dimension))

train_sentiment=add_sentiment(train,sentiment)
trade_sentiment=add_sentiment(trade,sentiment)

#train_risk=add_risk(train_sentiment,risk)
#trade_risk=add_risk(trade_sentiment,risk)


train_sentiment.to_csv('train_data_deepseek_sentiment_2013_2018.csv')
trade_sentiment.to_csv('trade_data_deepseek_sentiment_2019_2023.csv')
