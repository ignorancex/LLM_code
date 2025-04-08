import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import requests
import json
import os
from tqdm import tqdm
from bs4 import BeautifulSoup


def tickers_sp500():
    '''Downloads list of tickers currently listed in the S&P 500 '''
    #TODO check why does si.tickers_sp500() not work?
    # get list of all S&P 500 stocks 
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    sp_tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip()
        if ticker == "BRK.B" or ticker == "BF.B":
            continue
        sp_tickers.append(ticker)
    
    return sp_tickers

# Function to fetch and save S&P 500 stock prices
def fetch_and_save_prices(tickers, start=None, end=None, save_path='stock_history.csv'):
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    start = dt.datetime(2000, 1, 1).strftime('%Y-%m-%d') if not start else start # Start date
    end = dt.datetime.now().strftime('%Y-%m-%d') if not end else end # Format today's date
    prices = yf.download(tickers, start=start, end=end)
    # prices = prices['Close']  # We're interested in the closing prices
    
    if len(tickers) == 1:
        columns = prices.columns
        # Create a MultiIndex for the columns
        arrays = [columns, tickers * len(columns)]
        multi_index = pd.MultiIndex.from_arrays(arrays, names=('Price', 'Ticker'))
        # Assign the MultiIndex to the DataFrame columns
        prices.columns = multi_index
        
    prices.to_csv(save_path)  # Save to a CSV file
    
    return prices

# Function to fetch and save S&P 500 EPS data
def fetch_and_save_eps(tickers, apikey, save_dir='./eps_history'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    fail_list = []   
    for ticker in tqdm(tickers, desc= 'Fetching EPS data: '):
        url = 'https://www.alphavantage.co/query?function=EARNINGS&symbol='+ticker+'&apikey='+apikey
        retry = 0
        while retry < 3:
            try:
                r = requests.get(url)
                data = r.json()
                break
            except Exception as e:
                retry += 1
                print('Error fetching data for', ticker, 'raising', e)
        
        if retry == 3:
            print('Failed to fetch data for', ticker)
            fail_list.append(ticker)
            continue
        
        filename = save_dir + '/data-' + ticker + '.json'
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    
    return fail_list
    
# Main script
if __name__ == "__main__":
    tickers = tickers_sp500()
    tickers = tickers[:500]
    fetch_and_save_prices(tickers)
    tickers = tickers[:25]
    fetch_and_save_eps(tickers, 'JPRE1OVJIZOWJS0O', save_dir='./eps_data')