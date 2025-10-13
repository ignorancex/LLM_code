"""
Data processors for stock parity data
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt



def compute_log_returns_and_parity(ticker, data=None, start=None, end=None, save_dir=None):
    """Compute log(Y_t/Y_{t-1}) and its sign for a ticker (e.g., AAPL).

    Optionally download the raw data from Yahoo Finance.
     
    Return a data frame for that ticker, including the original price.
    
    Column names: Price, LogReturns, Parity (1 if non-negative, 0 if negative)
    """
    if data is None:
        data = yf.download(ticker, start=start, end=end)["Close"]

    # Compute the log returns
    log_returns = np.log(data[ticker] / data[ticker].shift(1))
    
    # Compute the price parity (first entry is NaN)
    parity = (log_returns >= 0)
    parity.iloc[0] = np.nan

    # Compute the volatility (absolute log-returns)
    vol = np.abs(log_returns)

    # Create a data frame
    df = pd.DataFrame({
        'Price': data[ticker],
        'LogReturns': log_returns,
        'Parity': parity,
        'Volatility': vol,
    })

    if save_dir is not None:
        df.to_csv(os.path.join(save_dir, f'{ticker}.csv'))
    
    return df


def plot_log_returns_and_parity(ticker_data, ticker_name=None, title=None, save_dir=None):
    """Plot the log-returns and use the marker color to indicate the parity.
    
    Note that first entry is NaN.
    """
    if ticker_name is None:
        ticker_name = ticker_data.columns[0]
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    ax.plot(ticker_data['LogReturns'], marker='.', linestyle='-', color='gray', alpha=0.7)
    ax.scatter(ticker_data.index, ticker_data['LogReturns'], 
               c=ticker_data['Parity'], cmap='coolwarm', s=20, alpha=0.9)
    ax.set_title(f'Log-Returns and Parity: {ticker_name}' if title is None else title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Log-Returns')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)

    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, f'LogReturns{ticker_name}.png'), dpi=350)
        fig.savefig(os.path.join(save_dir, f'LogReturns{ticker_name}.pdf'))

    return fig, ax


def plot_volatility(ticker_data, threshold=None, ticker_name=None, title=None, save_dir=None):
    """Plot the squared log-returns (volatility).
    
    Note that first entry is NaN.
    """
    if ticker_name is None:
        ticker_name = ticker_data.columns[0]
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    ax.plot(ticker_data['Volatility'], marker='.', linestyle='-', color='gray', alpha=0.7)
    ax.set_title(f'Volatility: {ticker_name}' if title is None else title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility')

    if threshold is not None:
        ax.axhline(threshold, color='black', linestyle='--', linewidth=1, alpha=0.7)
        # highlight only the points above the threshold in red
        mask = ticker_data['Volatility'] > threshold
        ax.scatter(ticker_data[mask].index, ticker_data[mask]['Volatility'], 
                   c='red', s=20, alpha=0.9)

    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, f'Volatility{ticker_name}.png'), dpi=350)
        fig.savefig(os.path.join(save_dir, f'Volatility{ticker_name}.pdf'))

    return fig, ax
