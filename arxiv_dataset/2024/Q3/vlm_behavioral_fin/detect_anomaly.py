import requests
import json
import time
import argparse
import os
import io
import datetime as dt
import numpy as np
import pandas as pd
from tqdm import tqdm


def detect_recency_bias(ticker, stock_file, eps_dir, window=5):
    assert window > 1, "Window size must be greater than 1, otherwise, it only contains itself."
    dataframe = stock_file
    comp_stock = dataframe.xs(ticker, axis=1, level=1, drop_level=True)
    comp_stock.columns.name = None
    comp_stock.reset_index(inplace=True)
    files = os.listdir(eps_dir)
    file = None
    for f in files:
        if ticker == f.split('.')[0].split('-')[1]:
            file = f
            break
    if not file:
        raise FileNotFoundError(f"No EPS data found for {ticker}")
    with open(os.path.join(eps_dir,file), "r") as f:
        eps_dict = json.load(f)
    f.close()
    assert 'quarterlyEarnings' in eps_dict, f"'quarterlyEarnings' not found for {ticker} in {file}"
    quarterly_eps = eps_dict['quarterlyEarnings']
    quarterly_eps_df = pd.DataFrame(quarterly_eps)
    quarterly_eps_df = quarterly_eps_df.sort_values(by='reportedDate', ascending=False)
    
    comp_stock.loc[:, 'Date'] = pd.to_datetime(comp_stock['Date'])
    quarterly_eps_df.loc[:, 'reportedDate'] = pd.to_datetime(quarterly_eps_df['reportedDate'])
    # EPS report within stock time range
    quarterly_eps_df = quarterly_eps_df[quarterly_eps_df['reportedDate'].between(comp_stock['Date'].min(), comp_stock['Date'].max())]
    
    def _up_or_down(date):
        before = comp_stock.loc[comp_stock['Date'].between(date-pd.Timedelta(days=7), date), 'Close'].mean()
        after = comp_stock.loc[comp_stock['Date'].between(date+pd.Timedelta(days=1), date+pd.Timedelta(days=8)), 'Close'].mean()
        
        return int(after >= before)
    
    bias_time, bias, gt = [], [], []
    for i in tqdm(range(len(quarterly_eps_df) - window + 1), leave=False, desc="Fetching: "):
        
        if quarterly_eps_df.iloc[i]['surprise'] == 'None':
            continue
        
        surprise = float(quarterly_eps_df.iloc[i]['surprise']) >= 0
        gt_up_or_down = _up_or_down(quarterly_eps_df.iloc[i]['reportedDate'])      
        last_up_or_down = None
        bias_diff = 0
        bias_same = 0
        no_surprise_report = False
        threshold = min(0.8, (window - 2) / (window - 1))
        for j in range(i+1, i+window):
            if quarterly_eps_df.iloc[j]['surprise'] == 'None':
                no_surprise_report = True
                break
            
            if (float(quarterly_eps_df.iloc[j]['surprise']) >= 0) == surprise:
                if last_up_or_down is None:
                    last_up_or_down = _up_or_down(quarterly_eps_df.iloc[j]['reportedDate'])
                    bias_same += 1
                else:
                    if _up_or_down(quarterly_eps_df.iloc[j]['reportedDate']) != last_up_or_down:
                        bias_diff += 1
                    else:
                        bias_same += 1
        
        if bias_same == 0 or bias_diff / (bias_diff + bias_same) < threshold or no_surprise_report:
            continue
        
        cur_time = ((quarterly_eps_df.iloc[i+window-1]['reportedDate']-pd.Timedelta(days=30)).strftime('%Y-%m-%d'), quarterly_eps_df.iloc[i]['reportedDate'].strftime('%Y-%m-%d'))
        if comp_stock.loc[cur_time[0]:cur_time[1], ['Low', 'High']].isna().any().any():
            continue
        
        bias_time.append(cur_time)
        bias.append(last_up_or_down)
        gt.append(gt_up_or_down)
    
    return bias_time, bias, gt


def detect_authority_bias(ticker, stock_file, eps_dir, window=5):
    assert window > 1, "Window size must be greater than 1, otherwise, it only contains itself."
    dataframe = stock_file
    comp_stock = dataframe.xs(ticker, axis=1, level=1, drop_level=True)
    comp_stock.columns.name = None
    comp_stock.reset_index(inplace=True)
    files = os.listdir(eps_dir)
    file = None
    for f in files:
        if ticker == f.split('.')[0].split('-')[1]:
            file = f
            break
    if not file:
        raise FileNotFoundError(f"No EPS data found for {ticker}")
    with open(os.path.join(eps_dir,file), "r") as f:
        eps_dict = json.load(f)
    f.close()
    assert 'quarterlyEarnings' in eps_dict, f"'quarterlyEarnings' not found for {ticker} in {file}"
    quarterly_eps = eps_dict['quarterlyEarnings']
    quarterly_eps_df = pd.DataFrame(quarterly_eps)
    quarterly_eps_df = quarterly_eps_df.sort_values(by='reportedDate', ascending=False)
    
    comp_stock.loc[:, 'Date'] = pd.to_datetime(comp_stock['Date'])
    quarterly_eps_df.loc[:, 'reportedDate'] = pd.to_datetime(quarterly_eps_df['reportedDate'])
    # EPS report within stock time range
    quarterly_eps_df = quarterly_eps_df[quarterly_eps_df['reportedDate'].between(comp_stock['Date'].min(), comp_stock['Date'].max())]
    
    def _up_or_down(date):
        before = comp_stock.loc[comp_stock['Date'].between(date-pd.Timedelta(days=7), date), 'Close'].mean()
        after = comp_stock.loc[comp_stock['Date'].between(date+pd.Timedelta(days=1), date+pd.Timedelta(days=8)), 'Close'].mean()
        
        return int(after >= before)
    
    bias_time, bias, gt = [], [], []
    for i in tqdm(range(len(quarterly_eps_df) - window + 1), leave=False, desc="Fetching: "):
        
        if quarterly_eps_df.iloc[i]['surprise'] == 'None':
            continue
        
        surprise = float(quarterly_eps_df.iloc[i]['surprise']) >= 0      
        gt_up_or_down = _up_or_down(quarterly_eps_df.iloc[i]['reportedDate'])
        up_count = 0
        down_count = 0
        no_surprise_report = False
        for j in range(i+1, i+window):
            if quarterly_eps_df.iloc[j]['surprise'] == 'None':
                no_surprise_report = True
                break
            
            if (float(quarterly_eps_df.iloc[j]['surprise']) >= 0) == surprise:
                if _up_or_down(quarterly_eps_df.iloc[j]['reportedDate']):
                    up_count += 1
                else:
                    down_count += 1
        
        if up_count == down_count:
            continue
        elif up_count > down_count:
            majority_count = up_count
            majority_up_or_down = 1
        else:
            majority_count = down_count
            majority_up_or_down = 0
        
        if (up_count + down_count) / (window - 1) < min(0.8, (window-2)/(window-1)) or majority_count / (up_count + down_count) < 0.8 or no_surprise_report:
            continue
        
        cur_time = ((quarterly_eps_df.iloc[i+window-1]['reportedDate']-pd.Timedelta(days=30)).strftime('%Y-%m-%d'), quarterly_eps_df.iloc[i]['reportedDate'].strftime('%Y-%m-%d'))
        if comp_stock.loc[cur_time[0]:cur_time[1], ['Low', 'High']].isna().any().any():
            continue
        
        bias_time.append(cur_time)
        bias.append(1 ^ majority_up_or_down)
        gt.append(gt_up_or_down)
    
    return bias_time, bias, gt

    
if __name__ == '__main__':
    bias_data, bias, gt= detect_recency_bias("AAPL", pd.read_csv("data/stock_history.csv", header=[0, 1], index_col=0), "data/eps_history/", window=4)
    # bias_data, bias, gt= detect_authority_bias("AAPL", pd.read_csv("data/stock_history.csv", header=[0, 1], index_col=0), "data/eps_history/", window=20)
    print(len(bias_data))
    print(bias_data)
    print(len(bias))
    print(bias)
    print(len(gt))
    print(gt)