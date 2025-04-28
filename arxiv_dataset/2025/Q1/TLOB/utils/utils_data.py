import pandas as pd
import numpy as np
import os

import torch
import pandas
import constants as cst


def z_score_orderbook(data, mean_size=None, mean_prices=None, std_size=None, std_prices=None):
    """ DONE: remember to use the mean/std of the training set, to z-normalize the test set. """
    if (mean_size is None) or (std_size is None):
        mean_size = data.iloc[:, 1::2].stack().mean()
        std_size = data.iloc[:, 1::2].stack().std()

    #do the same thing for prices
    if (mean_prices is None) or (std_prices is None):
        mean_prices = data.iloc[:, 0::2].stack().mean() #price
        std_prices = data.iloc[:, 0::2].stack().std() #price

    # apply the z score to the original data using .loc with explicit float cast
    price_cols = data.columns[0::2]
    size_cols = data.columns[1::2]

    #apply the z score to the original data
    for col in size_cols:
        data[col] = data[col].astype("float64")
        data[col] = (data[col] - mean_size) / std_size

    for col in price_cols:
        data[col] = data[col].astype("float64")
        data[col] = (data[col] - mean_prices) / std_prices

    # check if there are null values, then raise value error
    if data.isnull().values.any():
        raise ValueError("data contains null value")

    return data, mean_size, mean_prices, std_size,  std_prices


def normalize_messages(data, mean_size=None, mean_prices=None, std_size=None,  std_prices=None, mean_time=None, std_time=None, mean_depth=None, std_depth=None):
        
    #apply z score to prices and size column
    if (mean_size is None) or (std_size is None):
        mean_size = data["size"].mean()
        std_size = data["size"].std()

    if (mean_prices is None) or (std_prices is None):
        mean_prices = data["price"].mean()
        std_prices = data["price"].std()

    if (mean_time is None) or (std_time is None):
        mean_time = data["time"].mean()
        std_time = data["time"].std()

    if (mean_depth is None) or (std_depth is None):
        mean_depth = data["depth"].mean()
        std_depth = data["depth"].std()

    #apply the z score to the original data
    data["time"] = (data["time"] - mean_time) / std_time
    data["size"] = (data["size"] - mean_size) / std_size
    data["price"] = (data["price"] - mean_prices) / std_prices
    data["depth"] = (data["depth"] - mean_depth) / std_depth
    # check if there are null values, then raise value error
    if data.isnull().values.any():
        raise ValueError("data contains null value")

    data["event_type"] = data["event_type"]-1.0
    data["event_type"] = data["event_type"].replace(2, 1)
    data["event_type"] = data["event_type"].replace(3, 2)
    # order_type = 0 -> limit order
    # order_type = 1 -> cancel order
    # order_type = 2 -> market order
    return data, mean_size, mean_prices, std_size,  std_prices, mean_time, std_time, mean_depth, std_depth


def reset_indexes(dataframes):
    # reset the indexes of the messages and orderbooks
    dataframes[0] = dataframes[0].reset_index(drop=True)
    dataframes[1] = dataframes[1].reset_index(drop=True)
    return dataframes


def sampling_quantity(dataframes, quantity=1000):
    messages_df, orderbook_df = dataframes[0], dataframes[1]
    
    # Calculate cumulative sum and create boolean mask
    cumsum = messages_df['size'].cumsum()
    sample_mask = (cumsum % quantity < messages_df['size'])
    
    # Get indices where we need to sample
    sampled_indices = messages_df.index[sample_mask].tolist()
    
    # Update both dataframes efficiently using loc
    messages_df = messages_df.loc[sampled_indices].reset_index(drop=True)
    orderbook_df = orderbook_df.loc[sampled_indices].reset_index(drop=True)
    
    return [messages_df, orderbook_df]


def sampling_time(dataframes, time):
    # Convert the time column to datetime format if it's not already
    dataframes[0]['time'] = pd.to_datetime(dataframes[0]['time'], unit='s')

    # Resample the messages dataframe to get data at every second
    resampled_messages = dataframes[0].set_index('time').resample(time).first().dropna().reset_index()

    # Resample the orderbook dataframe to get data at every second
    resampled_orderbook = dataframes[1].set_index(dataframes[0]['time']).resample(time).first().dropna().reset_index(drop=True)

    # Update the dataframes with the resampled data
    dataframes[0] = resampled_messages
    
    # Transform the time column to seconds
    dataframes[0]['time'] = dataframes[0]['time'].dt.second + dataframes[0]['time'].dt.minute * 60 + dataframes[0]['time'].dt.hour * 3600 + dataframes[0]['time'].dt.microsecond / 1e6
    dataframes[1] = resampled_orderbook

    return dataframes


def preprocess_data(dataframes, n_lob_levels, sampling_type, time=None, quantity=None):
    dataframes = reset_indexes(dataframes)
    # take only the first n_lob_levels levels of the orderbook and drop the others
    dataframes[1] = dataframes[1].iloc[:, :n_lob_levels * cst.LEN_LEVEL]

    # take the indexes of the dataframes that are of type 
    # 2 (partial deletion), 5 (execution of a hidden limit order), 
    # 6 (cross trade), 7 (trading halt) and drop it
    indexes_to_drop = dataframes[0][dataframes[0]["event_type"].isin([2, 5, 6, 7])].index
    dataframes[0] = dataframes[0].drop(indexes_to_drop)
    dataframes[1] = dataframes[1].drop(indexes_to_drop)

    dataframes = reset_indexes(dataframes)

    # sample the dataframes according to the sampling type
    if sampling_type == "time":
        dataframes = sampling_time(dataframes, time)
    elif sampling_type == "quantity":
        dataframes = sampling_quantity(dataframes, quantity)
        
    dataframes = reset_indexes(dataframes)
    
    # drop index column in messages
    dataframes[0] = dataframes[0].drop(columns=["order_id"])

    # do the difference of time row per row in messages and subsitute the values with the differences
    # Store the initial value of the "time" column
    first_time = dataframes[0]["time"].values[0]
    # Calculate the difference using diff
    dataframes[0]["time"] = dataframes[0]["time"].diff()
    # Set the first value directly
    dataframes[0].iat[0, dataframes[0].columns.get_loc("time")] = first_time - 34200
        
    # add depth column to messages
    dataframes[0]["depth"] = 0

    # we compute the depth of the orders with respect to the orderbook
    # Extract necessary columns
    prices = dataframes[0]["price"].values
    directions = dataframes[0]["direction"].values
    event_types = dataframes[0]["event_type"].values
    bid_sides = dataframes[1].iloc[:, 2::4].values
    ask_sides = dataframes[1].iloc[:, 0::4].values
    
    # Initialize depth array
    depths = np.zeros(dataframes[0].shape[0], dtype=int)

    # Compute the depth of the orders with respect to the orderbook
    for j in range(1, len(prices)):
        order_price = prices[j]
        direction = directions[j]
        event_type = event_types[j]
        
        index = j if event_type == 1 else j - 1
        
        if direction == 1:
            bid_price = bid_sides[index, 0]
            depth = (bid_price - order_price) // 100
        else:
            ask_price = ask_sides[index, 0]
            depth = (order_price - ask_price) // 100
        
        depths[j] = max(depth, 0)
    
    # Assign the computed depths back to the DataFrame
    dataframes[0]["depth"] = depths
        
    # we eliminate the first row of every dataframe because we can't deduce the depth
    dataframes[0] = dataframes[0].iloc[1:, :]
    dataframes[1] = dataframes[1].iloc[1:, :]
    dataframes = reset_indexes(dataframes)
    
    dataframes[0]["direction"] = dataframes[0]["direction"] * dataframes[0]["event_type"].apply(
        lambda x: -1 if x == 4 else 1)
        
    return dataframes[1], dataframes[0]
 

def unnormalize(x, mean, std):
    return x * std + mean


def one_hot_encoding_type(data):
    encoded_data = torch.zeros(data.shape[0], data.shape[1] + 2, dtype=torch.float32)
    encoded_data[:, 0] = data[:, 0]
    # encoding order type
    one_hot_order_type = torch.nn.functional.one_hot((data[:, 1]).to(torch.int64), num_classes=3).to(
        torch.float32)
    encoded_data[:, 1:4] = one_hot_order_type
    encoded_data[:, 4:] = data[:, 2:]
    return encoded_data


def tanh_encoding_type(data):
    data[:, 1] = torch.where(data[:, 1] == 1.0, 2.0, torch.where(data[:, 1] == 2.0, 1.0, data[:, 1]))
    data[:, 1] = data[:, 1] - 1
    return data


def to_sparse_representation(lob, n_levels):
    if not isinstance(lob, np.ndarray):
        lob = np.array(lob)
    sparse_lob = np.zeros(n_levels * 2)
    for j in range(lob.shape[0] // 2):
        if j % 2 == 0:
            ask_price = lob[0]
            current_ask_price = lob[j*2]
            depth = (current_ask_price - ask_price) // 100
            if depth < n_levels and int(lob[j*2]) != 0:
                sparse_lob[2*int(depth)] = lob[j*2+1]
        else:
            bid_price = lob[2]
            current_bid_price = lob[j*2]
            depth = (bid_price - current_bid_price) // 100
            if depth < n_levels and int(lob[j*2]) != 0:
                sparse_lob[2*int(depth)+1] = lob[j*2+1]
    return sparse_lob
