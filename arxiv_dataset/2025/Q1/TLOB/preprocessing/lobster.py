import os
from utils.utils_data import z_score_orderbook, normalize_messages, preprocess_data, one_hot_encoding_type
import pandas as pd
import numpy as np
import torch
import constants as cst
from torch.utils import data


def lobster_load(path, all_features, len_smooth, h, seq_size):
    set = np.load(path)
    if h == 10:
        tmp = 5
    if h == 20:
        tmp = 4
    elif h == 50:
        tmp = 3
    elif h == 100:
        tmp = 2
    elif h == 200:
        tmp = 1
    labels = set[seq_size-len_smooth:, -tmp]
    labels = labels[np.isfinite(labels)]
    labels = torch.from_numpy(labels).long()
    if all_features:
        input = set[:, cst.LEN_ORDER:cst.LEN_ORDER + 40]
        orders = set[:, :cst.LEN_ORDER]
        input = torch.from_numpy(input).float()
        orders = torch.from_numpy(orders).float()
        input = torch.cat((input, orders), dim=1)
    else:
        input = set[:, cst.LEN_ORDER:cst.LEN_ORDER + 40]
        input = torch.from_numpy(input).float()

    return input, labels

    
def labeling(X, len, h, stock):
    # X is the orderbook
    # len is the time window smoothing length
    # h is the prediction horizon
    [N, D] = X.shape
    
    if h < len:
        len = h
    # Calculate previous and future mid-prices for all relevant indices
    previous_ask_prices = np.lib.stride_tricks.sliding_window_view(X[:, 0], window_shape=len)[:-h]
    previous_bid_prices = np.lib.stride_tricks.sliding_window_view(X[:, 2], window_shape=len)[:-h]
    future_ask_prices = np.lib.stride_tricks.sliding_window_view(X[:, 0], window_shape=len)[h:]
    future_bid_prices = np.lib.stride_tricks.sliding_window_view(X[:, 2], window_shape=len)[h:]

    previous_mid_prices = (previous_ask_prices + previous_bid_prices) / 2
    future_mid_prices = (future_ask_prices + future_bid_prices) / 2

    previous_mid_prices = np.mean(previous_mid_prices, axis=1)
    future_mid_prices = np.mean(future_mid_prices, axis=1)

    # Compute percentage change
    percentage_change = (future_mid_prices - previous_mid_prices) / previous_mid_prices
    
    # alpha is the average percentage change of the stock
    alpha = np.abs(percentage_change).mean() / 2
    
    # alpha is the average spread of the stock in percentage of the mid-price
    #alpha = (X[:, 0] - X[:, 2]).mean() / ((X[:, 0] + X[:, 2]) / 2).mean()
        
    print(f"Alpha: {alpha}")
    labels = np.where(percentage_change < -alpha, 2, np.where(percentage_change > alpha, 0, 1))
    print(f"Number of labels: {np.unique(labels, return_counts=True)}")
    print(f"Percentage of labels: {np.unique(labels, return_counts=True)[1] / labels.shape[0]}")
    return labels


class LOBSTERDataBuilder:
    def __init__(
        self,
        stocks,
        data_dir,
        date_trading_days,
        split_rates,
        sampling_type,
        sampling_time,
        sampling_quantity,
    ):
        self.n_lob_levels = cst.N_LOB_LEVELS
        self.data_dir = data_dir
        self.date_trading_days = date_trading_days
        self.stocks = stocks
        self.split_rates = split_rates
        
        self.sampling_type = sampling_type
        self.sampling_time = sampling_time
        self.sampling_quantity = sampling_quantity


    def prepare_save_datasets(self):
        for i in range(len(self.stocks)):
            stock = self.stocks[i]
            path = "{}/{}/{}_{}_{}".format(
                self.data_dir,
                stock,
                stock,
                self.date_trading_days[0],
                self.date_trading_days[1],
            )
            self.dataframes = []
            self._prepare_dataframes(path, stock)

            path_where_to_save = "{}/{}".format(
                self.data_dir,
                stock,
            )

            self.train_input = pd.concat(self.dataframes[0], axis=1).values
            self.val_input = pd.concat(self.dataframes[1], axis=1).values
            self.test_input = pd.concat(self.dataframes[2], axis=1).values
            self.train_set = pd.concat([pd.DataFrame(self.train_input), pd.DataFrame(self.train_labels_horizons)], axis=1).values
            self.val_set = pd.concat([pd.DataFrame(self.val_input), pd.DataFrame(self.val_labels_horizons)], axis=1).values
            self.test_set = pd.concat([pd.DataFrame(self.test_input), pd.DataFrame(self.test_labels_horizons)], axis=1).values
            self._save(path_where_to_save)


    def _prepare_dataframes(self, path, stock):
        COLUMNS_NAMES = {"orderbook": ["sell1", "vsell1", "buy1", "vbuy1",
                                       "sell2", "vsell2", "buy2", "vbuy2",
                                       "sell3", "vsell3", "buy3", "vbuy3",
                                       "sell4", "vsell4", "buy4", "vbuy4",
                                       "sell5", "vsell5", "buy5", "vbuy5",
                                       "sell6", "vsell6", "buy6", "vbuy6",
                                       "sell7", "vsell7", "buy7", "vbuy7",
                                       "sell8", "vsell8", "buy8", "vbuy8",
                                       "sell9", "vsell9", "buy9", "vbuy9",
                                       "sell10", "vsell10", "buy10", "vbuy10"],
                         "message": ["time", "event_type", "order_id", "size", "price", "direction"]}
        self.num_trading_days = len(os.listdir(path))//2
        split_days = self._split_days()
        split_days = [i * 2 for i in split_days]
        self._create_dataframes_splitted(path, split_days, COLUMNS_NAMES)
        # divide all the price, both of lob and messages, by 10000, to have dollars as unit
        for i in range(len(self.dataframes)):
            self.dataframes[i][0]["price"] = self.dataframes[i][0]["price"] / 10000
            self.dataframes[i][1].loc[:, ::2] /= 10000
        train_input = self.dataframes[0][1].values
        val_input = self.dataframes[1][1].values
        test_input = self.dataframes[2][1].values
        #create a dataframe for the labels
        for i in range(len(cst.LOBSTER_HORIZONS)):
            if i == 0:
                train_labels = labeling(train_input, cst.LEN_SMOOTH, cst.LOBSTER_HORIZONS[i], stock)
                val_labels = labeling(val_input, cst.LEN_SMOOTH, cst.LOBSTER_HORIZONS[i], stock)
                test_labels = labeling(test_input, cst.LEN_SMOOTH, cst.LOBSTER_HORIZONS[i], stock)
                train_labels = np.concatenate([train_labels, np.full(shape=(train_input.shape[0] - train_labels.shape[0]), fill_value=np.inf)])
                val_labels = np.concatenate([val_labels, np.full(shape=(val_input.shape[0] - val_labels.shape[0]), fill_value=np.inf)])
                test_labels = np.concatenate([test_labels, np.full(shape=(test_input.shape[0] - test_labels.shape[0]), fill_value=np.inf)])
                self.train_labels_horizons = pd.DataFrame(train_labels, columns=["label_h{}".format(cst.LOBSTER_HORIZONS[i])])
                self.val_labels_horizons = pd.DataFrame(val_labels, columns=["label_h{}".format(cst.LOBSTER_HORIZONS[i])])
                self.test_labels_horizons = pd.DataFrame(test_labels, columns=["label_h{}".format(cst.LOBSTER_HORIZONS[i])])
            else:
                train_labels = labeling(train_input, cst.LEN_SMOOTH, cst.LOBSTER_HORIZONS[i], stock)
                val_labels = labeling(val_input, cst.LEN_SMOOTH, cst.LOBSTER_HORIZONS[i], stock)
                test_labels = labeling(test_input, cst.LEN_SMOOTH, cst.LOBSTER_HORIZONS[i], stock)
                train_labels = np.concatenate([train_labels, np.full(shape=(train_input.shape[0] - train_labels.shape[0]), fill_value=np.inf)])
                val_labels = np.concatenate([val_labels, np.full(shape=(val_input.shape[0] - val_labels.shape[0]), fill_value=np.inf)])
                test_labels = np.concatenate([test_labels, np.full(shape=(test_input.shape[0] - test_labels.shape[0]), fill_value=np.inf)])
                self.train_labels_horizons["label_h{}".format(cst.LOBSTER_HORIZONS[i])] = train_labels
                self.val_labels_horizons["label_h{}".format(cst.LOBSTER_HORIZONS[i])] = val_labels
                self.test_labels_horizons["label_h{}".format(cst.LOBSTER_HORIZONS[i])] = test_labels
        
        #self._sparse_representation()
        
        # to conclude the preprocessing we normalize the dataframes
        self._normalize_dataframes()


    def _sparse_representation(self):
        tick_size = 0.01
        for i in range(len(self.dataframes)):
            dense_repr = self.dataframes[i][1].values
            sparse_repr = np.zeros((dense_repr.shape[0], dense_repr.shape[1] + 1))
            for row in range(dense_repr.shape[0]):
                sparse_pos_ask = 0
                sparse_pos_bid = 0
                mid_price = (dense_repr[row][0] + dense_repr[row][2]) / 2
                sparse_repr[row][-1] = mid_price
                for col in range(0, dense_repr.shape[1], 2):
                    if col == 0:
                        start_ask = dense_repr[row][col]
                    elif col == 2:
                        start_bid = dense_repr[row][col]
                    elif col % 4 == 0:
                        if sparse_pos_ask < (sparse_repr.shape[1]) - 1 / 2:
                            actual_ask = dense_repr[row][col]
                            for level in range(0, actual_ask-start_ask, -tick_size):
                                if sparse_pos_ask < (sparse_repr.shape[1]) - 1 / 2:
                                    if level == actual_ask - start_ask - tick_size:
                                        sparse_repr[row][sparse_pos_ask] = dense_repr[row][col+1]
                                    else:
                                        sparse_repr[row][sparse_pos_ask] = 0
                                    sparse_pos_ask += 1
                                else:
                                    break
                            start_ask = actual_ask
                        else:
                            continue
                    elif col % 4 == 2:
                        if sparse_pos_bid < (sparse_repr.shape[1]) - 1 / 2:
                            actual_bid = dense_repr[row][col]
                            for level in range(0, start_bid-actual_bid, -tick_size):
                                if sparse_pos_bid < (sparse_repr.shape[1]) - 1 / 2:
                                    if level == start_bid - actual_bid - tick_size:
                                        sparse_repr[row][sparse_pos_ask] = dense_repr[row][col+1]
                                    else:
                                        sparse_repr[row][sparse_pos_ask] = 0
                                    sparse_pos_bid += 1
                                else:
                                    break
                            start_bid = actual_bid
                        else:
                            continue
                

    def _create_dataframes_splitted(self, path, split_days, COLUMNS_NAMES):
        # iterate over files in the data directory of self.STOCK_NAME
        total_shape = 0
        for i, filename in enumerate(sorted(os.listdir(path))):
            f = os.path.join(path, filename)
            print(f)
            if os.path.isfile(f):
                # then we create the df for the training set
                if i < split_days[0]:
                    if (i % 2) == 0:
                        if i == 0:
                            train_messages = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                        else:
                            train_message = pd.read_csv(f, names=COLUMNS_NAMES["message"])

                    else:
                        if i == 1:
                            train_orderbooks = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            total_shape += train_orderbooks.shape[0]
                            train_orderbooks, train_messages = preprocess_data([train_messages, train_orderbooks], self.n_lob_levels, self.sampling_type, self.sampling_time, self.sampling_quantity)
                            if (len(train_orderbooks) != len(train_messages)):
                                raise ValueError("train_orderbook length is different than train_messages")
                        else:
                            train_orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            total_shape += train_orderbook.shape[0]
                            train_orderbook, train_message = preprocess_data([train_message, train_orderbook], self.n_lob_levels, self.sampling_type, self.sampling_time, self.sampling_quantity)
                            train_messages = pd.concat([train_messages, train_message], axis=0)
                            train_orderbooks = pd.concat([train_orderbooks, train_orderbook], axis=0)

                elif split_days[0] <= i < split_days[1]:  # then we are creating the df for the validation set
                    if (i % 2) == 0:
                        if (i == split_days[0]):
                            self.dataframes.append([train_messages, train_orderbooks])
                            val_messages = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                        else:
                            val_message = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                    else:
                        if i == split_days[0] + 1:
                            val_orderbooks = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            total_shape += val_orderbooks.shape[0]
                            val_orderbooks, val_messages = preprocess_data([val_messages, val_orderbooks], self.n_lob_levels, self.sampling_type, self.sampling_time, self.sampling_quantity)
                            if (len(val_orderbooks) != len(val_messages)):
                                raise ValueError("val_orderbook length is different than val_messages")
                        else:
                            val_orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            total_shape += val_orderbook.shape[0]
                            val_orderbook, val_message = preprocess_data([val_message, val_orderbook], self.n_lob_levels, self.sampling_type, self.sampling_time, self.sampling_quantity)
                            val_messages = pd.concat([val_messages, val_message], axis=0)
                            val_orderbooks = pd.concat([val_orderbooks, val_orderbook], axis=0)

                else:  # then we are creating the df for the test set

                    if (i % 2) == 0:
                        if (i == split_days[1]):
                            self.dataframes.append([val_messages, val_orderbooks])
                            test_messages = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                        else:
                            test_message = pd.read_csv(f, names=COLUMNS_NAMES["message"])

                    else:
                        if i == split_days[1] + 1:
                            test_orderbooks = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            test_orderbooks, test_messages = preprocess_data([test_messages, test_orderbooks], self.n_lob_levels, self.sampling_type, self.sampling_time, self.sampling_quantity)
                            if (len(test_orderbooks) != len(test_messages)):
                                raise ValueError("test_orderbook length is different than test_messages")
                        else:
                            test_orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            test_orderbook, test_message = preprocess_data([test_message, test_orderbook], self.n_lob_levels, self.sampling_type, self.sampling_time, self.sampling_quantity)
                            test_messages = pd.concat([test_messages, test_message], axis=0)
                            test_orderbooks = pd.concat([test_orderbooks, test_orderbook], axis=0)
            else:
                raise ValueError("File {} is not a file".format(f))
        self.dataframes.append([test_messages, test_orderbooks])
        print(f"Total shape of the orderbooks is {total_shape}")


    def _normalize_dataframes(self):
        #apply z score to orderbooks
        for i in range(len(self.dataframes)):
            if (i == 0):
                self.dataframes[i][1], mean_size, mean_prices, std_size, std_prices = z_score_orderbook(self.dataframes[i][1])
            else:
                self.dataframes[i][1], _, _, _, _ = z_score_orderbook(self.dataframes[i][1], mean_size, mean_prices, std_size, std_prices)

        #apply z-score to size and prices of messages with the statistics of the train set
        for i in range(len(self.dataframes)):
            if (i == 0):
                self.dataframes[i][0], mean_size, mean_prices, std_size, std_prices, mean_time, std_time, mean_depth, std_depth = normalize_messages(self.dataframes[i][0])
            else:
                self.dataframes[i][0], _, _, _, _, _, _, _, _ = normalize_messages(self.dataframes[i][0], mean_size, mean_prices, std_size, std_prices, mean_time, std_time, mean_depth, std_depth)

    def _save(self, path_where_to_save):
        np.save(path_where_to_save + "/train.npy", self.train_set)
        np.save(path_where_to_save + "/val.npy", self.val_set)
        np.save(path_where_to_save + "/test.npy", self.test_set)


    def _split_days(self):
        train = int(self.num_trading_days * self.split_rates[0])
        val = int(self.num_trading_days * self.split_rates[1]) + train
        test = int(self.num_trading_days * self.split_rates[2]) + val
        print(f"There are {train} days for training, {val - train} days for validation and {test - val} days for testing")
        return [train, val, test]