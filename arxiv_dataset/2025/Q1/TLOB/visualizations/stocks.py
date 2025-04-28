import os
import pandas as pd
import numpy as np
import torch
import constants as cst
from torch.utils import data
import matplotlib.pyplot as plt


class LOBSTERDataBuilder:
    def __init__(
        self,
        stocks,
        data_dir,
        date_trading_days,
        split_rates,
    ):
        self.n_lob_levels = cst.N_LOB_LEVELS
        self.data_dir = data_dir
        self.date_trading_days = date_trading_days
        self.stocks = stocks
        self.split_rates = split_rates
        self.prepare_save_datasets()


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

            # Calculate mid-price and plot it
            self._plot_mid_price(self.dataframes[0][1], stock)
            self._compute_and_save_statistics(self.dataframes[0][1], self.dataframes[0][0], path_where_to_save, stock)
            

    def _plot_mid_price(self, orderbook_df, stock):
        # Calculate the mid-price
        best_bid = orderbook_df["buy1"]
        best_ask = orderbook_df["sell1"]
        mid_price = (best_bid + best_ask) / 2
        date_range = pd.date_range(start="01/02/2015", end="01/30/2015", periods=len(mid_price))

        # Plot the mid-price
        plt.figure(figsize=(10, 6))
        plt.plot(date_range, mid_price, label=f'{stock} Mid-Price')
        plt.xlabel('Time')
        plt.ylabel('Mid-Price')
        plt.title(f'{stock} Mid-Price')
        plt.legend()
        # Set x-axis labels
        plt.xticks(rotation=45)
        plt.gca().set_xticks([date_range[0], date_range[-1]])
        plt.gca().set_xticklabels(['01/02/2015', '01/30/2015'])

        # Save the plot
        plot_filename = os.path.join(os.getcwd(), f'{stock}_mid_price_plot.pdf')
        plt.savefig(plot_filename)
        plt.close()

    def _compute_and_save_statistics(self, orderbook_df, message_df, save_path, stock):
        # Calculate the mid-price
        best_bid = orderbook_df["buy1"]
        best_ask = orderbook_df["sell1"]
        spread = best_ask - best_bid
        avg_spread = spread.mean()
        liquidity = orderbook_df.iloc[:, 1::2].sum(axis=1).mean()
        avg_liquidity = liquidity.mean()
        self.open_mid_prices = np.array(self.open_mid_prices)
        self.daily_returns = (self.open_mid_prices[1:] - self.open_mid_prices[:-1]) / self.open_mid_prices[:-1]
        # Calculate statistics
        daily_return_std = np.std(self.daily_returns)
        daily_volume_std = np.std(self.daily_volumes)
        daily_return_mean = np.mean(self.daily_returns)
        daily_volume_mean = np.mean(self.daily_volumes)

        # Save statistics to a file
        stats = {
            'daily_return_std': daily_return_std,
            'daily_volume_std': daily_volume_std,
            'daily_return_mean': daily_return_mean,
            'daily_volume_mean': daily_volume_mean,
            'average_spread': avg_spread,
            'avgerage_spread_std': spread.std(),
            'average_liquidity': avg_liquidity,
            'average_liquidity_std': liquidity.std(),
        }
        stats_df = pd.DataFrame([stats])
        stats_filename = os.path.join(save_path, f'{stock}_statistics.csv')
        stats_df.to_csv(stats_filename, index=False)


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




    def _create_dataframes_splitted(self, path, split_days, COLUMNS_NAMES):
        # iterate over files in the data directory of self.STOCK_NAME
        self.open_mid_prices = []
        self.daily_volumes = []
        for i, filename in enumerate(sorted(os.listdir(path))):
            f = os.path.join(path, filename)
            print(f)
            if os.path.isfile(f):
                # then we create the df for the training set
                if i < split_days[0]:
                    if (i % 2) == 0:
                        if i == 0:
                            train_messages = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                            self.daily_volumes.append(train_messages["size"].sum())
                        else:
                            train_message = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                            self.daily_volumes.append(train_message["size"].sum())
                    else:
                        if i == 1:
                            train_orderbooks = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            self.open_mid_prices.append(train_orderbooks["sell1"][0] + train_orderbooks["buy1"][0] / 20000)
                            if (len(train_orderbooks) != len(train_messages)):
                                raise ValueError("train_orderbook length is different than train_messages")
                        else:
                            train_orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            self.open_mid_prices.append(train_orderbook["sell1"][0] + train_orderbook["buy1"][0] / 20000)
                            train_messages = pd.concat([train_messages, train_message], axis=0)
                            train_orderbooks = pd.concat([train_orderbooks, train_orderbook], axis=0)

                elif split_days[0] <= i < split_days[1]:  # then we are creating the df for the validation set
                    if (i % 2) == 0:
                        if (i == split_days[0]):
                            self.dataframes.append([train_messages, train_orderbooks])
                            val_messages = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                            self.daily_volumes.append(val_messages["size"].sum())
                        else:
                            val_message = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                            self.daily_volumes.append(val_message["size"].sum())
                    else:
                        if i == split_days[0] + 1:
                            val_orderbooks = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            self.open_mid_prices.append(val_orderbooks["sell1"][0] + val_orderbooks["buy1"][0] / 20000)
                            if (len(val_orderbooks) != len(val_messages)):
                                raise ValueError("val_orderbook length is different than val_messages")
                        else:
                            val_orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            self.open_mid_prices.append(val_orderbook["sell1"][0] + val_orderbook["buy1"][0] / 20000)
                            val_messages = pd.concat([val_messages, val_message], axis=0)
                            val_orderbooks = pd.concat([val_orderbooks, val_orderbook], axis=0)

                else:  # then we are creating the df for the test set

                    if (i % 2) == 0:
                        if (i == split_days[1]):
                            self.dataframes.append([val_messages, val_orderbooks])
                            test_messages = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                            self.daily_volumes.append(test_messages["size"].sum())
                        else:
                            test_message = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                            self.daily_volumes.append(test_message["size"].sum())
                    else:
                        if i == split_days[1] + 1:
                            test_orderbooks = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            self.open_mid_prices.append(test_orderbooks["sell1"][0] + test_orderbooks["buy1"][0] / 20000)
                            if (len(test_orderbooks) != len(test_messages)):
                                raise ValueError("test_orderbook length is different than test_messages")
                        else:
                            test_orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            self.open_mid_prices.append(test_orderbook["sell1"][0] + test_orderbook["buy1"][0] / 20000)
                            test_messages = pd.concat([test_messages, test_message], axis=0)
                            test_orderbooks = pd.concat([test_orderbooks, test_orderbook], axis=0)
            else:
                raise ValueError("File {} is not a file".format(f))
        self.dataframes.append([test_messages, test_orderbooks])




    def _split_days(self):
        train = int(self.num_trading_days * self.split_rates[0])
        val = int(self.num_trading_days * self.split_rates[1]) + train
        test = int(self.num_trading_days * self.split_rates[2]) + val
        print(f"There are {train} days for training, {val - train} days for validation and {test - val} days for testing")
        return [train, val, test]
    
    
data_builder = LOBSTERDataBuilder(
            stocks=["TSLA"],
            data_dir=cst.DATA_DIR,
            date_trading_days=cst.DATE_TRADING_DAYS,
            split_rates=cst.SPLIT_RATES,
        )