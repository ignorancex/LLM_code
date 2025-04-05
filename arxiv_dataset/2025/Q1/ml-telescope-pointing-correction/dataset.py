import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
import random
random_seed = 412069413
np.random.seed(random_seed)

class PrepareData():
    def __init__(self,
                df,
                parameter_keys = None,
                feature_list = None,
                target_key = None,
                run_number = None,
                path_results = './Results/None/',
                split_on_days = True,
                ) -> None:

        targets = {
            'az'   : ['Off_Az'],
            'el'   : ['Off_El'],
        }

        self.params = None
        self.target = targets[target_key]
        self.n_targets = len(self.target)
        self.dataset_key, self.timeperiod_key = parameter_keys
        self.run_number = run_number 
        self.path_results = path_results
        
        if 'rx' in df.columns:
            le = LabelEncoder()
            df['rx'] = le.fit_transform(df['rx'])
        
        self.df = df

        if split_on_days:
            self.train_test_split_days()
        else:
            self.df_train, self.df_test = train_test_split(df, test_size=0.4, random_state=random_seed)

        self.X_train = self.df_train[feature_list]
        self.X_test = self.df_test[feature_list]
        self.y_train = self.df_train[self.target]
        self.y_test = self.df_test[self.target]


        self.rms_offset = np.sqrt(np.mean(self.y_test**2)).get(0)
        self.rms_offset_optimal_correction = np.sqrt( np.mean( (df[self.target] - df[self.target].mean())**2 ) ).get(0)


    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
 
    def train_test_split_days(self):
        """
        This function splits a dataset by day into separate training and validation sets, ensuring no overlap between days in the two sets.
        """
        
        df = self.df
        df.insert(0, 'day', df['date'].dt.date)
        dfs = [df[df['day'] == day] for day in df['day'].unique()]
        random.Random(random_seed).shuffle(dfs)

        test_size = 0.33
        train_size = 1 - test_size
        n_days     = len(dfs)

        dfs_train = dfs[:int(train_size * n_days)]
        dfs_test  = dfs[int(train_size * n_days):]

        self.df_train = pd.concat(dfs_train)
        self.df_test  = pd.concat(dfs_test)

        self.df_train = self.df_train.loc[: , self.df_train.columns != 'day']
        self.df_test  = self.df_test.loc [: , self.df_test.columns  != 'day']
        train_days = len(self.df_train)
        test_days  = len(self.df_test)
        print(f'Train size: {train_days} | Test set: {test_days} | Test size: {test_days/(train_days+test_days):.2f}')

