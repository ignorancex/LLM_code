################################################################################################### 
#                               Load modules
###################################################################################################
import numpy as np
import pandas as pd

from keras.datasets import mnist, cifar10
from sklearn.datasets import load_diabetes

from sklearn.model_selection import train_test_split
from utils import load_train_test_images, load_images, reshape_images, convert_data, plot_sample




################################################################################################### 
#                               Helper function
###################################################################################################
def load_dataset(opts):
    obj = Dataset(opts)
    X_train, X_test, y_train, y_test, image_params = obj()
    dataset = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
                'width': image_params[0], 'height': image_params[1], 'channels': image_params[2]}

    return dataset



################################################################################################### 
#                               Datasets
###################################################################################################

class Dataset():
    def __init__(self, opts):
        
        self.tabular_names = ['diabetes', 'compas', 'seeds', 'htru2']
        self.image_names = ['mnist', 'cifar10', 'oxford_flowers', 'chd2r']
        
        self.opts = opts
        if self.opts["name"] not in self.tabular_names + self.image_names:
            raise NotImplementedError("For other types of datasets one must reimplement this class.")

        self.y, self.y_train, self.y_test = 3 * [None]
        self.features, self.target = 2 * [None]
        self.width, self.height, self.channels = 3 * [None]
        




    def get_data(self):
        if self.opts["name"] == "diabetes":
            self.X, self.y = load_diabetes(return_X_y = True)

            self.features = ['feat_' + str(i) for i in range(self.X.shape[1])]
            self.target = 'feat_' + str(len(self.features))

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
                                                                                    test_size = self.opts["test_size"], 
                                                                                    random_state = self.opts["random_state"])

        
        

        elif self.opts["name"] == "compas":
            self.df_train = pd.read_csv(self.opts["train_path"] + '/' + self.opts["train_filename"] + '.csv')
            self.df_test = pd.read_csv(self.opts["test_path"] + '/' + self.opts["test_filename"] + '.csv')

            self.df_train = self.df_train.dropna(axis = 0)
            self.df_test = self.df_test.dropna(axis = 0)

            self.target = 'ground_truth'
            self.features = [col for col in list(self.df_train.columns) if col != self.target]
            
            self.X_train, self.y_train = self.df_train[self.features], self.df_train[self.target]
            self.X_test, self.y_test = self.df_test[self.features], self.df_test[self.target]



        elif self.opts["name"] == "seeds":
            self.df = pd.read_csv(self.opts["train_path"] + '/' + self.opts["train_filename"] + '.csv')
            self.df = self.df.dropna(axis = 0)

            self.target = "target"
            self.features = [col for col in list(self.df.columns) if col != self.target]
            self.X, self.y = self.df[self.features], self.df[self.target]
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
                                                                                    test_size = self.opts["test_size"], 
                                                                                    random_state = self.opts["random_state"])
                



        elif self.opts["name"] == "htru2":
            self.df = pd.read_csv(self.opts["train_path"] + '/' + self.opts["train_filename"] + '.csv')
            self.df = self.df.dropna(axis = 0)
            
            self.df.columns = ['feat_' + str(i) for i in range(self.df.shape[1])]
            self.target = self.df.columns[-1]
            self.features = [col for col in list(self.df.columns) if col != self.target]
            self.X, self.y = self.df[self.features], self.df[self.target]

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
                                                                                    test_size = self.opts["test_size"], 
                                                                                    random_state = self.opts["random_state"])
            


        elif self.opts["name"] == "mnist":
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test
            
            self.width, self.height = X_train.shape[1:]
            self.channels = 1

            self.X_train, self.X_test = reshape_images(self.X_train, self.X_test, self.channels)
            
            
            split_results = train_test_split(self.X_train, self.y_train, test_size = self.X_train.shape[0] - self.opts["N_train"], 
                                                                                        random_state = self.opts["random_state"], 
                                                                                        stratify = self.y_train)
            self.X_train, self.y_train = split_results[0], split_results[2]

            test_images = np.zeros((self.opts["N_test"], self.width * self.height))
            test_labels = np.zeros((self.opts["N_test"], ))
            
            for label in range(self.opts["N_test"]):
                index = np.where(y_test == label)[0][0]
                test_images[label] = self.X_test[index]
                test_labels[label] = self.y_test[index]

            self.X_test, self.y_test = test_images, test_labels

            

        elif self.opts["name"] == "cifar10":
            (X_train, y_train), (X_test, y_test) = cifar10.load_data()
            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test
            self.width, self.height, self.channels = X_train.shape[1:]

            self.X_train, self.X_test = reshape_images(self.X_train, self.X_test, self.channels)

            split_results = train_test_split(self.X_train, self.y_train, test_size = self.X_train.shape[0] - self.opts["N_train"], 
                                                                                        random_state = self.opts["random_state"], 
                                                                                        stratify = self.y_train)
            self.X_train, self.y_train = split_results[0], split_results[2]

            test_images = np.zeros((self.opts["N_test"], self.width * self.height, self.channels))
            test_labels = np.zeros((self.opts["N_test"], ))

            for label in range(self.opts["N_test"]):
                index = np.where(y_test == label)[0][0]
                test_images[label] = self.X_test[index]
                test_labels[label] = self.y_test[index]

            self.X_test, self.y_test = test_images, test_labels



        elif self.opts["name"] == "oxford_flowers":
            self.X, _, w, h, c = load_train_test_images(self.opts)
            self.X_train, self.X_test = train_test_split(self.X, test_size = self.opts["N_test"], 
                                                                            random_state = self.opts["random_state"])
                                                                            
            self.width, self.height, self.channels = w, h, c
            self.X_train, self.X_test = reshape_images(self.X_train, self.X_test, self.channels)



        elif self.opts["name"] == "chd2r":
            self.X, w, h, c = load_images(self.opts)
            self.width, self.height, self.channels = w, h, c

            self.X_train, self.X_test = train_test_split(self.X, test_size = self.opts["N_test"], 
                                                                            random_state = self.opts["random_state"])
            self.X_train, self.X_test = reshape_images(self.X_train, self.X_test, self.channels)
        
        



    def get_infos(self):
        
        print('NAME: ', self.opts["name"].upper(), '\n')
        
        head_no_rows = min(10, (self.X_train.shape)[0])
        head_no_cols = min(10, (self.X_train.shape)[1])


        if isinstance(self.X_train, pd.DataFrame):
            X_train_sample = self.X_train.iloc[:head_no_rows, range(head_no_cols)]
        else:
            if self.opts["name"] in ["cifar10", "oxford_flowers", "chd2r"]:
                X_train_sample = self.X_train[:head_no_rows, :head_no_cols, 0]
            elif self.opts["name"] == "mnist":
                X_train_sample = self.X_train[:head_no_rows, :head_no_cols]
            else:
                X_train_sample = self.X_train[:head_no_rows, :head_no_cols]


        print('X_train shape = ', self.X_train.shape, '\n')
        print('X_test shape = ', self.X_test.shape, '\n\n')


        if self.y_train is not None and self.y_test is not None:
            y_train_unique = len(np.unique(self.y_train))
            y_test_unique = len(np.unique(self.y_test))

            print('y_train unique number of targets = ', y_train_unique, '\n')
            print('y_test unique number of targets = ', y_test_unique, '\n\n')

            if isinstance(self.y_train, pd.DataFrame):
                y_train_sample = self.y_train.iloc[:head_no_rows]
            else:
                y_train_sample = self.y_train[:head_no_rows]
 


        if self.opts["name"] in ["diabetes", "seeds", "compas", "seeds", "htru2"]:
            print('feature_names = ', self.features, '\n\n')
            if self.y_train is not None and self.y_test is not None:
                print('target = ', self.target, '\n\n')
        else:
            print('[width, height, channels] = (', self.width, ', ', self.height, ', ', self.channels, ')\n\n')


        print('X_train sample element :\n', X_train_sample, '\n\n')
        if self.y_train is not None and self.y_test is not None:
            print('y_train sample element :\n', y_train_sample, '\n\n')
        


        if self.opts["name"] in ["mnist", "cifar10", "oxford_flowers", "chd2r"]:
            color = True if self.opts["name"] != "mnist" else False
            plot_sample(self.X_test, self.width, self.height, self.channels, color)




    def __call__(self):
        self.get_data()
        
        if self.opts["verbose"]:
            self.get_infos()

        self.X_train, self.X_test, self.y_train, self.y_test = convert_data(self.X_train, self.X_test, self.y_train, self.y_test)        
        
        image_params = [self.width, self.height, self.channels]
        return self.X_train, self.X_test, self.y_train, self.y_test, image_params

        