import numpy as np
import time
import datetime as dt
from experiments import *
from data import dataset

data_dir = '../Data/'

def main():
    for dt in dataset:
        #df = preprocess(dt)
        #tokenize(df,dt)
        run_bandit_round(dt,'scratch')
        run_bandit_arms(dt,'scratch')
        run_bandit_round(dt,'pretrained')
        #run_bandit_arms(dt,'pretrained')
        

if __name__ == '__main__':
    main()
