import numpy as np
from hipmk import HIPMK_Kernel



def run_hipmk(train_X, test_X, data_stats):

    param_value = None  # use default: log2(num of inst) + 1
    hipmk_krn = HIPMK_Kernel(param_value, data_stats)
    hipmk_krn.set_nbins(param_value)
    train, test = hipmk_krn.build_model(train_X, test_X)  # this does the pre-processing step
    print("- Sim: Train")
    sim_train = hipmk_krn.transform(train)
    print("- Sim: Train/Test")
    sim_test = hipmk_krn.transform(train,test)

    return sim_train, sim_test.T