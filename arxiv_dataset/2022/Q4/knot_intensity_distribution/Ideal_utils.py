import numpy as np
import pickle

IDEAL_KNOTS = ['3_1','4_1','5_1', '5_2', '6_1', '6_2', '6_3', '3_1_#_3_1','3_1_#_3_1_m','3_1_#_4_1']



def load_ideal(K):
    return np.load("ideal-knots/{}.npy".format(K))
    


def load_cores():
    with open( 'ideal-knots/knots_cores.pickle','rb') as handle:
        dic = pickle.load(handle)
    return(dic) 


IDEAL_CORES = load_cores()