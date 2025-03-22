import numpy as np
import scipy.io as sio
import pandas as pd
import pygmo as pg
import matplotlib.pyplot as plt

def hv_ploter(n_iter, path, flag):
    hyper_val_all = np.array([])
    ParetoFront_tmp = np.array([])
    ParetoFront_tmp = ParetoFront_tmp.reshape((-1, 2))
    latent_all = np.array([])

    for i in range(n_iter):
        if flag == 0:
            mat = sio.loadmat(path % (i))
            ParetoFront = mat['ParetoFront']

        elif flag == 1:
            mat = sio.loadmat(path % (i+1))
            ParetoFront = mat['pareto_iter']

        ParetoFront_tmp = np.concatenate((ParetoFront_tmp, ParetoFront), axis=0)


        solutions_df = pd.DataFrame(index=range(int(ParetoFront_tmp.shape[0])),
                                    columns=["f1", "f2"]).astype(float)
        for j in range(int(ParetoFront_tmp.shape[0])):
            solutions_df.loc[j, "f1"] = ParetoFront_tmp[j, 0]
            solutions_df.loc[j, "f2"] = ParetoFront_tmp[j, 1]

        hyp = pg.hypervolume(solutions_df[["f1", "f2"]].values)
        hyp_value = hyp.compute([11, 11]) / np.prod([11, 11])
        hyper_val_all = np.concatenate((hyper_val_all, np.array([hyp_value])), axis=0)
    return hyper_val_all, ParetoFront_tmp