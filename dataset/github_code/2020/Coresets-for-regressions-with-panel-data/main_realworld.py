from l2_regression import *
from coreset import *
from generate import *
import pandas as pd
import time
import evaluation as eva
import numpy as np
import pickle
import sys

##########################################
# output an excel that records statistics
def arrays_write_to_excel(eps, evaluate_glsek, evaluate_uniform1, size_glsek, construction_time_glsek, opt_time_glsek, opt_time, k):
    l = []
    for i in range(len(eps)):
        l.append([eps[i], np.max(evaluate_glsek[i]), np.max(evaluate_uniform1[i]), np.mean(evaluate_glsek[i]), np.std(evaluate_glsek[i]), \
                  np.sqrt(np.mean(np.square(evaluate_glsek[i]))), np.mean(evaluate_uniform1[i]), \
                  np.std(evaluate_uniform1[i]), np.sqrt(np.mean(np.square(evaluate_uniform1[i]))), size_glsek[i], \
                  construction_time_glsek[i], opt_time_glsek[i], opt_time[i]])

    arr = np.asarray(l)
    df = pd.DataFrame(arr, columns=['eps', 'max emp_c', 'max emp_u1', 'avg emp_c', 'std emp_c', \
                                     'RMSE_c', 'avg emp_u1','std emp_u1', 'RMSE_u1', 'size', 'T_C', \
                                    'T_C+T_S', \
                                    'T_X'])
    df.to_csv('result' + str(k) + '_realworld.csv')
    return df

#############################################
# record empirical errors by an excel for drawing boxplot
def arrays_to_boxplot(eps, glsek, uniform1):
    l = []
    T = len(glsek[0])
    for i in range(len(eps)):
        for t in range(T):
            l.append([eps[i], glsek[i][t], uniform1[i][t]])
    arr = np.asarray(l)
    df = pd.DataFrame(arr, columns = ['eps', 'CGLSE', 'Uni'])
    df.to_csv('emp' + str(k) + '_realworld.csv')
    return

#####################################
if __name__ == "__main__":
    times = int(sys.argv[1])

# glse
    k = 1
    q = 1
    lam = 0.2

    start = time.time()
    panel = np.load('realworld.npy')

    # compute statistics
    eps, evaluate_glse, evaluate_uniform1, size_glse, construction_time_glse, opt_time_glse, opt_time = eva.evaluate_glse(panel, q, lam, times)

    # record by excel
    arrays_write_to_excel(eps, evaluate_glse, evaluate_uniform1, size_glse, construction_time_glse, opt_time_glse, opt_time, k)

    # draw boxplot of empirical errors
    arrays_to_boxplot(eps, evaluate_glse, evaluate_uniform1)

    print(time.time() - start)


# ##############################
# # glsek
#     k = 3
#     q = 1
#     lam = 0.2
#
#     start = time.time()
#     panelk = np.load('realworld.npy')
#
#     eps, evaluate_glsek, evaluate_uniform1, size_glsek, construction_time_glsek, opt_time_glsek, opt_time = eva.evaluate_glsek(panelk, k, q, lam, times)
#     arrays_write_to_excel(eps, evaluate_glsek, evaluate_uniform1, size_glsek, construction_time_glsek, opt_time_glsek, opt_time, k)
#
#     print(time.time() - start)



