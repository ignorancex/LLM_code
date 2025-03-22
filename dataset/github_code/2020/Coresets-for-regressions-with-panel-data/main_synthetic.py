from l2_regression import *
from coreset import *
from generate import *
import pandas as pd
import time
import evaluation as eva
import pickle
import sys

##########################################
# output an excel that records statistics
def arrays_write_to_excel(eps, evaluate_glsek, evaluate_uniform1, size_glsek, construction_time_glsek, opt_time_glsek, opt_time, k, type):
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
    if type == 0:
        df.to_csv('result' + str(k) + '_synthetic_gaussian.csv')
    else:
        df.to_csv('result' + str(k) + '_synthetic_cauchy.csv')
    return df

#############################################
# record empirical errors by an excel for drawing boxplot
def arrays_to_boxplot(eps, glse_0, uniform_0, glse_1, uniform_1):
    l = []
    T = len(glse_0[0])
    for i in range(len(eps)):
        for t in range(T):
            l.append([eps[i], glse_0[i][t], uniform_0[i][t], glse_1[i][t], uniform_1[i][t]])
    arr = np.asarray(l)
    df = pd.DataFrame(arr, columns = ['eps', 'CGLSE (Gaussian)', 'Uni (Gaussian)', 'CGLSE (Cauchy)', 'Uni (Cauchy)'])
    df.to_csv('emp' + str(k) + '_synthetic.csv')
    return

#####################################
if __name__ == "__main__":
    times = int(sys.argv[1])
# glse
    N = 500
    T = 500
    d = 11
    k = 1
    q = 1
    lam = 0.2

    start = time.time()
    panel0,panel1 = generate_panel(N, T, 1, q, d, lam)
    np.save("synthetic_gaussian", panel0)
    np.save("synthetic_cauchy", panel1)

    print("Gaussian error")
    type = 0
    # compute statistics
    eps, evaluate_glse_0, evaluate_uniform1_0, size_glse, construction_time_glse, opt_time_glse, opt_time = eva.evaluate_glse(panel0, q, lam, times)
    # record by excel
    arrays_write_to_excel(eps, evaluate_glse_0, evaluate_uniform1_0, size_glse, construction_time_glse, opt_time_glse, opt_time, k, type)

    print("Cauchy error")
    type = 1
    # compute statistics
    eps, evaluate_glse_1, evaluate_uniform1_1, size_glse, construction_time_glse, opt_time_glse, opt_time = eva.evaluate_glse(
        panel1, q, lam, times)
    # record by excel
    arrays_write_to_excel(eps, evaluate_glse_1, evaluate_uniform1_1, size_glse, construction_time_glse, opt_time_glse,
                          opt_time, k, type)

    # draw boxplot of empirical errors
    arrays_to_boxplot(eps, evaluate_glse_0, evaluate_uniform1_0, evaluate_glse_1, evaluate_uniform1_1)

    print(time.time() - start)

# # glsek
#     N = 500
#     T = 500
#     d = 11
#     k = 3
#     q = 1
#     lam = 0.2
#
#     start = time.time()
#     panelk = generate_panel(N, T, k, q, d, lam)
#     with open("panelk", "wb") as f:
#         pickle.dump(panelk, f)
#     panelk = panelk / N
#
#     eps, evaluate_glsek, evaluate_uniform1, size_glsek, construction_time_glsek, opt_time_glsek, opt_time = eva.evaluate_glsek(panelk, k, q, lam, times)
#     arrays_write_to_excel(eps, evaluate_glsek, evaluate_uniform1, size_glsek, construction_time_glsek, opt_time_glsek, opt_time, k)
#
#     print(time.time() - start)

