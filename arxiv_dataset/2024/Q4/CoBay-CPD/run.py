import numpy as np
import pandas as pd
import copy
from scipy.misc import derivative
from scipy.special import logsumexp
import math
from scipy.stats import beta
from scipy.special import expit
from scipy.special import psi
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import poisson
from scipy.stats import bernoulli
from scipy.stats import uniform
from scipy.stats import multinomial
from scipy.stats import multivariate_normal
from scipy.stats import invgauss
from scipy.stats import dirichlet
from numpy.polynomial import legendre
import matplotlib.pyplot as plt
import scipy.integrate as si
from scipy import stats, optimize
import matplotlib.pyplot as plt
from scipy import stats, optimize
from IPython.display import display, Math, Latex, clear_output
import multiprocessing
from functools import partial
from pypolyagamma import PyPolyaGamma
import time
from model.CoBay_CPD import CoBay_CPD
from model.SMCPD import SMCPD
from model.SVCPD import SVCPD
from model.SVCPDIn import SVOCPDIn

# read data
df = pd.read_csv(r"./DATA/generate_gibbs_2cp1.csv")
# df = pd.read_csv(r"/home/luxiaoling_students/zhangzeyue/Hawkes/Gibbs_revise/yalices/t_d=15")
ttall = df.value.values
ttall -= ttall[0]
tt = ttall[:]

start_time = time.time() 

cobaycpd = SVCPD(tt)
cobaycpd.apply()

print(cobaycpd.changepoints)
mse = np.sum((tt - cobaycpd.pred_mean)**2) / len(tt)
mae = np.sum((tt - cobaycpd.pred_mean)) / len(tt)
print("mse=", mse)
print("mae=", mae)

end_time = time.time()

elapsed_time = end_time - start_time
print(f"time=", elapsed_time)
