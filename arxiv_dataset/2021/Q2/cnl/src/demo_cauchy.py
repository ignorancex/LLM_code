import numpy as np
import scipy as sp

from scipy import stats
from matrix_util import *
from random_matrices import *
import matplotlib.pyplot as plt
from timer import Timer


import time
import logging

#from cauchy import SemiCircular as SC
from spn_c2 import SemiCircular as SC ###for rectanglar


def plot_true_sample_and_model_FDE(min_x, max_x, dim, p_dim, scale, dim_cauchy_vec, max_sigma=0.1,
num_shot=1,\
jobname="true_sample_and_model_FDE"):
    true_sc = SC(dim=dim, p_dim = p_dim, scale=scale)
    true_sc.set_params(np.random.uniform(low=0.5, high=1, size =dim), \
    np.random.uniform(low=0, high =max_sigma, size=1)[0])
    #true_sc.set_params(np.random.uniform(low=0.9, high=1, size =dim), \
    #0.2)
    #true_sc.set_params(np.zeros(dim), 0.2)
    #true_sc.set_params(0.4*np.random.uniform(low=0.9, high=1, size =dim), \
    #0.2)


    model_sc = SC(dim=dim, p_dim = p_dim, scale=scale)
    model_sc.set_params( np.random.uniform(low=0., high=1, size =dim),\
    np.random.uniform(low=0, high =max_sigma, size=1)[0])

    #import pdb; pdb.set_trace()
    #model_sc.set_params( np.random.uniform(low=0.5, high=1, size =dim),\
    #)

    sample = true_sc.ESD(num_shot=num_shot,dim_cauchy_vec=dim_cauchy_vec)
    #sample = true_sc.ESD_symm(num_shot=num_shot,dim_cauchy_vec=dim_cauchy_vec)

    x_array = np.linspace(min_x, max_x, 400)
    true_density = true_sc.density_subordinaiton(x_array)
    #true_density= true_sc.density_subordinaiton_symm(x_array)

    model_density = model_sc.density_subordinaiton(x_array)




    plt.figure()
    plt.rc("text", usetex=True)
    plt.hist(sample, range=(min_x, max_x), bins=100, normed=True, label="sampling from true model \n perturbed by cauchy($0,\gamma$)",color="pink")
    plt.plot(x_array, model_density, label="$\gamma$-slice of init FDE model",)
    plt.plot(x_array, true_density, linestyle="--", label="$\gamma$-slice of FDE of true model", color="red")
    #plt.title("hahaha")

    plt.legend(loc="upper left")
    dirname = "../images/plot_density"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = "{}/{}.png".format(dirname, jobname)
    print ("output: {}".format(filename))
    plt.savefig(filename,dpi=300)

    return sample, model_density

plot_true_sample_and_model_FDE(-4,4, dim=40,p_dim=240,  scale=2e-1, dim_cauchy_vec=100, max_sigma=0.1, num_shot=1)
