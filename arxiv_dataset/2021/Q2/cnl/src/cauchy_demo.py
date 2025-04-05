import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
#from cauchy import *

from cw import *
from spn_c2 import *

def draw_loss_heatmap(size = 32, num_shot = 1, true_a=5, true_sigma=1,\
min_var = 0.9, max_var = 1.1, min_a = 4, max_a=6,\
resolution=11,version=1):
    true_diag_A = true_a*np.ones(size)
    #for n in range(size/2):
    #    true_diag_A[n] = 1
    true_A = np.diag(true_diag_A)
    evs_list = []
    for n in range(num_shot):
        evs= np.linalg.eigh(signal_plus_noise(size, true_A,true_sigma, COMPLEX=True))[0]
        evs_list += evs.tolist()
    #plt.figure()
    #plt.hist(evs_list, bins=100, normed=True, label="empirical eigenvalues")
    #plt.show()


    sample =   np.array(evs_list)
    sc = SemiCircular()

    min_loss = np.inf

    data = np.zeros((resolution,resolution))
    base = np.identity(size)

    if version == 1:
        row_labels = np.linspace(min_var, max_var, resolution)
        column_labels = np.linspace(min_a, max_a, resolution)

        i = 0
        for sigma in row_labels:
            j=0
            for a in column_labels:
                diag_A = a*base
                loss = sc.loss(diag_A, sigma, sample)
                if loss < min_loss:
                    i_min = i
                    j_min = j
                    v_min = np.copy(sigma)
                    a_min = np.copy(a)
                    min_loss = np.copy(loss)
                data[i][j] = loss
                logging.info("sigma={}, a={} : loss = {}".format(sigma,a, loss))
                j+=1
            i+=1
        x_min = v_min
        logging.info("minimum loss = {} at sigma= {}, a={}".format(min_loss, v_min, a_min    ))

    elif version == 2:
        sigma = true_sigma
        column_labels = np.linspace(min_a, max_a, resolution)
        row_labels = column_labels
        low_labels = column_labels
        base_2 = np.identity(size)
        for n in range(size/2):
            base_2[n][n] = 0
        base_1 = base - base_2
        i = 0
        for b in row_labels:
            j=0
            for a in column_labels:
                diag_A = a*base_1 + b*base_2
                loss = sc.loss(diag_A, sigma, sample)
                if loss < min_loss:
                    i_min = i
                    j_min = j
                    b_min = b
                    a_min = a
                    min_loss = np.copy(loss)
                data[i][j] = loss
                logging.info("b={},a={} : loss = {}".format(b,a, loss))
                j+=1
            i+=1
        x_min = b_min
        logging.info("minimum loss = {} at b= {}, a={}".format(min_loss, b_min, a_min    ))



    #fig, ax = plt.subplots()
    mask = np.zeros((resolution,resolution))
    mask[i_min][j_min] = 1
    ax=sns.heatmap(data,mask=mask)
    if version == 1:
        x_name = "sigma"
    elif version ==2:
        x_namae = "b"
    ax.set_title("The heatmap of the loss: true sigma={}, true_a={} \n [masked tile : ({}={},a={}) with the minimum loss  {}]".format(true_sigma, true_a,x_name, x_min, a_min, min_loss))
    if version == 1:
        ax.set_ylabel("sigma")
    elif version== 2:
        ax.set_ylabel("b")
    ax.set_xlabel("a")
    ax.set_xticklabels(column_labels, minor=False, rotation=45)
    #ax.set_yticklabels("auto")
    ax.set_yticklabels(row_labels[::-1], minor=False, rotation=45)
    plt.legend()
    jobname = "version-{}_var-{}_a-{}_{}x{}_{}-shot".format(version,true_sigma,true_a,size,size,num_shot)
    plt.savefig('images/loss_heatmap/hmp_{}.png'.format(jobname))

    #plt.show()

def plot_diag_A():
    result=[ 1.86118048,  1.86135701,  1.86785427,  1.89205575,  1.90232415,  1.94137182,
      1.96693471,  1.99411409,  2.0109729,   2.01464403,  2.03368412 , 2.04589832,
      2.0519821,   2.06155915,  2.07205636,  2.10047752, 2.15251164 , 2.48935786,
      4.6733921,   4.67420988,  4.79150905,  4.80594921,  4.85441517 , 4.90415685,
      4.92234306,  4.94433349,  5.03670975,  5.08139888,  5.18532338 , 5.23251321,
      5.25166202,  5.28116253]
    size = 32
    true_diag_A = 5*np.ones(size)
    for i in range(size/2):
        true_diag_A[i] = 2
    sq_sample = [ 1.1491804,   1.40600594,  1.48221607,  1.57600981,  1.72191391,  1.82681936,
      1.86881396,  1.99760292,  2.09487744,  2.12232706,  2.19218798, 2.36462248,\
      2.44905798,  2.49397598,  2.60231496,  2.80231778,  4.36791194,  4.45367608,\
      4.56954924,  4.70934905,  4.82818918,  4.95984569,  4.98511903,  5.11656876,\
      5.21940259,  5.31377114,  5.37850193,  5.4989789,   5.57869409 , 5.77553586,\
      5.91743402,  6.01945669]

    sc = SemiCircular()
    A = np.diag(true_diag_A)

    x = np.linspace(0.01, 40, 201) ## Modify
    sigma = 1                ## as you want
    y = sc.square_density(x,A, sigma)
    plt.figure()
    plt.plot(x,y, label="Truth")


    sigma = 1
    sample = np.array(sq_sample)**2
    true_loss = sc.loss(A, 1, sample)
    logging.info("true_loss={}".format(true_loss))

    A = np.diag(result)
    y = sc.square_density(x,A, sigma)
    plt.plot(x,y, label="Result")
    plt.legend()
    plt.savefig("images/diag_A/2-5_distr.png")
    plt.show()
    result_loss = sc.loss(A, 1, sample)
    logging.info("result_loss={}".format(result_loss))

    plt.figure()
    plt.plot(true_diag_A, label="truth")
    plt.plot(sq_sample, label="sq_sample")
    plt.plot(result, label="reslut")
    plt.savefig("images/diag_A/2-5.png")
    plt.show()


plot_diag_A()
