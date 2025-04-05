import numpy as np
import scipy as sp

from matrix_util import *
from random_matrices import *
import matplotlib.pyplot as plt
from timer import Timer
import os
import time
import logging
from datetime import datetime

from argparse import ArgumentParser
from train_fde import *


import env_logger

def options(logger=None):
    desc   = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)
    parser = ArgumentParser(description = desc)

    # options
    parser.add_argument('-d', '--dim',
                        type     = int,
                        dest     = 'dim',
                        required = False,
                        default  =  50,
                        help     = "column (default: %(default)s)")
    parser.add_argument('-p', '--p_dim',
                        type     = int,
                        dest     = 'p_dim',
                        required = False,
                        default  =  50,
                        help     = "row (default: %(default)s)")
    parser.add_argument('-m', '--minibatch',
                        type     = int,
                        dest     = 'minibatch',
                        required = False,
                        default  =  1,
                        help     = "minibatch_size (default: %(default)s)")
    parser.add_argument('-j', '--jobname',
                        type     = str,
                        dest     = 'jobname',
                        required = False,
                        default  =  50,
                        help     = "min_singular (default: %(default)s)")
    parser.add_argument('-nt', '--num_test',
                        type     = int,
                        dest     = 'num_test',
                        required = False,
                        default  =  10,
                        help     = "Number of tests (default: %(default)s)")
    parser.add_argument('-me', '--max_epoch',
                        type     = int,
                        dest     = 'max_epoch',
                        required = False,
                        default  =  400,
                        help     = "max_epoch (default: %(default)s)")
    parser.add_argument('-dpi', '--dpi',
                        type     = int,
                        dest     = 'dpi',
                        required = False,
                        default  =  300,
                        help     = "Resolution of figures (default: %(default)s)")
    parser.add_argument('-ext', '--ext',
                        type     = str,
                        dest     = 'ext',
                        required = False,
                        default  =  "pdf",
                        help     = "image (default: %(default)s)")

    return parser.parse_args()

opt =options()


i_dim = min(opt.dim, opt.p_dim)
i_p_dim = max(opt.dim, opt.p_dim)
jobname = opt.jobname
i_dpi = opt.dpi





def _mean_and_std(results):
    m = np.mean(results, axis = 0)
    v = np.mean( (results - m)**2, axis=0)
    if len(results) == 1:
        std = 0*v
    else:
        v *= len(results) / (len(results) -1)
        std = sp.sqrt(v)
    return m,std


def test_optimize(\
    base_scale ,dim_cauchy_vec, \
    base_lr=1e-4 ,  minibatch_size=1,  max_epoch=20,\
    jobname="test_optimize_cw",\
    min_singular=0,
    dim=i_dim,
     zero_dim=16,
     COMPLEX = False
    ):
    dirname = "../images/{}".format(jobname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    num_sample =1
    #to file
    file_log = logging.FileHandler('../log/{}.log'.format(jobname), 'w')
    file_log.setLevel(logging.INFO)
    file_log.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(file_log)
    #logging.getLogger().setLevel(logging.INFO)
    #param = 2 + np.random.rand(size)*3
    #param = np.sort(param)
    #"""
    p_dim = dim
    ###marhcnko 1 + \sqrt{p/d}
    MP_ratio = p_dim/dim
    #high_singular = 1./ ( 1 + sp.sqrt(MP_ratio))
    high_singular = 0.1
    assert min_singular < high_singular
    param = np.random.uniform(low=min_singular, high=high_singular, size = p_dim)
    #param = 2*np.random_sample(dim)
    for i in range(zero_dim):
        param[i] = 0.
    #"""
    param = np.sort(param)
    logging.info( "zero_dim={}".format(zero_dim) )
    logging.info( "min_singular={}".format(min_singular) )
    logging.info( "truth=\n{}".format(np.array(param)))

    ### For generate sample, we do not need scale.
    cw_for_sample = CompoundWishart(dim = dim, p_dim=p_dim, scale = 1e-9)
    cw_for_sample.b = param
    plt.rc("text", usetex=True)

    evs_list = cw_for_sample.ESD(num_shot = num_sample, COMPLEX=COMPLEX)

    if num_sample == 1:
        logging.info( "sample=\n{}".format(np.asarray(evs_list)))




    result= train_cw(dim, p_dim,\
        sample=evs_list,\
        base_scale=base_scale ,\
        dim_cauchy_vec=dim_cauchy_vec,\
        base_lr =base_lr ,\
        minibatch_size=minibatch_size,\
        max_epoch=max_epoch,\
        monitor_validation=True,\
        test_b=param)

    b=result["b"]
    train_loss_array=result["train_loss"]
    val_loss_array=result["val_loss"]
    forward_iter=result["forward_iter"]



    plt.figure()
    plt.plot(np.sort(param), label="Truth")
    plt.plot(np.sort(evs_list), label="Sample")
    plt.plot(np.sort(b), label="Result")
    plt.legend()

    dirname = "../images/val_train_cw"
    if not os.path.exists(dirname):
            os.makedirs(dirname)

    plt.savefig("{}/{}.{}".format(dirname, jobname,opt.ext),dpi=i_dpi)
    plt.clf()
    plt.close()
    logging.getLogger().removeHandler(file_log)

    epoch = int(i_dim/minibatch_size)
    train_loss_array = train_loss_array.reshape([-1,epoch]).mean(axis=1)
    val_loss_array = val_loss_array.reshape([-1,epoch]).mean(axis=1)

    return b, train_loss_array, val_loss_array, forward_iter


def test_scale_balance():
    num_test = opt.num_test
    ### MP-ratio
    #min_singular = - 1./( 1  + np.sqrt(1))
    min_singular = -0.1
    zero_dim = 0
    max_epoch = opt.max_epoch
    #base_lr = 1e-2
    base_lr = 1e-4
    minibatch_size = opt.minibatch
    ### TODO for paper
    base_scale = 1e-1
    list_base_scale =[ 1e-1*base_scale, base_scale, base_scale*10]
    list_dim_cauchy_vec =  [1]
    ### for test
    #list_base_scale =[ 1e-1]





    list_val_loss_array = []
    list_train_loss_array = []
    list_forward_iter = []
    for base_scale in list_base_scale:
        for dim_cauchy_vec in list_dim_cauchy_vec:
            result_b = []
            result_val_loss = []
            result_train_loss = []
            result_forward_iter = []
            for n in range(num_test):
                b, train_loss_array, val_loss_array, forward_iter=test_optimize(\
                base_lr = base_lr,minibatch_size=minibatch_size,\
                 max_epoch=max_epoch,\
                min_singular=min_singular, zero_dim = zero_dim, \
                base_scale=base_scale, dim_cauchy_vec=dim_cauchy_vec)
                result_b.append(b)
                result_val_loss.append(val_loss_array)
                result_train_loss.append(train_loss_array)
                result_forward_iter.append(forward_iter)


            m_b, s_b = _mean_and_std(result_b)
            m_val_loss, s_val_loss = _mean_and_std(result_val_loss)
            m_train_loss, s_train_loss = _mean_and_std(result_train_loss)
            m_forward_iter, s_forward_iter = _mean_and_std(result_forward_iter)

            logging.info("RESULT:base_scale = {}, ncn = {}, val_loss = {},\n  average_b = \n{}".format(\
            base_scale, dim_cauchy_vec, m_val_loss[-1], m_b) )
            list_val_loss_array.append(m_val_loss)
            list_train_loss_array.append(m_train_loss)
            list_forward_iter.append([m_forward_iter, s_forward_iter])







    jobname = "test_cw"
    now = datetime.now()
    temp_jobname = jobname + '_{0:%m%d%H%M}'.format(now)
    dirname = "../images/{}".format(temp_jobname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    setting_log = open("{}/setting.txt".format(dirname), "w")
    setting_log.write("jobname:{}\n".format(temp_jobname))
    setting_log.write("dim:{}\n".format(opt.dim))
    setting_log.write("p_dim:{}\n".format(opt.p_dim))
    setting_log.write("minibatch:{}\n".format(minibatch_size))
    setting_log.write("num_test:{}\n".format(opt.num_test))
    setting_log.write("max_epoch:{}\n".format(max_epoch         ))
    setting_log.write("base_lr:{}\n".format(base_lr           ))
    setting_log.write("list_dim_cauchy_vec:{}\n".format(list_dim_cauchy_vec))
    setting_log.write("list_base_scale:{}\n".format(list_base_scale   ))
    setting_log.write("min_singular:{}\n".format(min_singular ))
    setting_log.close()



    iter_log = open("{}/forward_iter.txt".format(dirname), "w")
    iter_log.write("{}".format(list_forward_iter))
    iter_log.close()

    plt.clf()
    plt.close()
    plt.figure(figsize=(6,4))
    plt.style.use("seaborn-paper")
    plt.rc('text', usetex=True)
    plt.rc('font', family="sans-serif", serif='Helvetica')
    plt.rcParams["font.size"] = 8*2
    linestyles = ["-", "--", "-.", ":"]

    x_axis = np.arange(m_val_loss.shape[0])
    ### plot validation
    n = 0
    for base_scale in list_base_scale:
        for dim_cauchy_vec in list_dim_cauchy_vec:
            ### separate dim_cauchy_vec
            #plt.plot(x_axis,list_val_loss_array[n], label="({0:3.2e}, {1})".format(base_scale, dim_cauchy_vec))
            ### set dim_cauchy_vec == 1
            base_scale = round(base_scale, 2)
            plt.plot(x_axis,list_val_loss_array[n], label="$\gamma={}$".format(base_scale), linestyle=linestyles[n % 4])
            n+=1

    plt.title("CW")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.ylim(0., 0.3)
    plt.legend()
    filename = "{}/test_val.{}".format(dirname,opt.ext)
    logging.info(filename)
    plt.savefig(filename,dpi=i_dpi)
    plt.clf()
    plt.close()


    n = 0
    for base_scale in list_base_scale:
        for dim_cauchy_vec in list_dim_cauchy_vec:
            plt.plot(x_axis,list_train_loss_array[n], label="({0:3.2e}, {1})".format(base_scale, dim_cauchy_vec))
            n+=1
    #plt.title("Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("train loss")
    plt.legend()
    filename = "{}/test_train.{}".format(dirname, opt.ext)
    plt.savefig(filename,dpi=i_dpi)
    plt.clf()
    plt.close()




timer = Timer()
timer.tic()
test_scale_balance()

timer.toc()
logging.info("total_time:{}".format(timer.total_time))
