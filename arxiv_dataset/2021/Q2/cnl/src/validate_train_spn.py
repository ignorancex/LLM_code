import numpy as np
import scipy as sp


from matrix_util import *
from random_matrices import *
import matplotlib.pyplot as plt
from timer import Timer
import os
import time
import logging
import sys
import json
from tqdm import tqdm,trange
from datetime import datetime
from train_fde import *
from vbmf.vbmf import VBMF2
from vbmf.validate_vbmf_spn import validate_vbmf_spn


from argparse import ArgumentParser
import logging

import env_logger

linestyles = ["-", "--", "-.", ":"]

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
    parser.add_argument('-s', '--base_scale',
                        type     = float,
                        dest     = 'base_scale',
                        required = False,
                        default  =  1e-1,
                        help     = "base scale for CNL (default: %(default)s)")
    parser.add_argument('-lr', '--base_lr',
                        type     = float,
                        dest     = 'base_lr',
                        required = False,
                        default  =  1e-4,
                        help     = "base learning rate (default: %(default)s)")
    parser.add_argument('-j', '--jobname',
                        type     = str,
                        dest     = 'jobname',
                        required = False,
                        default  = "scale_balance",
                        help     = "scale_balance or min_singular (default: %(default)s)")
    parser.add_argument('-v', '--vbmf',
                        type     = bool,
                        dest     = 'vbmf',
                        required = False,
                        default  =  False,
                        help     = "empirical vbmf (default: %(default)s)")
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
    parser.add_argument('-se', '--step_epoch',
                        type     = int,
                        dest     = 'step_epoch',
                        required = False,
                        default  =  -1,
                        help     = "First step epoch (default: %(default)s)")
    parser.add_argument('-dpi', '--dpi',
                        type     = int,
                        dest     = 'dpi',
                        required = False,
                        default  =  150,
                        help     = "Resolution of figures (default: %(default)s)")
    parser.add_argument('-ext', '--ext',
                        type     = str,
                        dest     = 'ext',
                        required = False,
                        default  =  "png",
                        help     = "image (default: %(default)s)")
    parser.add_argument('-tex', '--usetex',
                        type     = int,
                        dest     = 'usetex',
                        required = False,
                        default  =  1,
                        help     = "Usetex for plotting graphs. Set 0 if latex error occurs(default: %(default)s)")

    return parser.parse_args()

opt =options()


i_dim = min(opt.dim, opt.p_dim)
i_p_dim = max(opt.dim, opt.p_dim)
i_vbmf = opt.vbmf

if opt.jobname in ["m", "min", "ms", "min_singular"]:
    jobname = "min_singular"
elif opt.jobname in ["s", "scale", "sb", "scale_balance"]:
    jobname = "scale_balance"
else:
    jobname = "scale_balance"

def test_optimize(\
    base_scale ,dim_cauchy_vec, \
    base_lr=1e-1 ,   max_epoch=20,\
    reg_coef=0,\
    jobname="train_v2",\
    min_singular=0,
    dim=i_dim,p_dim = i_p_dim,\
    zero_dim=0,\
    list_zero_thres=[],
    true_sigma=0.1,\
    COMPLEX=False):
    dirname = "../images/{}".format(jobname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    num_sample =1

    logging.info( "zero_dim={}".format(zero_dim) )
    logging.info( "min_singular={}".format(min_singular) )

    param_mat = random_from_diag(p_dim, dim, zero_dim, min_singular, COMPLEX)
    _, param, _ = np.linalg.svd(param_mat)
    logging.debug( "truth=\n{}".format(np.sort(param)))

    evs_list =[]
    for i  in range(num_sample):
        ### A + sZ
        sample_spn = signal_plus_noise(param_mat, true_sigma, RAW=True)
        U, evs, V = np.linalg.svd(sample_spn)
        evs_list += (evs**2).tolist()
    if num_sample == 1:
        logging.debug( "sqrt(sample)=\n{}".format(sp.sqrt(np.array(evs_list))))
    #mean_param = 2
    sq_sample = np.array(sp.sqrt(evs_list))
    mean_param =np.average(sq_sample)


    if opt.step_epoch > 0:
        lr_policy = "step"
        step_epochs = [opt.step_epoch, max_epoch]

    else:
        lr_policy = "fix"
        step_epochs= []

    #TODO: initial num zero_array

    sample = np.asarray(evs_list)

    result= train_fde_spn(dim, p_dim,  sample,\
        base_scale=base_scale ,\
        dim_cauchy_vec=dim_cauchy_vec,\
        base_lr =base_lr ,\
        minibatch_size=opt.minibatch,\
        max_epoch=max_epoch,\
        reg_coef=reg_coef,\
        lr_policy=lr_policy,\
        step_epochs=step_epochs,\
        monitor_validation=True,\
        test_diag_A=param,\
        test_sigma =true_sigma,\
        test_U= U, test_V=V, Z_FLAG=True,\
        list_zero_thres = list_zero_thres,\
        )

    r_diag_A        =result["diag_A"]
    r_sigma         =result["sigma"]
    train_loss_array=result["train_loss"]
    val_loss_array  =result["val_loss"]
    num_zero_array  =result["num_zero"]
    forward_iter    =result["forward_iter"]


    plt.figure()
    plt.plot(param, label="Truth")
    plt.plot(sq_sample, label="Sample")
    plt.plot(r_diag_A, label="Result")
    plt.legend()
    dirname = "../images/train_v2"
    plt.savefig("{}/{}.{}".format(dirname, jobname, opt.ext),dpi=opt.dpi)
    plt.clf()
    plt.close()
    #logging.getLogger().removeHandler(file_log)


    epoch = int(dim/opt.minibatch)
    train_loss_array = train_loss_array.reshape([-1, epoch]).mean(axis=1)
    val_loss_array = val_loss_array.reshape([-1,epoch]).mean(axis=1)
    num_zero_array = num_zero_array.reshape([-1,epoch, len(list_zero_thres)]).mean(axis=1)

    return r_diag_A, r_sigma, train_loss_array, val_loss_array, num_zero_array, forward_iter


def _mean_and_std(results):
    m = np.mean(results, axis = 0)
    v = np.mean( (results - m)**2, axis=0)
    if len(results) == 1:
        std = 0*v
    else:
        v *= len(results) / (len(results) -1)
        std = sp.sqrt(v)
    return m,std



#@param jobname = "min_singular" or "scale_balance"
def test_sc(jobname, VS_VBMF=False):
    num_test = opt.num_test
    max_epoch = opt.max_epoch
    max_epoch= int(max_epoch*i_p_dim/i_dim)
    #base_scale *= i_p_dim/i_dim
    base_lr = opt.base_lr
    base_scale = opt.base_scale
    true_sigma = 0.1
    if jobname == "scale_balance":
        reg_coef = 0 ###no regularization
        list_dim_cauchy_vec =  [1]
        list_base_scale = [ base_scale/10, base_scale,base_scale*10]
        ###fix
        list_min_singular = [0.]
        list_zero_dim = [0]
        list_zero_thres = [0.]

        ###for debug
        list_base_scale = [base_scale]
    elif jobname == "min_singular":
        ### for paper
        reg_coef = 1e-3

        #TODO to check robustness
        ROBUST_CHECK = False
        if ROBUST_CHECK:
            #reg_coef = 5e-4
            #reg_coef = 2e-3
            reg_coef = 1e-3

        ratio = [0, 0.2, 0.4, 0.6, 0.8, 1]

        list_zero_dim = [int(i_dim*r) for r in ratio]
        list_min_singular =[0.05,0.1,0.15,0.2,0.3,0.4]
        list_base_scale = [base_scale]
        list_dim_cauchy_vec = [1] ### set 2 for version 1
        list_zero_thres = [reg_coef, 1e-1]


        ### for debug
        #list_zero_dim = [40]
        #list_min_singular=[ 0.1, 0.2, 0.3]
        #list_dim_cauchy_vec = [1]
        #

    else:
        sys.exit(-1)






    num_zero_dim = len(list_zero_dim)
    num_min_singular = len(list_min_singular)

    now = datetime.now()
    temp_jobname = jobname + '_{0:%m%d%H%M}'.format(now)

    setting = {
    "jobname": temp_jobname,
    "dim":i_dim,
    "p_dim":i_p_dim,
    "minibatch": opt.minibatch,
    "num_test":num_test,
    "max_epoch":max_epoch,
    "base_lr":base_lr,
    "base_scale":base_scale,
    "step_epoch": opt.step_epoch,
    "reg_coef":reg_coef,
    "list_dim_cauchy_vec":list_dim_cauchy_vec,
    "list_base_scale":list_base_scale   ,
    "list_min_singular":list_min_singular ,
    "list_zero_dim":list_zero_dim     ,
    "list_zero_thres":list_zero_thres   ,
    "true_sigma": true_sigma
    }


    dirname = "../images/{}".format(temp_jobname)
    while(os.path.exists(dirname)):
        dirname = dirname + "0"
    os.makedirs(dirname)
    setting_log = open("{}/setting.json".format(dirname), "w")
    json.dump(setting, setting_log, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
    setting_log.close()



    ###base_line
    #rank_recovery_baseline(dirname, list_zero_dim, list_min_singular, list_zero_thres, sigma=0.1)

    linestyles = ["-", "--", "-.", ":"]

    if VS_VBMF:
        ### Run VBMF
        vbmf_estimated_rank = np.empty((num_zero_dim, num_min_singular))
        std_vbmf_estimated_rank = np.empty((num_zero_dim, num_min_singular))

        for i in range(num_zero_dim):
            zero_dim = list_zero_dim[i]
            for j in range(num_min_singular):
                min_singular = list_min_singular[j]
                results = []
                for n in range(num_test):
                    r = validate_vbmf_spn(i_dim,i_p_dim, 0.1, min_singular, zero_dim)
                    results.append(r)

                m_r, v_r = _mean_and_std(results)
                vbmf_estimated_rank[i][j] =  m_r
                std_vbmf_estimated_rank[i][j] =  v_r

        ### plot results
        x_axis = list_min_singular
        for i in range(num_zero_dim):
            plt.figure()
            plt.style.use("seaborn-paper")
            plt.rc('text', usetex=opt.usetex)
            plt.rc('font', family="sans-serif", serif='Helvetica')
            plt.rcParams["font.size"] = 8*2

            zero_dim = list_zero_dim[i]
            true_rank = i_dim - zero_dim
            plt.plot(x_axis, vbmf_estimated_rank[i], label="true rank: {}".format(true_rank), marker="+")
            plt.xlabel("$\lambda_{min}$")
            plt.ylabel("Estimated rank")
            plt.legend()
            plt.savefig("{}/vbmf_true_ranks-{}.{}".format(dirname,true_rank , opt.ext),dpi=opt.dpi)
            plt.clf()
            plt.close()

        ### concated plot
        plt.figure()
        plt.style.use("seaborn-paper")
        plt.rc('text', usetex=opt.usetex)
        plt.rc('font', family='sans-serif', serif='Helvetica')
        plt.rcParams["font.size"] = 8*2


        for i in range(num_zero_dim):
            zero_dim = list_zero_dim[i]
            true_rank = i_dim - zero_dim
            diff_rank = vbmf_estimated_rank[i] - true_rank
            std_rank = std_vbmf_estimated_rank[i]
            plt.errorbar(x_axis, diff_rank, std_rank, label="{}".format(true_rank), linestyle=linestyles[i])


        plt.title("EVBMF")
        plt.xlabel("$\lambda_{min}$")
        ##plt.ylim(-i_dim, 30)
        plt.ylabel("Estimated Rank $-$ True Rank")
        plt.legend()
        filename = "{}/rank-recovery-evbmf.{}".format(dirname, opt.ext)
        plt.savefig(filename,dpi=opt.dpi)
        logging.info(filename)
        plt.clf()
        plt.close()




    ### Run
    num_thres = len(list_zero_thres)

    list_val_loss_curve = []
    list_train_loss_curve = []
    list_estimated_rank_curve  = []
    list_std_estimated_rank_curve  = []

    list_forward_iter = []

    for base_scale in list_base_scale:
        for dim_cauchy_vec in list_dim_cauchy_vec:
            for zero_dim in list_zero_dim:
                for min_singular in list_min_singular:
                    result_diag_A = []; result_sigma=[]; result_val_loss=[];result_train_loss=[];
                    result_num_zero = []; result_forward_iter = []
                    for n in trange(num_test, desc="num_test"):
                        diag_A, sigma, train_loss_array, val_loss_array,num_zero_array,forward_iter\
                        =test_optimize(\
                        base_lr=base_lr,
                        max_epoch=max_epoch,min_singular=min_singular, zero_dim = zero_dim,\
                        base_scale=base_scale, dim_cauchy_vec=dim_cauchy_vec,\
                        reg_coef=reg_coef,\
                        list_zero_thres= list_zero_thres,
                        true_sigma=true_sigma)
                        result_diag_A.append( diag_A)
                        result_sigma.append(sigma)
                        result_val_loss.append( val_loss_array)
                        result_train_loss.append( train_loss_array)
                        #result_num_zero.append( num_zero_array.T[idx_zero_thres_for_plot])
                        result_num_zero.append( num_zero_array)
                        result_forward_iter.append(forward_iter)
                    m_diag_A, v_diag_A = _mean_and_std(result_diag_A)
                    m_sigma, v_sigma = _mean_and_std(result_sigma)
                    m_train_loss, v_train_loss = _mean_and_std(result_train_loss)
                    m_val_loss, v_val_loss = _mean_and_std(result_val_loss)
                    m_num_zero, v_num_zero = _mean_and_std(result_num_zero)
                    m_f_iter, v_f_iter = _mean_and_std(result_forward_iter)

                    logging.info("RESULT:base_scale = {}, ncn = {}, val_loss = {},\n average_sigma = {}".format(\
                    base_scale, dim_cauchy_vec, m_val_loss[-1], m_sigma) )
                    logging.debug("average_a= {}".format(m_diag_A))
                    list_train_loss_curve.append(m_train_loss)
                    list_val_loss_curve.append(m_val_loss)
                    list_estimated_rank_curve.append( i_dim - m_num_zero )
                    list_std_estimated_rank_curve.append( v_num_zero)
                    list_forward_iter.append([m_f_iter, v_f_iter])

    x_axis = np.arange(m_val_loss.shape[0])




    ### print average forward_iter
    iter_log = open("{}/forward_iter.txt".format(dirname), "w")
    iter_log.write("{}".format(list_forward_iter))
    iter_log.close()





    if jobname == "scale_balance":
        #############
        ###plot val loss
        #############
        plt.clf()
        plt.close()
        plt.figure(figsize=(6,4))
        plt.style.use("seaborn-paper")
        plt.rc('text', usetex=opt.usetex)

        n = 0
        linestyles = ["-", "--", "-.", ":"]
        for base_scale in list_base_scale:
            line_idx  = 0
            for dim_cauchy_vec in list_dim_cauchy_vec:
                ### separate dim cauchy vec
                #plt.plot(x_axis,list_val_loss_curve[n], label="({0:3.2e}, {1})".format(base_scale, dim_cauchy_vec))
                ### TODO set dim_cauchy_vec =  1
                plt.plot(x_axis,list_val_loss_curve[n], label="$\gamma={0:3.2e}$".format(base_scale), linestyle=linestyles[ line_idx % 4])
                n+=1
                line_idx += 1
        plt.title("SPN")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        ### TODO modify here
        #plt.ylim(0., 0.3)
        plt.legend()
        filename = "{}/test_scale_val.{}".format(dirname, opt.ext)
        logging.info(filename)
        plt.savefig(filename,dpi=opt.dpi)
        plt.clf()
        plt.close()


        #############
        ###plot train loss
        #############
        plt.figure()
        n = 0
        for base_scale in list_base_scale:
            for dim_cauchy_vec in list_dim_cauchy_vec:
                plt.plot(x_axis,list_train_loss_curve[n], label="(s,c)=({0:3.2e}, {1})".format(base_scale, dim_cauchy_vec))
                n+=1

        plt.legend()
        plt.title("Train loss")
        plt.xlabel("Epoch")
        plt.ylabel("Train loss")
        plt.savefig("{}/test_scale_train.{}".format(dirname, opt.ext),dpi=opt.dpi)

        plt.clf()
        plt.close()

    elif jobname == "min_singular":
        #import pdb; pdb.set_trace()
        num = len(list_zero_dim)
        half = int(num /2)

        ################
        ###plot val_loss
        ################
        n = 0
        plt.figure()
        for zero_dim in list_zero_dim[:half]:
            idx = 0
            for min_singular in list_min_singular:
                plt.plot(x_axis,list_val_loss_curve[n], label="({}, {})".format(i_dim - zero_dim, min_singular), linestyle=linestyles[idx % 4])
                n+=1
                idx+=1

        #plt.title("SPN")
        plt.xlabel("Epoch")
        plt.ylabel("Validation loss")
        #plt.ylim(0, 0.7)

        plt.legend()
        plt.savefig("{}/test_ms_low_val.{}".format(dirname,opt.ext),dpi=opt.dpi)
        plt.clf()
        plt.close()

        plt.figure()
        for zero_dim in list_zero_dim[half:]:
            for min_singular in list_min_singular:
                plt.plot(x_axis,list_val_loss_curve[n], label="({}, {})".format(i_dim -zero_dim, min_singular), linestyle=linestyles[ n % 4])
                n+=1

        #plt.title("SPN")
        plt.xlabel("Epoch")
        plt.ylabel("Validation loss")
        #plt.ylim(0, 0.7)

        plt.legend()
        plt.savefig("{}/test_ms_high_val.{}".format(dirname, opt.ext),dpi=opt.dpi)
        plt.clf()
        plt.close()


        ################
        ###plot estimated rank
        ################
        for k in range(len(list_zero_thres)):
            n = 0
            logging.info(list_estimated_rank_curve)
            plt.figure()
            for zero_dim in list_zero_dim[:half]:
                for min_singular in list_min_singular:
                        plt.plot(x_axis,list_estimated_rank_curve[n][:,k],\
                        label="({}, {})".format(\
                        i_dim - zero_dim, min_singular))
                        n+=1
                plt.plot(x_axis, (i_dim-zero_dim)*np.ones(len(x_axis)), linestyle="--", color="black")

            #plt.title("Validation loss")
            plt.xlabel("Epoch")
            plt.ylabel("Estimated rank")
            #plt.ylim(0,i_dim)

            plt.legend()
            name = "{}/test_ms_low_zeros_thres-{}.{}".format(dirname, list_zero_thres[k],opt.ext)
            logging.info(name)
            plt.savefig(name,dpi=opt.dpi)
            plt.clf()
            plt.close()

            plt.figure()
            for zero_dim in list_zero_dim[half:]:
                for min_singular in list_min_singular:
                    plt.plot(x_axis,list_estimated_rank_curve[n][:,k], label="({}, {})".format(i_dim - zero_dim, min_singular))
                    n+=1
                plt.plot(x_axis, (i_dim - zero_dim)*np.ones(len(x_axis)), linestyle="--", color="black")

            #plt.title("Validation loss")
            plt.title("Estimated rank")
            plt.xlabel("Epoch")
            plt.ylabel("Estimated rank")
            #plt.ylim(0,i_dim)
            #plt.yticks( np.arange(5, i_dim+1))
            plt.legend()
            name = "{}/test_ms_high_zeros_thres-{}.{}".format(dirname, list_zero_thres[k], opt.ext)
            logging.info(name)
            plt.savefig(name,dpi=opt.dpi)
            plt.clf()
            plt.close()



        ##################
        ###plot train_loss
        ##################
        n = 0

        plt.figure()
        for zero_dim in list_zero_dim[:half]:
            for min_singular in list_min_singular:
                plt.plot(x_axis,list_train_loss_curve[n], label="({}, {})".format(i_dim - zero_dim, min_singular))
                n+=1

        #plt.title("Validation loss")
        plt.xlabel("epoch")
        plt.ylabel("train loss")
        plt.legend()

        plt.savefig("{}/test_ms_low_train.{}".format(dirname,opt.ext),dpi=opt.dpi)
        plt.clf()
        plt.close()

        plt.figure()
        for zero_dim in list_zero_dim[half:]:
            for min_singular in list_min_singular:
                plt.plot(x_axis,list_train_loss_curve[n], label="({}, {})".format(i_dim - zero_dim, min_singular))
                n+=1

        #plt.title("Validation loss")
        plt.xlabel("epoch")
        plt.ylabel("train loss")

        plt.legend()
        plt.savefig("{}/test_ms_high_train.{}".format(dirname,opt.ext),dpi=opt.dpi)
        plt.clf()
        plt.close()

        logging.info(list_zero_dim)
        logging.info(list_min_singular)
        fde_spn_estimated_rank_curve = np.asarray(list_estimated_rank_curve).reshape( num_zero_dim, num_min_singular, -1, num_thres)
        std_fde_spn_estimated_rank_curve = np.asarray(list_std_estimated_rank_curve).reshape( num_zero_dim, num_min_singular, -1, num_thres)

        ### Consider last result
        fde_spn_estimated_rank = fde_spn_estimated_rank_curve[:,:,-1,:]
        std_fde_spn_estimated_rank = std_fde_spn_estimated_rank_curve[:,:,-1,:]

        x_axis = list_min_singular
        for i in range(num_zero_dim):
            plt.figure()
            plt.style.use("seaborn-paper")
            plt.rc('text', usetex=opt.usetex)
            plt.rc('font', family="sans-serif", serif='Helvetica')
            plt.rcParams["font.size"] = 8*2

            zero_dim = list_zero_dim[i]
            true_rank = i_dim - zero_dim
            ### plot true_rank
            plt.plot(x_axis, (true_rank)*np.ones(len(x_axis)), linestyle="--", color="black")

            for k in range(num_thres):
                if list_zero_thres[k] == 0.0001:
                    label = "FDESPN: $\delta=10^{-4}$"
                elif list_zero_thres[k] == 0.01:
                    label = "FDESPN: $\delta=10^{-2}$"
                else:
                    label = "FDESPN: $\delta={}$".format(list_zero_thres[k])
                plt.plot(x_axis,fde_spn_estimated_rank[i,:,k], label=label, marker="+")

            if VS_VBMF:
                plt.plot(x_axis, vbmf_estimated_rank[i], label="EVBMF", marker="+")


            plt.title("True rank = {}".format(true_rank))
            plt.xlabel("$\lambda_{min}$")
            #plt.ylim(0,i_dim)
            plt.ylabel("Estimated rank")
            plt.legend()
            plt.savefig("{}/true_ranks-{}.{}".format(dirname, true_rank ,opt.ext),dpi=opt.dpi)
            plt.clf()
            plt.close()

        ### plot esimted rank - true rank
        x_axis = list_min_singular
        for k in range(num_thres):
            plt.figure()
            plt.style.use("seaborn-paper")
            plt.rc('text', usetex=opt.usetex)
            plt.rc('font', family="sans-serif", serif='Helvetica')
            plt.rcParams["font.size"] = 8*2
            thres = list_zero_thres[k]
            for i in range(num_zero_dim):
                zero_dim = list_zero_dim[i]
                true_rank = i_dim - zero_dim
                diff_rank = fde_spn_estimated_rank[i,:,k] - true_rank
                std_rank = std_fde_spn_estimated_rank[i,:, k]
                label = "${}$".format(true_rank)
                plt.errorbar(x_axis,diff_rank, std_rank, label=label,linestyle=linestyles[i])


            plt.xlabel("$\lambda_{min}$")
            plt.title("CNL")
            #plt.ylim(-i_dim, 30)

            plt.ylabel("Estimated Rank $-$ True Rank")
            plt.legend()
            filename = "{0}/rank-recovery_th-{1:1.0e}.{2}".format(dirname, thres,opt.ext)
            plt.savefig(filename,dpi=opt.dpi)
            plt.clf()
            plt.close()
            logging.info(filename)


        ### plot esimted rank - true rank
        ### for robust check,
        ### yrange is scoped
        x_axis = list_min_singular
        for k in range(num_thres):
            plt.figure()
            plt.style.use("seaborn-paper")
            plt.rc('text', usetex=opt.usetex)
            plt.rc('font', family="sans-serif", serif='Helvetica')
            plt.rcParams["font.size"] = 8*2
            thres = list_zero_thres[k]
            for i in range(num_zero_dim):
                zero_dim = list_zero_dim[i]
                true_rank = i_dim - zero_dim
                diff_rank = fde_spn_estimated_rank[i,:,k] - true_rank
                std_rank = std_fde_spn_estimated_rank[i,:, k]
                label = "${}$".format(true_rank)
                plt.errorbar(x_axis,diff_rank, std_rank, label=label,linestyle=linestyles[i])

            plt.xlabel("$\lambda_{min}$")
            plt.title(r"$\xi =$ {0}".format(reg_coef))
            #plt.yticks([2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5,2.0])
            #plt.ylim(-10, 10)

            plt.ylabel("Estimated Rank $-$ True Rank")
            plt.legend()
            plt.savefig("{0}/rank-recovery_th-{1:1.0e}_scoped.{2}".format(dirname, thres,opt.ext),dpi=opt.dpi)
            plt.clf()
            plt.close()


    else: sys.exit(-1)



def rank_recovery_baseline(dirname, list_zero_dim, list_min_singular, list_zero_thres,sigma=0.1):
    """
    Rank recovery based on baseline method;
    cutting off the lower singular values which are smaller than zerto_thres
    """
    logging.info("rank_recovery_baseline")
    p_dim = opt.p_dim
    dim = opt.dim
    COMPLEX = False
    for thres in list_zero_thres:
        plt.figure()
        plt.rc('text', usetex=opt.usetex)
        plt.style.use("seaborn-paper")
        #plt.rc('font', family="sans-serif", serif='Helvetica')
        #plt.rcParams["font.size"] = 8*2
        plt.title("Baseline")
        x_axis = list_min_singular
        ls_idx=0 ###linestyle index
        for zero_dim in list_zero_dim:
            num_zero_list = []
            std_num_zero_list = []

            for min_singular in list_min_singular:
                list_results = []
                for n in range(opt.num_test):
                    logging.info( "zero_dim={}".format(zero_dim) )
                    logging.info( "min_singular={}".format(min_singular) )

                    param_mat = random_from_diag(p_dim, dim, zero_dim, min_singular, COMPLEX)
                    _, param, _ = np.linalg.svd(param_mat)
                    #logging.info( "truth=\n{}".format(np.sort(param)))

                    evs= np.linalg.eigh(signal_plus_noise(param_mat,sigma, COMPLEX=False))[0]
                    sq_sample = sp.sqrt(evs)
                    ### count zero
                    list_results.append(  np.where( sq_sample < thres)[0].size )
                mean, std = _mean_and_std(list_results)
                num_zero_list.append(mean)
                std_num_zero_list.append(std)

            diff_rank = zero_dim - np.asarray(num_zero_list)
            true_rank = dim -  zero_dim
            label = "${}$".format(true_rank)
            plt.errorbar(x_axis,diff_rank, std_num_zero_list, label=label, linestyle=linestyles[ls_idx])
            ls_idx+=1


        plt.xlabel("$\lambda_{min}$")
        #TODO modify here
        #plt.ylim(-dim, 30)
        plt.ylabel("Estimated Rank $-$ True Rank")
        plt.legend()
        filename = "{0}/rank-recovery-baseline-{1:1.0e}.{2}".format(dirname,thres,opt.ext)
        logging.info(filename)
        plt.savefig(filename)
        plt.clf()
        plt.close()






timer = Timer()
timer.tic()
test_sc(jobname =jobname,VS_VBMF=i_vbmf)
timer.toc()
logging.info("total_time= {} min.".format(timer.total_time/60.))
