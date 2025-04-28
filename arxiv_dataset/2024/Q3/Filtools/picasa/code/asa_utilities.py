import os
import numpy as np
import scipy.linalg as linalg
import scipy.optimize as opt
import scipy.spatial as spatial
import scipy.special as special
from time import time
import sys
import gc
import multiprocessing as mp
import psutil

#######################################################################

class ASAUtilities(object):
    """ Useful general-purpose functions. """

    def __init__(self,base_dir=None,logfile=None):
        self.logfile = logfile
        self.base_dir = base_dir if base_dir is not None else os.getcwd()+'/'

        # from sec 15.6 of Numerical Recipes;
        # for use in ellipse plotting
        self.dchi2 = {'68p3':np.array([0.0,1.0,2.3,3.53,4.72,5.89,7.04]),
                      '90':np.array([0.0,2.71,4.61,6.25,7.78,9.24,10.6]),
                      '95p4':np.array([0.0,4.0,6.17,8.02,9.7,11.3,12.8]),
                      '99p73':np.array([0.0,9.0,11.8,14.2,16.3,18.2,20.1])}

        self.TINY = 1e-10
        self.NPROC = np.max([1,psutil.cpu_count(logical=False)])

    ############################################################
    def gen_latin_hypercube(self,Nsamp=10,dim=2,symmetric=True,param_mins=None,param_maxs=None,
                            rng=None):
        """ Generate Latin hypercube sample (symmetric by default). 
             Either param_mins and param_maxs should both be None or both be array-like of shape (dim,). 
            -- rng: either None or instance of numpy.random.RandomState(). Default None.
             Code from FirefoxMetzger's answer at
             https://codereview.stackexchange.com/questions/223569/generating-latin-hypercube-samples-with-numpy
             See https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.49.7292&rep=rep1&type=pdf
             for some ideas reg utility of symmetric Latin hypercubes.
             Returns array of shape (Nsamp,dim) with values in range (0,1) or respective minimum to maximum values.
        """
        if (param_mins is not None): 
            if (param_maxs is None):
                raise TypeError("param_mins and param_maxs should both be None or array-like of shape (dim,)")
            if len(param_mins) != dim:
                raise TypeError("len(param_mins) should be equal to dim")
        if (param_maxs is not None) :
            if(param_mins is None):
                raise TypeError("param_mins and param_maxs should both be None or array-like of shape (dim,)")
            if len(param_maxs) != dim:
                raise TypeError("len(param_maxs) should be equal to dim")

        rng_use = rng if rng is not None else np.random.RandomState()

        if symmetric:
            available_indices = [set(range(Nsamp)) for _ in range(dim)]
            samples = []

            # if Nsamp is odd, we have to choose the midpoint as a sample
            if Nsamp % 2 != 0:
                k = Nsamp//2
                samples.append([k] * dim)
                for idx in available_indices:
                    idx.remove(k)

            # sample symmetrical pairs
            for _ in range(Nsamp//2):
                sample1 = list()
                sample2 = list()

                for idx in available_indices:
                    k = rng_use.choice(np.array(list(idx)),size=1,replace=False)[0]# random.sample(idx, 1)[0]
                    sample1.append(k)
                    sample2.append(Nsamp-1-k)
                    idx.remove(k)
                    idx.remove(Nsamp-1-k)

                samples.append(sample1)
                samples.append(sample2)

            samples = np.array(samples)/(1.0*Nsamp)
        else:
            samples = np.array([rng_use.permutation(Nsamp) for i in range(dim)])/(1.0*Nsamp)
            samples = samples.T

        if (param_mins is not None):
            for d in range(dim):
                samples[:,d] *= (param_maxs[d] - param_mins[d])
                samples[:,d] += param_mins[d]

        return samples
    ############################################################

    #######################
    # Convenience function for using Pool
    def do_this_param(self,param,model_func,model_args,cost_func,cost_args,data,invcov_data):
            # predict model for each parameter vector
            model = model_func(model_args,param)
            # compute and store cost function
            this_chi2 = cost_func(data,invcov_data,model,cost_args) 
            return this_chi2
    #######################


    ############################################################
    def asa_chi2min(self,data,invcov_data,param_mins,param_maxs,model_func,model_args,cost_func,cost_args,
                    read_file = None,rotate=None,
                    Nsamp=50,Nzoom=100,zoomer=1.0,eps_close=0.1,eps_conv=0.1,robust_convergence=True,
                    last_iters=None,last_frac=0.5,burn_in_iter=0,C_sig=1.0,
                    parallel=False,rng=None,write_out=None,over_write=False,verbose=True):
        """ Anisotropic simulated annealing to minimise cost function given data and errors. 
            -- data: 1-d array of shape (Ndata,).
            -- invcov_data: either 1-d array of shape (Ndata,) [assumed to contain 1/sig_data^2]
                                      or 2-d array of shape (Ndata,Ndata) [assumed to contain inverse covariance matrix]
            -- param_mins,param_maxs: lists of length = parameter dimensions, containing starting values of
                                      minima and maxima along each direction.
            -- model_func: callable as model_func(model_args,params) where params is parameter vector. 
                           Should return model array of shape (Ndata,).
            -- model_args: tuple/list containing other arguments used by model_func.
            -- cost_func: cost function to be minimised. callable as cost_func(data,invcov_data,model,cost_args)
            -- cost_args: tuple/list containing other arguments for cost_func.
            -- read_file: if not None, should be a valid path (relative to self.base_dir) 
                                to input file containing chi2,param values.
                                In this case, param_mins and param_maxs will not be used and only last iterations (if requested)
                                will be performed.
            -- rotate: None or matrix. If not None, code assumes that input ranges are for y = rotate.T . param_vals, 
                           and will undo this rotation when evaluating the model. 
            -- Nsamp: number of Latin hypercube samples at each iteration.
            -- Nzoom: number of zoom iterations. 
                              Code will automatically stop if chi2 is converged, so set to some large value.
            -- zoomer: factor to multiply anisotropic dispersions at each iteration.
            -- eps_close: fraction of current parameter range to decide whether current minimum is too close to edge
            -- eps_conv: ** percentage** change in chi2 to decide convergence
            -- last_iters: None or list of positive floats [sig_1,sig_2...,sig_N]. 
                                If floats supplied, N additional iterations will be performed by scaling each parameter direction
                                by factor sig_n at iteration n. Number of samples will be controlled by value of last_frac. 
                                (Default: None, i.e. no additional iterations, same as setting last_frac = 0.0.)
            -- last_frac: ratio of total number of last iterations samples to total number of evaluations. (Default 0.5)
            -- C_sig: factor to increase Gaussian widths in parameter eigenspace, to approximately account for
                          non-Gaussian shape of exp(-chi2/2) when setting up last iterations. (Default 1.0).
            -- burn_in_iter: number of samples to exclude from quadratic form estimation and sampling rate calculation
                                      when setting up last iterations. (Default 0, i.e. use full sample).
            -- robust_convergence: if True, will attempt to jump out of local minima.
            -- parallel: if True, multiprocessing.Pool.apply_async() will be used for parallel computing in each zoom level.
                             Default False (since pooling inside a class seems to break Jupyter notebooks).
            -- rng: passed to gen_latin_hypercube()
            -- write_out: if not None, should be a valid path (relative to self.base_dir) 
                                 to output file for writing (appending) chi2,param values.
            -- over_write: if True, over-write existing file if any. (Default False)
            -- verbose: if True, generate progress output.

            Output chi2 = cost_func() will be minimised.

            Returns:
              1. chi2 values, shape (Nzoom_actual*Nsamp,)
              2. parameter values, shape (Nzoom_actual*Nsamp,len(param_mins))
            where Nzoom_actual is actual number of zoom levels completed before convergence.
        """
        if verbose:
            self.print_this("Anisotropic simulated annealing... ",self.logfile)

        if last_iters is not None:
            if not isinstance(last_iters,list):
                self.print_this("... last_iters is not a valid list. No extra iterations will be performed.",self.logfile)
                last_iters = None
            if C_sig < 1.0:
                self.print_this("... C_sig = {0:.3f} < 1.0 detected. Setting equal to 1.0".format(C_sig),self.logfile)
                C_sig = 1.0

        if read_file is not None:
            if os.path.isfile(self.base_dir + read_file):
                if verbose:
                    self.print_this("... reading from file: "+self.base_dir + read_file,self.logfile)
                chi2,param_vals = self.read_data_from_file(read_file)
                dim = param_vals.shape[1]
                if last_iters is None:
                    if verbose:
                        self.print_this("... nothing more requested: exiting ",self.logfile)
                    return chi2,param_vals
            else:
                raise ValueError('Invalid file path specified for reading in asa_chi2min()')

        if len(data.shape) != 1:
            raise TypeError('data vector should be 1-d array in asa_chi2min()')
        Ndata = data.size 

        if invcov_data.shape not in [(Ndata,),(Ndata,Ndata)]:
            raise TypeError('Incompatible inverse covariance array detected in asa_chi2min()')

        if read_file is None:
            if len(param_mins) != len(param_maxs):
                raise TypeError('Incompatible lists param_mins,param_maxs detected in asa_chi2min().')
            dim = len(param_mins)

        if rotate is not None:
            if rotate.shape != (dim,dim):
                raise TypeError('Invalid rotation matrix detected in asa_chi2min()')
            if verbose:
                self.print_this("... (possible) initial iterations will be sampled in eigenspace",self.logfile)
            use_rotate = rotate.copy()
        else:
            use_rotate = np.eye(dim)
            if verbose:
                self.print_this("... (possible) initial iterations will be sampled in parameter space",self.logfile)

        # number of degrees of freedom. for use in testing goodness of fit.
        dof = Ndata - dim

        if (read_file is not None) & (last_iters is not None) & (write_out is not None):
            if not over_write:
                if verbose:
                    self.print_this("... write_out requested after reading file and performing additional iterations",self.logfile)
                    self.print_this("... ... forcing over_write = True",self.logfile) # else will repeat entries.
            over_write = True

        if write_out is not None:
            if (not os.path.isfile(self.base_dir + write_out)) | over_write:
                if verbose:
                    self.print_this("... output will be (over-)written to file: "+self.base_dir + write_out,self.logfile)
                with open(self.base_dir + write_out,'w') as f:
                    f.write('# chi2 | a0 | ... | a{0:d}\n'.format(dim-1))
            else:
                if verbose:
                    self.print_this("... output will be appended to file: "+self.base_dir + write_out,self.logfile)
        else:
            if verbose:
                self.print_this("... output will not be written",self.logfile)


        if read_file is None:
            ########
            # for use in testing for local minima
            jump_out = False 
            delta_p_input = np.array(param_maxs) - np.array(param_mins)
            ########
            
            chi2 = np.zeros(Nzoom*Nsamp,dtype=float)
            param_vals = np.zeros((Nzoom*Nsamp,dim),dtype=float)

            param_mins_next = np.array(param_mins)
            param_maxs_next = np.array(param_maxs)

            chi2_prev = 0.0
            chi2_prev_prev = 0.0
            params_sig_prev = np.zeros(dim,dtype=float)

            for zoom in range(Nzoom):
                ####################################
                if verbose:
                    self.print_this("... zoom level {0:d}".format(zoom+1),self.logfile)

                sl = np.s_[zoom*Nsamp:(zoom+1)*Nsamp]

                # sample (eigen)parameters for this zoom level
                this_paramvals = self.gen_latin_hypercube(Nsamp=Nsamp,dim=dim,
                                                          param_mins=param_mins_next,param_maxs=param_maxs_next,rng=rng)
                use_paramvals = np.dot(use_rotate,this_paramvals.T).T # ensure args of model_func are actual params
                param_vals[sl] = this_paramvals

                ##################################
                # Optional parallelisation
                ##################################
                if parallel:
                    pool = mp.Pool(processes=self.NPROC)
                    results = [pool.apply_async(self.do_this_param,
                                                args=(use_paramvals[n],model_func,model_args,cost_func,cost_args,
                                                      data,invcov_data)) for n in range(Nsamp)]
                    for n in range(Nsamp):
                        chi2[n + zoom*Nsamp] = results[n].get()
                        if verbose:
                            self.status_bar(n,Nsamp)
                    pool.close()
                    pool.join()
                else:
                    for n in range(Nsamp):
                        # predict model for each parameter vector
                        model = model_func(model_args,use_paramvals[n])
                        # compute and store cost function
                        this_chi2 = cost_func(data,invcov_data,model,cost_args) 
                        chi2[n + zoom*Nsamp] = this_chi2 
                        if verbose:
                            self.status_bar(n,Nsamp)
                ##################################

                # current best-fit
                ind_finite = np.where(np.isfinite(chi2[sl]))[0]
                if ind_finite.size > 0:
                    i_min = chi2[sl][ind_finite].argmin()
                    params_best = param_vals[sl][ind_finite][i_min]
                    chi2_min = chi2[sl][ind_finite][i_min]
                else:
                    params_best = 1.0*param_mins_next
                    chi2_min = np.inf

                delta_p = param_maxs_next - param_mins_next
                too_close = ((params_best > param_maxs_next - eps_close*delta_p) 
                             | (params_best < param_mins_next + eps_close*delta_p))
                
                if ind_finite.size > 0:
                    params_sig = self.asa_calc_sig(chi2[sl][ind_finite]-chi2_min,param_vals[sl][ind_finite],dim,trim=True) 
                else:
                    params_sig = 1.0*params_sig_prev
                params_sig[too_close] = delta_p[too_close]

                del ind_finite
                gc.collect()

                if verbose:
                    self.print_this("... current (eigen)  min ( p0,..,p{0:d} ) = ( ".format(dim-1)
                                    +','.join(['%.3e' % (pval,) for pval in params_best])+" )",self.logfile)
                    self.print_this("... (eigen) dispersions ( sig_p0,..,sig_p{0:d} ) = ( ".format(dim-1)
                                    +','.join(['%.3e' % (sval,) for sval in params_sig])+" )",self.logfile)
                    self.print_this("... chi2 = {0:.2f}".format(chi2_min),self.logfile)

                if (not np.isfinite(chi2_min)) & (not np.isfinite(chi2_prev)):
                    if verbose:
                        self.print_this("... chi2 has not been finite over 2 zoom levels. Consider repeating with new config.",
                                        self.logfile)
                    sl_brk = np.s_[:(zoom+1)*Nsamp]
                    chi2 = chi2[sl_brk]
                    param_vals = param_vals[sl_brk]
                    if verbose:
                        self.print_this("... ... breaking out",self.logfile)
                    break

                cond_breakout = np.fabs(chi2_min - chi2_prev) < 0.01*eps_conv*np.fabs(chi2_min)
                cond_breakout = (cond_breakout & (np.fabs(chi2_min - chi2_prev_prev) 
                                                  < 0.01*eps_conv*np.fabs(chi2_min))) 
                cond_breakout = cond_breakout & np.all(params_sig <= params_sig_prev)
                cond_breakout = cond_breakout & (zoom < Nzoom-1)
                if cond_breakout:
                    if verbose:
                        # self.print_this("... chi2 appears converged over 2 zoom levels",
                        self.print_this("... chi2 appears converged over 2 zoom levels, with decreasing parameter ranges",
                                        self.logfile)

                    # testing for local minima if requested
                    chi2_red = chi2_min/dof 
                    pvalue = special.gammaincc(0.5*dof,0.5*chi2_min) if robust_convergence else 0.1
                    chi2_is_good = (pvalue > 0.05) 
                    if chi2_is_good:
                        if verbose:
                            if robust_convergence:
                                self.print_this("... ... chi2/dof = {0:.3f}; pvalue = {1:.2e} looks good".format(chi2_red,pvalue),
                                                self.logfile)
                            else:
                                self.print_this("... ... chi2/dof = {0:.3f}; not checking for local minimum".format(chi2_red),
                                                self.logfile)
                        sl_brk = np.s_[:(zoom+1)*Nsamp]
                        chi2 = chi2[sl_brk]
                        param_vals = param_vals[sl_brk]
                        if verbose:
                            self.print_this("... ... breaking out",self.logfile)
                        break
                    else:
                        if verbose:
                            self.print_this("... ... chi2/dof = {0:.3f}; pvalue = {1:.2e} indicates local minimum; trying to jump out"
                                            .format(chi2_red,pvalue),self.logfile)
                        jump_out = True

                chi2_prev_prev = 1.0*chi2_prev
                chi2_prev = 1.0*chi2_min
                params_sig_prev = 1.0*params_sig

                # update parameter ranges for next zoom level
                if jump_out:
                    param_mins_next = params_best - 0.05*delta_p_input - 1e-10
                    param_maxs_next = params_best + 0.05*delta_p_input + 1e-10
                    jump_out = False
                else:
                    param_mins_next = params_best - zoomer*params_sig - 1e-10
                    param_maxs_next = params_best + zoomer*params_sig + 1e-10
                ####################################

            ind_finite = np.where(np.isfinite(chi2))[0]
            if ind_finite.size < chi2.size:
                if verbose:
                    self.print_this("... discarding {0:d} indeterminate chi2 values".format(chi2.size - ind_finite.size),self.logfile)
                chi2 = chi2[ind_finite]
                param_vals = param_vals[ind_finite]
            del ind_finite
            gc.collect()

            ########
            if rotate is not None:
                if verbose:
                    self.print_this("... returning to parameter space",self.logfile)
                param_vals = np.dot(rotate,param_vals.T).T # param_vals now in parameter space
                # Recall use_paramvals = np.dot(use_rotate,this_paramvals.T).T 
            ########

        ######################
        # Last iterations
        ######################
        if last_iters is not None:
            Nsamp_last = int(last_frac/(1-last_frac)*chi2[burn_in_iter:].size) 
            if Nsamp_last > 0:
                if verbose:
                    self.print_this("... performing last {0:d} iteration(s)".format(len(last_iters)),self.logfile)
                    self.print_this("... excluding first {0:d} samples from sampling setup".format(burn_in_iter),
                                    self.logfile)
                cov_asa,params_best,chi2_min,eigvals,rotate = self.asa_get_bestfit(chi2[burn_in_iter:],
                                                                                   param_vals[burn_in_iter:])
                if verbose:
                    self.print_this("... obtained quadratic form",self.logfile)
                params_sig = np.sqrt(np.diag(cov_asa))
                for p in range(dim):
                    if (param_vals.T[p].max() - param_vals.T[p].min()) < 2*params_sig[p]:
                        self.print_this("Warning!: parameter range explored is < 2*sigma for parameter a{0:d}".format(p),
                                        self.logfile)
                ##########
                # Eigenspace
                if verbose:
                    self.print_this("... setting up eigenspace sampling",self.logfile)
                params_best = np.dot(rotate.T,params_best)
                params_sig = np.sqrt(eigvals)

                ##########
                # Sample distribution
                if verbose:
                    self.print_this("... defining sample distribution across iterations using C_sig = {0:.3f}".format(C_sig),
                                    self.logfile)
                dP = np.diff(special.erf(np.insert(last_iters,0,0.0)/np.sqrt(2)/C_sig)**dim)
                Weights = np.zeros((len(last_iters),len(last_iters)),dtype=float)
                for j in range(len(last_iters)):
                    for jpr in range(j,len(last_iters)):
                        Weights[j,jpr] = ((j+1)/(jpr+1))**dim*(1 - (1-1/(j+1))**dim)

                f_vec = np.dot(linalg.inv(Weights),dP)
                Nsamp_last_array = Nsamp_last*f_vec/np.sum(f_vec)
                Nsamp_last_array = Nsamp_last_array.astype('int')
                if verbose:
                    self.print_this("... ... {0:d} iterations have non-zero sample sizes"
                                    .format(np.where(Nsamp_last_array > 0)[0].size),self.logfile)

                for n in np.where(Nsamp_last_array > 0)[0]:
                    param_mins_last = params_best - last_iters[n]*params_sig
                    param_maxs_last = params_best + last_iters[n]*params_sig
                    chi2_last,params_last = self.asa_chi2min(data,invcov_data,param_mins_last,param_maxs_last,
                                                             model_func,model_args,cost_func,cost_args,
                                                             last_iters=None,write_out=None,rotate=rotate, 
                                                             Nsamp=Nsamp_last_array[n],Nzoom=1,rng=rng,verbose=verbose)
                    chi2 = np.concatenate((chi2,chi2_last))
                    param_vals = np.concatenate((param_vals,params_last),axis=0) # param_vals stays in param space

                if verbose:
                    ind_finite = np.where(np.isfinite(chi2))[0]
                    if ind_finite.size < chi2.size:
                        self.print_this("... discarding {0:d} indeterminate chi2 values".format(chi2.size - ind_finite.size),self.logfile)
                        chi2 = chi2[ind_finite]
                        param_vals = param_vals[ind_finite]
                    del ind_finite
                    gc.collect()
        else:
            if verbose:
                self.print_this("... ... no additional iterations to be performed",self.logfile)
        ######################

        if verbose:
            self.print_this("... annealing done",self.logfile)

        if write_out is not None:
            write_str = 'writing' if over_write else 'appending'
            if verbose:
                self.print_this('... ... '+write_str+' {0:d} entries to file: '.format(chi2.size)+self.base_dir + write_out,self.logfile)
            self.write_data_to_file(self.base_dir + write_out,chi2,param_vals,mode='a')

        if (read_file is None) & (write_out is not None) & (not over_write):
            # in this case, we have calculated new values and appended to pre-existing file.
            # so return the full set of parameters.
            if verbose:
                self.print_this("... ... reporting full output",self.logfile)
            chi2,param_vals = self.read_data_from_file(write_out)

        return chi2,param_vals
    ############################################################

    ############################################################
    def asa_calc_sig(self,chi2,param_vals,dim,trim=False):
        """ Convenience function to calculate anisotropic dispersions.
             Input:
               -- chi2,param_vals: arrays containing chain of chi2 (N,) and parameter vectors (N,dim)
               -- dim: number of parameters
               -- trim: boolean. if True, trim large chi2 values (default False)
             Returns:
               -- params_sig: array (dim,) containing parameter dispersions
        """            
        # setup exp(-chi^2/2) for estimating dispersion at this zoom level
        prob = -0.5*chi2
        if trim:
            prob[prob < -20.0] = 0.0 # if chi2 is too bad, contribute to width wrongly by giving unit weight
        prob = np.exp(prob)
        params_sig = np.zeros(dim,dtype=float)

        # estimate dispersion along each parameter axis as weighted integral
        for p in range(dim):
            sorter = param_vals[:,p].argsort()
            param_sorted = param_vals[:,p][sorter]
            prob_param = prob[sorter]
            mean = np.trapz(prob_param*param_sorted,x=param_sorted)/np.trapz(prob_param,x=param_sorted)
            sig_temp = np.sqrt(np.trapz(prob_param*(param_sorted-mean)**2,x=param_sorted)
                               /np.trapz(prob_param,x=param_sorted))
            params_sig[p] = sig_temp

        return params_sig
    ############################################################


    ############################################################
    def read_data_from_file(self,read_file):
        """ Convenience function to read ASA-compatible data file. Returns chi2 (N,) and params (N,dim). """
        data_prods = np.loadtxt(self.base_dir + read_file).T
        chi2 = data_prods[0].copy()
        param_vals = data_prods[1:].T.copy()
        dim = param_vals.shape[1]
        del data_prods
        gc.collect()                
        return chi2,param_vals
    ############################################################


    ############################################################
    def write_data_to_file(self,write_file,chi2,params,mode='w'):
        """ Convenience function to write ASA data to file. """
        dim = params.shape[1]
        Nout = params.shape[0]
        if chi2.size != Nout:
            raise TypeError('Incompatible data sets in write_data_to_file().')
        data_prods = np.zeros((dim+1,Nout),dtype=float)
        data_prods[0] = chi2
        data_prods[1:] = params.T
        data_prods = data_prods.T
        if mode=='a':
            with open(write_file,'a') as f:
                np.savetxt(f,data_prods,fmt='%.6e')
        else:
            np.savetxt(write_file,data_prods,header='chi2 | a0 | ... | a{:d}'.format(dim),fmt='%.6e')
        del data_prods
        gc.collect()
        return 
    ############################################################


    ############################################################
    def create_info_dict(self,params_best,cov_asa,chi2_file,chain_dir,
                         dsig=5.0,prop_fac=0.1,burn_in=500,train_size=None,rotate=None,
                         estimate='gpr',kernel='rbf',kernel_start=None,skip_train=False,
                         bound_fac=10.0,Nsamp_hyper=50,rng=None,mcmc_prior=None,max_samples=10000000,
                         verbose=False,latex_symb=None,Rminus1_stop=0.1,Rminus1_cl_stop=0.2,force=True):
        """ Convenience function to create info dictionary for Cobaya.
            -- params_best,cov_asa: from output of self.asa_get_bestfit().
            -- chi2_file: output file name, relative to self.base_dir, for ASA output.
            -- chain_dir: output folder name, relative to self.base_dir, for mcmc output.
            -- mcmc_prior: if not None, should be list of [min,max] pairs for 
                                      boundary of uniform prior along each parameter direction.
            -- dsig: if prior is None, uniform prior will be set between 
                         (best fit) +/- dsig*(std dev) along each parameter direction.
            -- prop_fac: factor multiplying cov_asa to set proposal covariance matrix.
            -- burn_in: no. of burn-in samples to discard in mcmc.
            -- train_size: integer size of training set for GPR. Default None (calculated internally by ASALike().
            -- rotate: if not None, expect matrix to rotate into eigenspace [y = rotate.T param]
            -- estimate: either 'gpr' or 'rbf', type of interpolation estimator. Default 'gpr'.
            -- kernel: type of kernel for each interpolator. 
                            For estimate = 'gpr': 'rq','rbf' or 'matern'.
                            For estimate = 'rbf': 'cubic' or 'gaussian'
                            Default 'rq'.
            -- kernel_start: None or list of floats. If latter, should contain starting values for appropriate kernel hyperparams.
            -- skip_train: if True, fixed RBF will be used instead of training GPR. 
                                  In this case, kernel_start must be a valid list of hyperparameters.
            -- bound_fac: float > 1.0, controls starting range of hyperparameters in GPR training
            -- Nsamp_hyper: int > 0, controls ASA sampling of hyperparameters in GPR training
            -- rng: None or instance of random state, used to set up training set in GPR training
            -- verbose: if True, training progress reported.
            -- latex_symb: list of strings, of length = no. of params. Default None (latex symbols set automatically)
            -- Rminus1_stop: convergence criterion of means for mcmc.
            -- Rminus1_cl_stop: convergence criterion of 95% bounds for mcmc.
            -- max_samples: maximum number of accepted samples allowed in MCMC sampler
            -- force: if True, mcmc will over-write existing chains.
            Returns: info dictionary for use with Cobaya.
        """
        dim = params_best.size
        if latex_symb is not None:
            if not isinstance(latex_symb,list):
                self.print_this('Invalid Latex symbols supplied: setting automatically')
                latex_symb = None
            elif len(latex_symb) != dim:
                self.print_this('Invalid number of Latex symbols supplied: setting automatically')
                latex_symb = None

        if mcmc_prior is not None:
            if not isinstance(mcmc_prior,list):
                self.print_this('Invalid prior supplied: setting automatically')
                mcmc_prior = None
            elif len(mcmc_prior) != dim:
                self.print_this('Invalid number of priors supplied: setting automatically')
                mcmc_prior = None
            else:
                for d in range(dim):
                    if len(mcmc_prior[d]) != 2:
                        self.print_this('Invalid prior boundaries supplied: setting automatically')
                        mcmc_prior = None
        
        sig_asa = np.sqrt(np.diag(cov_asa))
        info = {
            "likelihood": {
                "asa_likelihood.ASALike": {
                    "python_path": self.base_dir + "code/",
                    "data_file": self.base_dir + chi2_file,
                    "base_dir": self.base_dir,
                    "tmp_dir": self.base_dir + chain_dir,
                    "logfile": self.logfile,
                    "rotate": rotate,
                    "estimate": estimate,
                    "kernel": kernel,
                    "kernel_start": kernel_start,
                    "skip_train": skip_train,
                    "bound_fac": bound_fac,
                    "Nsamp_hyper": Nsamp_hyper,
                    "rng": rng,
                    "verbose": verbose,
                    "train_size": train_size
                }}}

        info["sampler"] = {"mcmc": {
            "Rminus1_stop": Rminus1_stop,
            "Rminus1_cl_stop": Rminus1_cl_stop,
            "learn_proposal": True,
            "measure_speeds": False,
            "burn_in": burn_in,
            "covmat": prop_fac*cov_asa,
            "covmat_params": ['a'+str(d) for d in range(dim)],
            "max_samples": max_samples
            }}

        param_dict = {}
        for d in range(dim):
            ref = params_best[d]
            sig = sig_asa[d]
            latex = 'a_'+str(d) if latex_symb is None else latex_symb[d]
            if mcmc_prior is not None:
                prior_min = mcmc_prior[d][0]
                prior_max = mcmc_prior[d][1]
            else:
                prior_min = ref-dsig*sig
                prior_max = ref+dsig*sig
            prior = {"dist": "uniform", "min": prior_min, "max": prior_max} 
            param_dict['a'+str(d)] = {"ref": ref,
                                     "prior": prior,
                                     "proposal": 0.01, # won't be used 
                                      "latex": latex
                                     }
        info["params"] = param_dict
        info["output"] = self.base_dir + chain_dir
        info["force"] = force
        
        return info
    ############################################################


    ############################################################
    def calc_Neval(self,sampler):
        """ Convenience function to calculate total number of likelihood evaluations in mcmc sampler."""
        
        N_acc = sampler.products()['progress']['N'].to_numpy()
        r_acc = sampler.products()['progress']['acceptance_rate']
        N_acc = np.insert(N_acc,0,0)
        N = int(np.sum(np.diff(N_acc)/r_acc))
        return N

    ############################################################

    ############################################################
    # Quadratic form fitting routines
    ############################################################
    ############################################################
    def asa_get_bestfit(self,chi2,param_vals,Ns=None,tree=None):
        """ Wrapper to extract quadratic estimator for best-fit, covariance and minimum chi2. 
             chi2, param_vals: output of asa_chi2min()
             Ns: None or int: number of samples to use (default None, then calculated internally)
             tree: None or instance of scipy.spatial.cKDTree using param_vals (useful for iterative calls).
                                       
             Returns: cov_asa (D,D), params_best (D,), chi2_min (scalar),eigvals (D,),rotate(D,D)
             Rotation convention is such that y = (rotate.T) . param_vals will give vector in eigenspace. 
        """
        dim = param_vals.shape[1]
        Ns_min = (dim+1)*(dim+2) // 2
        if Ns is not None:
            Ns_use = Ns
        else:
            # Ns_use = np.max([int(0.02*chi2.size),10*Ns_min])
            # Ns_use = np.min([chi2.size,Ns_use])
            Ns_use = 2*Ns_min

        if chi2.size <= Ns_use:
            raise ValueError('Too few sampling points. Use higher Nsamp or append chain with a few more iterations.')
        
        imin = chi2.argmin()
        tree = tree if tree is not None else spatial.cKDTree(param_vals)
        dist_nn,ind_nn = tree.query(param_vals[imin],k=Ns_use) 
        # Ns_use nearest neighbours of evaluated minimum

        Qparams,flag_ok_qfit = self.fit_quad_form(chi2[ind_nn],param_vals[ind_nn])
        if flag_ok_qfit:
            Q0,Q1,Q2 = self.unwrap_Qs(Qparams,dim)
            cov_asa = linalg.inv(Q2)
            params_best = -0.5*np.dot(cov_asa,Q1)
            chi2_min = Q0 + 0.5*np.dot(Q1,params_best)
            eigvals,rotate = linalg.eigh(cov_asa,check_finite=False)
        else:
            self.print_this('! Warning !... quadratic form fitting failed. Increasing sample size',self.logfile)
            eigvals = -1.0*np.ones(dim)
            cov_asa,params_best,chi2_min,rotate = 0,0,0,0

        if np.any(eigvals < 0.0): 
            if Ns_use >= 0.5*chi2.size:
                self.print_this('! Warning !',self.logfile)
                self.print_this('... positive definite quadratic form not obtained with as much as 50% of sample',self.logfile)
                self.print_this('... use higher Nsamp or append chain with a few more iterations',self.logfile)
                self.print_this('... returning inf eigen-values',self.logfile)
                self.print_this('! Warning !',self.logfile)
                eigvals[:] = np.inf
            else:
                cov_asa,params_best,chi2_min,eigvals,rotate = self.asa_get_bestfit(chi2,param_vals,Ns=2*Ns_use,tree=tree)

        # if lambda,R = eigh(C), then C = R Lambda R^T where Lambda is diagonal with lamda on the diagonal.
        # so y = R^T param_vals will give a correctly rotated parameter vector
        # such that the norm relative to C or C^-1 is preserved.
        # (note C^-1 = (R^T)^-1 Lambda^-1 R^-1 = R Lambda^-1 R^T).
        # E.g.: Q = p^T C^-1 p = p^T R Lambda^-1 R^T p = y^T Lambda^-1 y.
        return cov_asa,params_best,chi2_min,eigvals,rotate
    ############################################################

    ############################################################
    def quad_form(self,Qparams,x):
        """ Quadratic form for D-dimensional data vector x.
            Expect x.shape = (N,D)
            Qparams: ((D+1)(D+2)/2,), to be unwrapped into
            Q0: scalar; Q1: (D,) vector; Q2: (D,D) symmetric matrix
            Returns: x^T Q2 x + x^T Q1 + Q0 (shape (N,))
        """
        if len(x.shape) != 2:
            raise TypeError("Incompatible shape for x in quad_form(). Need (N,D), detected (" 
                            + ','.join(['%d' % (i,) for i in x.shape]) +').')
        D = x.shape[1]
        Q0,Q1,Q2 = self.unwrap_Qs(Qparams,D)
        return np.sum(np.dot(x,Q2)*x,axis=1) + np.dot(x,Q1) + Q0
    ############################################################

    ############################################################
    def quad_residuals(self,Qparams,x,y):
        """Wrapper for calculating residuals. Assumes Qparams ((D+1)(D+2)/2,), x (N,D) and y (N,)."""
        return y - self.quad_form(Qparams,x)
    ############################################################

    ############################################################
    def unwrap_Qs(self,Qparams,D):
        """ Convenience function to extract Q0,Q1,Q2 for use in quad_form(). """
        if Qparams.size != int((D+1)*(D+2)/2):
            raise TypeError("Incompatible shape for Qparams in unwrap_Qs(). Need ((D+1)(D+2)/2,), detected (" 
                            + ','.join(['%d' % (i,) for i in Qparams.shape]) +').')
        Q0 = Qparams[0]
        count = D+1
        Q1 = Qparams[1:count]
        if D > 1:
            Q2 = np.zeros((D,D),dtype=float)
            for di in range(D):
                for dj in range(di,D):
                    Q2[di,dj] = Qparams[count]
                    count += 1
            for di in range(1,D):
                for dj in range(di):
                    Q2[di,dj] = Q2[dj,di]
        else:
            Q2 = np.array([[Qparams[-1]]])

        return Q0,Q1,Q2
    ############################################################

    ############################################################
    def fit_quad_form(self,y,x):
        """ Fit quadratic form y = f(x) = x^T Q2 x + x^T Q1 + Q0 given N samples of y and x.
            Expect y and x to have shape (N,) and (N,D).
            Returns: 
            -- Qparams ((D+1)(D+2)/2,) that can be unwrapped into Q0 (scalar), Q1 (D,), Q2 (D,D).
            -- flag_ok_qfit: boolean flag to indicate if fit succeeded.
        """
        flag_ok_qfit = True

        if len(x.shape) != 2:
            raise TypeError("Incompatible shape for x in fit_quad_form(). Need (N,D), detected (" 
                            + ','.join(['%d' % (i,) for i in x.shape]) +').')
        D = x.shape[1]
        if y.size != x.shape[0]:
            raise TypeError("Incompatible shape for y in fit_quad_form(). Need ({0:d},), detected (".format(x.shape[0]) 
                            + ','.join(['%d' % (i,) for i in y.shape]) +').')
        Qparams_0 = np.ones(int((D+1)*(D+2)/2),dtype=float)
        try:
            opt_res = opt.least_squares(self.quad_residuals,Qparams_0,args=(x,y))
            output = opt_res.x
        except ValueError as exc:
            self.print_this(exc.args,self.logfile)
            output = 0.0*Qparams_0
            flag_ok_qfit = False

        return output,flag_ok_qfit
    ############################################################
    ############################################################

    ############################################################
    def extract_cell_coord(self,c,Ng,D):
        """ Convenience function on Ng-grid in D-dimensions 
            to convert integer c between 0..Ng**D into integer cell coordinates (d1,..,dD)
            Returns list of length D.
        """
        if (c < 0) | (c >= Ng**D):
            raise ValueError('Invalid value for c in extract_cell_coord().')
        ceval = 1*c
        dtup = [0]*D
        for i in range(1,D):
            di = ceval // Ng**(D-i)
            dtup[i-1] = di
            ceval -= di*Ng**(D-i)
        dtup[D-1] = ceval % Ng
        return dtup
    ############################################################


    ############################################################
    def model_poly(self,x,params):
        """ Polynomial in x, of degree n = len(params)-1. 
            Expect x to be scalar or numpy array (arbitrary dimensions).
            Expect params to be list ordered as a_0,a_1..a_n
            where y = sum_i a_i x**i.
            Returns array y of shape x.shape.
        """
        if len(params) == 1:
            out = params[0]*np.ones_like(x)
        else:
            out = np.sum(np.array([params[i]*x**i for i in range(len(params))]),axis=0)
        return out
    ############################################################


    ############################################################
    def polyfit_custom(self,x,y,deg,sig2=None,start=0):
        """ Polynomial fit of degree deg to data y at locations x.
            Optionally pass squared errors sig2 on y.
            Minimises chi2 = sum_i (y_i - p(x_i))^2 / sig2_i
            with p(x) = sum_alpha a[alpha]*x^alpha
            for alpha=start..deg.
            Returns minimum variance estimator a[alpha] 
            and covariance matrix C[alpha,beta].

            Not very well tested, so use with care. """

        Y = np.zeros(deg+1-start,dtype=float) 

        # Matrix
        F = np.zeros((deg+1-start,deg+1-start),dtype=float)

        if sig2 is None:
            sig2 = np.ones(x.size,dtype=float)

        for alpha in range(start,deg+1):
            Y[alpha-start] = np.sum(y*(x**(alpha))/sig2)
            for beta in range(start,deg+1):
                F[alpha-start,beta-start] = np.sum(x**(alpha+beta)/sig2)
                F[beta-start,alpha-start] = F[alpha-start,beta-start]

        Y = Y.T
        U,s,Vh = linalg.svd(F)
        Cov = np.dot(Vh.T,np.dot(np.diag(1.0/s),U.T))
        a_minVar = Cov.dot(Y)

        return np.squeeze(np.asarray(a_minVar)),np.asarray(Cov)
    ############################################################

        
    ############################################################
    def error_to_ellipse(self,err_x,err_y,cc,dchi2=2.3):
        """ Use 2-d covariance matrix to return matplotlib ellipse plotting parameters. """
        ell_angle = 0.5*np.arctan(2*cc*err_x*err_y/(err_x**2-err_y**2))
        a2inv = (np.cos(ell_angle)**2/err_x**2 + np.sin(ell_angle)**2/err_y**2 
                 - cc*np.sin(2*ell_angle)/err_x/err_y)/(1-cc**2)
        b2inv = (np.cos(ell_angle)**2/err_y**2 + np.sin(ell_angle)**2/err_x**2 
                 + cc*np.sin(2*ell_angle)/err_x/err_y)/(1-cc**2)
        ell_width = 2*np.sqrt(1/a2inv*dchi2)
        ell_height = 2*np.sqrt(1/b2inv*dchi2)
        ell_angle *= 180./np.pi

        return ell_width,ell_height,ell_angle
    ############################################################


    ############################################################
    def time_this(self,start_time):
        totsec = time() - start_time
        minutes = int(totsec/60)
        seconds = totsec - 60*minutes
        self.print_this("{0:d} min {1:.2f} seconds\n".format(minutes,seconds),self.logfile)
        return
    ############################################################


    ############################################################
    def print_this(self,print_string,logfile=None,overwrite=False):
        """ Convenience function for printing to logfile or stdout."""
        if logfile is not None:
            self.writelog(logfile,print_string+'\n',overwrite=overwrite)
        else:
            print(print_string)
        return
    ############################################################



    ############################################################
    def write_structured(self,filestring,recarray,dlmt=' '):
        """ Opens filestring for appending and writes 
        rows of structured array recarray to it.
        """
        with open(filestring,'a') as f:
            for row in recarray:
                f.write(dlmt.join([str(item) for item in row]))
                f.write('\n')
        return
    ############################################################


    ############################################################
    def writelog(self,logfile,strng,overwrite=False):
        """ Convenience function for pipe-safety. """
        app_str = 'w' if overwrite else 'a'
        with open(logfile,app_str) as g:
            g.write(strng)
        return
    ############################################################


    ############################################################
    def status_bar(self,n,ntot,freq=100,text='done'):
        """ Print status bar with user-defined text and frequency. """
        if freq > ntot:
            freq = ntot
        if ((n+1) % int(1.0*ntot/freq) == 0):
            frac = (n+1.)/ntot
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %.f%% " % ('.'*int(frac*20),100*frac) + text)
            sys.stdout.flush()
        if n==ntot-1: self.print_this('',self.logfile)
        return
    ############################################################


#######################################################################

if __name__=="__main__":
    ut = Utilities()
    start_time = time()
    params_true = np.array([1.0,1.0,1.0])
    dim = len(params_true) # parameter dimensionality

    xvals = np.linspace(-1.0,1.0,20)
    err_data = 0.5*np.ones_like(xvals)
    invcov_data = 1.0/err_data**2

    SEED = 1983 
    rng = np.random.RandomState(seed=SEED)

    data = ut.model_poly(xvals,params_true) + err_data*rng.randn(xvals.size)

    apoly,cov_poly = ut.polyfit_custom(xvals,data,dim-1,sig2=err_data**2)
    sig_apoly = np.sqrt(np.diagonal(cov_poly))

    # print('Best poly fit...')
    # print("... values ( a0,..,a{0:d} ) = ( ".format(dim-1)
    #       +','.join(['%.3f' % (ai,) for ai in apoly])+" )")
    # print("... errors ( sig_a0,..,sig_a{0:d} ) = ( ".format(dim-1)
    #       +','.join(['%.3f' % (sigi,) for sigi in sig_apoly])+" )")
    # print("... full cov = \n",cov_poly)

    chi2_analytical = np.sum(((data - ut.model_poly(xvals,apoly))/err_data)**2)
    # print('... chi2 = {0:.2f}'.format(chi2_analytical))
    # print('... done')

    def cost_func(data,invcov_data,model,cost_args):
        return np.sum(((model - data))**2*invcov_data)

    param_mins = [-4.0]*dim
    param_maxs = [4.0]*dim

    NSAMP = 15
    EPS_CONV = 50.
    LAST_FRAC = 0.1

    dof = xvals.size - dim

    ut.print_this('----------TEST 0-----------')
    chi2,params = ut.asa_chi2min(data,invcov_data,param_mins,param_maxs,ut.model_poly,xvals,cost_func,None,
                                 Nsamp=NSAMP,rng=rng,eps_conv=EPS_CONV,last_frac=LAST_FRAC,
                                 read_file=None,
                                 write_out='test.txt',over_write=True,last_iters=None)
    ut.print_this('---------------------------')

    ut.print_this('----------TEST 1-----------')
    chi2,params = ut.asa_chi2min(data,invcov_data,param_mins,param_maxs,ut.model_poly,xvals,cost_func,None,
                                 Nsamp=NSAMP,rng=rng,eps_conv=EPS_CONV,last_frac=LAST_FRAC,
                                 read_file=None,
                                 write_out='test.txt',over_write=False,last_iters=None)
    ut.print_this('---------------------------')

    ut.print_this('----------TEST 2-----------')
    chi2,params = ut.asa_chi2min(data,invcov_data,param_mins,param_maxs,ut.model_poly,xvals,cost_func,None,
                                 Nsamp=NSAMP,rng=rng,eps_conv=EPS_CONV,last_frac=LAST_FRAC,
                                 read_file='test.txt',
                                 write_out='test.txt',over_write=True,last_iters=None)
    ut.print_this('---------------------------')

    ut.print_this('----------TEST 3-----------')
    chi2,params = ut.asa_chi2min(data,invcov_data,param_mins,param_maxs,ut.model_poly,xvals,cost_func,None,
                                 Nsamp=NSAMP,rng=rng,eps_conv=EPS_CONV,last_frac=LAST_FRAC,
                                 read_file='test.txt',
                                 write_out='test.txt',over_write=True,last_iters=4)
    ut.print_this('---------------------------')

    ut.print_this('----------TEST 4-----------')
    chi2,params = ut.asa_chi2min(data,invcov_data,param_mins,param_maxs,ut.model_poly,xvals,cost_func,None,
                                 Nsamp=NSAMP,rng=rng,eps_conv=EPS_CONV,last_frac=LAST_FRAC,
                                 read_file='test.txt',
                                 write_out='test.txt',over_write=True,last_iters=[4])
    ut.print_this('---------------------------')


    # ut.print_this('----------TEST 5-----------')
    # chi2,params = ut.asa_chi2min(data,invcov_data,param_mins,param_maxs,ut.model_poly,xvals,cost_func,None,
    #                              Nsamp=NSAMP,rng=rng,eps_conv=EPS_CONV,last_frac=LAST_FRAC,
    #                              read_file='test.txt',
    #                              write_out='test.txt',over_write=True,last_iters=[4,3,2,1])
    # ut.print_this('---------------------------')

    # imin = chi2.argmin()

    # cov_asa,params_best,chi2_min = ut.asa_get_bestfit(chi2,params)
    # sig_asa = np.sqrt(np.diag(cov_asa))
    # cc_asa = np.zeros(dim*(dim-1)//2,dtype=float)
    # cond_sig_asa = np.zeros_like(cc_asa)
    # c = 0
    # for i in range(dim-1):
    #     for j in range(i+1,dim):
    #         cc_asa[c] = cov_asa[i,j]/(sig_asa[i]*sig_asa[j])
    #         cond_sig_asa[c] = sig_asa[i]*np.sqrt(1 - cc_asa[c]**2)
    #         c += 1


    # print('\nBest ASA fit (evaluated)...')
    # print("... global min ( a0,..,a{0:d} ) = ( ".format(dim-1)
    #       +','.join(['%.3f' % (pval,) for pval in params[imin]])+" )")
    # print("... % residuals ( a0,..,a{0:d} ) = ( ".format(dim-1)
    #       +','.join(['%.1e' % (pval,) for pval in np.fabs(params[imin]/apoly - 1.0)*1e2])+" )")

    # print('... chi2 = {0:.4f} (analytical = {1:.4f})'.format(chi2[imin],chi2_analytical))


    # print('\nBest ASA fit (quadratic estimate)...')
    # print("... global min ( a0,..,a{0:d} ) = ( ".format(dim-1)
    #       +','.join(['%.3f' % (pval,) for pval in params_best])+" )")
    # print("... % residuals ( a0,..,a{0:d} ) = ( ".format(dim-1)
    #       +','.join(['%.1e' % (pval,) for pval in np.fabs(params_best/apoly - 1.0)*1e2])+" )")

    # print('... chi2 = {0:.4f} (analytical = {1:.4f})'.format(chi2_min,chi2_analytical))
    # print('\n... cov mat difference:')
    # print(cov_asa-cov_poly)

    print('\n... total number of function evaluations = {0:d}'.format(chi2.size))

    ut.time_this(start_time)
