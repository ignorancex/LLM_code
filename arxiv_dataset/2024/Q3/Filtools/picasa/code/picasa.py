import os
import numpy as np
from cobaya.run import run
from cobaya.log import LoggedError
from asa_utilities import ASAUtilities
from getdist.mcsamples import loadMCSamples
from time import time
import copy

class PICASA(ASAUtilities):
    """ Higher level functions for using PICASA. """

    def __init__(self,base_dir=None,logfile=None):
        ASAUtilities.__init__(self,base_dir=base_dir,logfile=logfile)


    ############################################################
    def mcmc(self,data_pack):
        """ Wrapper to intelligently call ASAUtilities.asa_chi2min() repeatedly 
             to simultaneoulsy optimize the cost function while generating a sample for use with Cobaya.
             -- data_pack: dictionary containing the following keys:
                -----------------------
                Mandatory keys:
                -----------------------
                [Used only by mcmc()]
                'chain_dir': str - path to output directory to store GPR training products and MCMC chains 
                                    (needs write access)

                [Used by mcmc() and optimize()]
                'chi2_file': str - path to output file to store chi2|params
                'dsig': float - largest multiple of std dev in each parameter direction 
                                     for generating last iterations and (optionally) setting MCMC prior.
                'param_mins','param_maxs': lists (or arrays) of floats - initial minimum and maximum values for parameters 

                [Only passed to optimize()]
                'Nsamp': int - number of samples per iteration
                'model_func': callable required by ASAUtilities.asa_chi2min()
                'model_args': object required by ASAUtilities.asa_chi2min(), compatible with model_func
                If 'no_cost' is not set or False (default), then additionally supply:
                    'data': (Ndata,) array containing observed data values
                    'invcov_data': (Ndata,Ndata)  array containing inverse of data covariance matrix
                    'cost_func': callable required by ASAUtilities.asa_chi2min()
                    'cost_args': object required by ASAUtilities.asa_chi2min(), compatible with cost_func

                -----------------------
                Optional keys:
                -----------------------
                [Used only by mcmc()]
                'read_file_mcmc': None or str - If not None, should be valid path to file containing chi2|params. 
                                               In this case, optimize() will not be called. Instead, the requested file will be read
                                               and used to estimate a best-fit quadratic form, which is then supplied to 
                                               gpr_train_and_run() or run_trained_gpr().
                'max_opt_iter': int - maximum number of time to try optimize() with different starting ranges.
                                          (Default value 10)
                'prop_fac': float - factor to multiply proposal covariance matrix in mcmc sampler.
                                  (Default value 0.1)
                'kernel': str - kernel defining GPR. Accepted values are {'rbf','rq'}
                               (Default value 'rbf')
                'latex': None or list of strings - passed to gpr_train_and_run() or run_trained_gpr().
                            (Default value None)
                'burn_in': int. Number of burn in samples to discard in MCMC 
                                 passed to gpr_train_and_run() or run_trained_gpr().
                                 (Default value 1000)
                'Rminus1_stop': float. Stopping criterion (convergence of means) for MCMC 
                                            passed to gpr_train_and_run() or run_trained_gpr().
                                            (Default value 0.005)
                'Rminus1_cl_stop': float. Stopping criterion (convergence of 95% bounds) for MCMC 
                                                 passed to gpr_train_and_run() or run_trained_gpr().
                                                 (Default value 0.2)
                'no_training': boolean. If set and True, by-pass GPR training and directly call run_trained_gpr().
                                       This assumes that the .tmp files created by gpr_train_and_run() exist at the correct location.
                                       Useful if stuck chains need to be re-run with modified 'prop_fac'.
                                       (Default False)
                'robust_convergence': boolean. Passed to gpr_train_and_run().
                                                      (Default True)
                'cv_thresh': float. stopping criterion for GPR training passed to gpr_train_and_run().
                                    (Default value 0.01)
                'max_gpr_iter': int. Maximum number of GPR training iterations.
                                          (Default value 25)
                'verbose_training': boolean controlling verbosity of GPR training.
                                                (Default False)
                'mcmc_prior': None or list of [min,max] pairs specifying boundaries of uniform priors for all parameters.
                                         If None, key 'dsig' will be used to set priors. (Default value None)

                [Passed to optimize()]
                'last_frac': float controlling size of last iterations, passed to second call of ASAUtilities.asa_chi2min().
                                  (Default value 0.5)
                'C_sig': factor to increase Gaussian widths in parameter eigenspace, to approximately account for
                             non-Gaussian shape of exp(-chi2/2) when setting up last iterations. (Default value 1.0)
                'eps_conv': float controlling ASA convergence, passed to second call of ASAUtilities.asa_chi2min().
                                  (Default value 0.1)
                'use_coarse': If set and equal to True, coarse sample from first call to ASAUtilities.asa_chi2min()
                                       will be included when estimating quadratic forms, else will only be included in the final sample.
                                       (Default True)    
                'no_cost': If set and equal to True, only 'model_func' and 'model_args' to be specified in data_pack, 
                                  and user should ensure that model_func(model_args,params) returns value of chi2 
                                  (i.e. quantity to be minimised).
                                  If False, 'data', 'invcov_data', 'cost_func' and 'cost_args' should be separately provided.
                                  (Default False)
                'Ndata': If 'no_cost' is true, then 'Ndata' must be set to the length of the data vector.
                              This will be used to calculate number of degrees of freedom as dof = Ndata - dim
                              where dim is parameter dimensionality, and goodness of fit during ASA convergence
                              test will assume chi2_min is chi2-distributed with dof degrees of freedom. 
                'parallel': boolean controlling parallel processing, passed to both calls of ASAUtilities.asa_chi2min().
                                  (Default False)

                [Used by mcmc() and optimize()]
                'rng': None or instance of numpy.random.RandomState(), passed to both calls of ASAUtilities.asa_chi2min()
                          and gpr_train_and_run() and run_trained_gpr().
                'verbose': boolean controlling verbosity.
                                  (Default True)
                                  
                'only_ASA': If True, does not perform mcmc (Default False)
        
            Given the data_pack, this routine first calls optimize() to efficiently sample chi2 values.
            These samples are then used to iteratively train a GPR interpolator using gpr_train_and_run(). 
            The interpolator is then used to perform a standard MCMC using run_trained_gpr().
            Optionally, a pre-trained GPR can be used to directly perform the MCMC using the key 'no_training'.

            Returns:
            -- info,sampler: outputs of cobaya.run.run()
            -- chi2, params: sampled points and chi2 values (shapes (Neval,), (Neval,dim))
            -- cov_asa,params_best,chi2_min: best-fit quadratic form, output by ASAUtilities.asa_get_bestfit().
        """

        dp = copy.deepcopy(data_pack)

        # look for optional keys, if not found then set to default values
        opt = {'max_opt_iter':10,'prop_fac':0.1,'kernel':'rbf','latex':None,'burn_in':1000,'rotate':None,
               'Rminus1_stop':0.005,'Rminus1_cl_stop':0.2,'no_training':False,
               'robust_convergence':True,'cv_thresh':1e-2,'rng':None,'max_gpr_iter':25,
               'verbose_training':False,'verbose':True,'read_file_mcmc':None,'mcmc_prior':None, 'only_ASA':False}
        for key in opt.keys():
            if key in dp.keys():
                opt[key] = dp[key]

        if opt['verbose']:
            self.print_this('***********************\n-----------------------\n'
                            +'PICASA\n-----------------------\n***********************',self.logfile)
            self.print_this('Extracting MCMC parameters',self.logfile)

        expected_keys = ['chain_dir','chi2_file','dsig']
        missing_key = False
        for key in expected_keys:
            if key not in dp.keys():
                self.print_this("Key '"+key+"' missing in data_pack used by PICASA.mcmc()",self.logfile)
        if missing_key:
            raise KeyError('Missing key(s) detected. Try again.')

        dim = len(dp['param_mins'])

        if opt['verbose']:
            self.print_this('... done',self.logfile)

        if opt['read_file_mcmc'] is not None:
            read_file = self.base_dir + opt['read_file_mcmc']
            if os.path.isfile(read_file):
                if opt['verbose']:
                    self.print_this("... reading chi2|params data from file: "+read_file,self.logfile)
                    self.print_this("... setting chi2_file to: "+read_file,self.logfile)
                chi2,params = self.read_data_from_file(opt['read_file_mcmc'])
                dp['chi2_file'] = opt['read_file_mcmc']
                cov_asa,params_best,chi2_min,eigvals,rotate = self.asa_get_bestfit(chi2,params)
                opt['rotate'] = rotate
            else:
                raise ValueError('Invalid file path specified for reading in PICASA.mcmc()')
        else:
            if opt['verbose']:
                self.print_this('... optimizing',self.logfile)
            opt_iter = 0
            delta_params = np.array(dp['param_maxs']) - np.array(dp['param_mins'])
            while opt_iter < opt['max_opt_iter']:
                chi2,params,cov_asa,params_best,chi2_min,eigvals,rotate,flag_ok = self.optimize(dp)
                opt['rotate'] = rotate
                if np.isscalar(chi2):
                    raise ValueError('Slow convergence detected. Exiting without output to avoid unnecessary evaluations.')
                if flag_ok:
                    if opt['verbose']:
                        self.print_this('... successful optimization by ASA!',self.logfile)
                    if opt['only_ASA']:
                        return params_best, chi2_min
                    break
                if opt['verbose']:
                    self.print_this('... retrying optimization for {0:d}th time, centered on current best-fit with half original range'
                                    .format(opt_iter+1),self.logfile)
                # at this stage, params_best returned by optimize() is not reliable, so use evaluated minimum.
                # modified 03 Nov 22
                # added by saee

                dp['param_mins'] = params[chi2.argmin()] - 0.25*delta_params
                dp['param_maxs'] = params[chi2.argmin()] + 0.25*delta_params
                # dp['param_mins'] = params_best - 0.25*delta_params
                # dp['param_maxs'] = params_best + 0.25*delta_params
                opt_iter += 1
                

            if opt_iter >= opt['max_opt_iter']:
                if opt['verbose']:
                    self.print_this('... maximum number of restarts reached for PICASA.optimize() without convergence. '
                                    + 'Try manually adjusting starting parameter range. Returning [0]*7',self.logfile)
                return [0]*7

        if opt['verbose']:
            self.print_this('***********************\n'
                            +'GPR training and MCMC\n***********************',self.logfile)
        if opt['no_training']:
            if opt['verbose']:
                self.print_this('... ** N.B **: using pre-trained GPR kernel',self.logfile)
            info,sampler = self.run_trained_gpr(params_best,cov_asa,dp['chi2_file'],dp['chain_dir'],
                                                dsig=dp['dsig'],prop_fac=opt['prop_fac'],burn_in=opt['burn_in'],
                                                rotate=opt['rotate'],verbose=opt['verbose_training'],kernel=opt['kernel'],
                                                latex_symb=opt['latex'],rng=opt['rng'],mcmc_prior=opt['mcmc_prior'],
                                                Rminus1_stop=opt['Rminus1_stop'],Rminus1_cl_stop=opt['Rminus1_cl_stop'])
        else:
            if opt['verbose']:
                self.print_this('... training GPR kernel and performing MCMC',self.logfile)
            info,sampler = self.gpr_train_and_run(params_best,cov_asa,dp['chi2_file'],dp['chain_dir'],
                                                  dsig=dp['dsig'],prop_fac=opt['prop_fac'],burn_in=opt['burn_in'],
                                                  rotate=opt['rotate'],robust_convergence=opt['robust_convergence'],
                                                  cv_thresh=opt['cv_thresh'],max_iter=opt['max_gpr_iter'],
                                                  verbose=opt['verbose_training'],kernel=opt['kernel'],
                                                  latex_symb=opt['latex'],rng=opt['rng'],mcmc_prior=opt['mcmc_prior'],
                                                  Rminus1_stop=opt['Rminus1_stop'],Rminus1_cl_stop=opt['Rminus1_cl_stop'])

        gd_sample = loadMCSamples(os.path.abspath(info["output"]))
        # samples contain a0..a(dim-1) chi2 | chi2__name
        ibest = gd_sample.samples.T[dim].argmin()
        mcmc_best = gd_sample.samples[ibest,:dim]
        mcmc_16pc = np.percentile(gd_sample.samples,16,axis=0)[:dim]
        mcmc_50pc = np.percentile(gd_sample.samples,50,axis=0)[:dim]
        mcmc_84pc = np.percentile(gd_sample.samples,84,axis=0)[:dim]
        mcmc_chi2 = gd_sample.samples[ibest,dim]
        if opt['verbose']:
            self.print_this('... MCMC done',self.logfile)
            self.print_this("... best fit ( a0,..,a{0:d} ) = ( ".format(dim-1)
                            +','.join(['%.3f' % (pval,) for pval in mcmc_best])+" )",self.logfile)
            self.print_this("... percentiles (16,50,84):",self.logfile)
            for p in range(dim):
                self.print_this("... ... a{0:d}: {1:.3e},{2:.3e},{3:.3e}".format(p,mcmc_16pc[p],mcmc_50pc[p],mcmc_84pc[p]),
                                self.logfile)
            self.print_this("... chi2: {0:.3f}".format(mcmc_chi2),self.logfile)

        sig_mcmc = 0.5*(mcmc_84pc-mcmc_16pc)

        flag_final_ok = True
        for p in range(dim):
            if np.fabs(params_best[p] - mcmc_best[p]) > sig_mcmc[p]:
                self.print_this('!! WARNING !!: MCMC best fit is more than 1sigma away'
                                +' from estimated (quadratic form) best fit for parameter a{0:d}.'.format(p),self.logfile)
                flag_final_ok = False
        if not flag_final_ok:
            self.print_this('Consider re-training the GPR with' 
                            + ' a more stringent cv_thresh and/or robust_convergence = True.',self.logfile)

        return info,sampler,chi2,params,cov_asa,params_best,chi2_min

    ############################################################


    ############################################################
    def optimize(self,data_pack):
        """ Wrapper to intelligently call ASAUtilities.asa_chi2min() repeatedly 
             to simultaneoulsy optimize the cost function while generating a sample for use with Cobaya.
             -- data_pack: dictionary containing the following keys [subset of keys used by PICASA.mcmc()]:
                -----------------------
                Mandatory keys:
                -----------------------
                'chi2_file': str - path to output file to store chi2|params
                'Nsamp': int - number of samples per iteration
                'dsig': float - largest multiple of std dev in each parameter direction for generating last iterations
                'param_mins','param_maxs': lists (or arrays) of floats - initial minimum and maximum values for parameters 
                'model_func': callable required by ASAUtilities.asa_chi2min()
                'model_args': object required by ASAUtilities.asa_chi2min(), compatible with model_func
                If 'no_cost' is not set or False (default), then additionally supply:
                    'data': (Ndata,) array containing observed data values
                    'invcov_data': (Ndata,Ndata)  array containing inverse of data covariance matrix
                    'cost_func': callable required by ASAUtilities.asa_chi2min()
                    'cost_args': object required by ASAUtilities.asa_chi2min(), compatible with cost_func
                -----------------------
                Optional keys:
                -----------------------
                'last_frac': float controlling size of last iterations, passed to second call of ASAUtilities.asa_chi2min().
                                  (Default value 0.5)
                'C_sig': factor to increase Gaussian widths in parameter eigenspace, to approximately account for
                             non-Gaussian shape of exp(-chi2/2) when setting up last iterations. (Default value 1.0)
                'eps_conv': float controlling ASA convergence, passed to second call of ASAUtilities.asa_chi2min().
                                  (Default value 0.1)
                'use_coarse': If set and equal to True, coarse sample from first call to ASAUtilities.asa_chi2min()
                                       will be included when estimating quadratic forms, 
                                       else will only be included in the final sample. (Default True)    
                'no_cost': If set and equal to True, only 'model_func' and 'model_args' to be specified in data_pack, 
                                  and user should ensure that model_func(model_args,params) returns value of chi2 
                                  (i.e. quantity to be minimised).
                                  If False, 'data', 'invcov_data', 'cost_func' and 'cost_args' should be separately provided.
                                  (Default False)
                'Ndata': If 'no_cost' is true, then 'Ndata' must be set to the length of the data vector.
                              This will be used to calculate number of degrees of freedom as dof = Ndata - dim
                              where dim is parameter dimensionality, and goodness of fit during ASA convergence
                              test will assume chi2_min is chi2-distributed with dof degrees of freedom. 
                'rng': None or instance of numpy.random.RandomState(), passed to both calls of ASAUtilities.asa_chi2min().
                'parallel': boolean controlling parallel processing, passed to both calls of ASAUtilities.asa_chi2min().
                                  (Default False)
                'verbose': boolean controlling verbosity, passed to all calls of ASAUtilities.asa_chi2min(). 
                                  (Default True)
        
            Given the data_pack, this routine performs a first call to ASAUtilities.asa_chi2min() using a 
            sampling determined by the value of Nsamp, param_mins and param_maxs and a coarse convergence
            criterion. It then zooms in on the minimum of the sampled chi2 values and performs a second call to
            ASAUtilities.asa_chi2min(), with parameter ranges given by the width of the anisotropic best-fit quadratic form
            (or 0.1 times the initial ranges, if a positive definite quadratic form is not obtained by the initial chain).
            These two samples (the 'minimiser-mode' output) provide an estimate of the best-fit quadratic form, 
            which is then used to perform a series of last iterations controlled by dsig. 
            The final sample is output, along with its resulting best-fit quadratic form and a boolean flag. 
            A warning is raised if the best-fit parameters from the evaluated minimum and quadratic form differ by 
            more than 1-sigma after the minimiser-mode (in which case only the minimiser-mode output is returned)
            or after the last iterations (full output is returned). 
            If the boolean flag returned is False, it is highly recommended to repeat the call to optimize()
            by modifying the starting range of parameter values (keys 'param_mins' and 'param_maxs').

            Returns:
            -- chi2, params: sampled points and chi2 values (shapes (Neval,), (Neval,dim))
            -- cov_asa,params_best,chi2_min,eigvals,rotate: best-fit quadratic form 
               and related output generated by ASAUtilities.asa_get_bestfit().
            -- flag_ok: boolean. True if no warnings raised, False otherwise.
        """

        flag_ok = True
        dp = copy.deepcopy(data_pack)

        # look for optional keys, if not found then set to default values
        opt = {'last_frac':0.5,'eps_conv':0.1,'no_cost':False,'use_coarse':True,
               'rng':None,'parallel':False,'verbose':True,'Ndata':0,'C_sig':1.0}
        for key in opt.keys():
            if key in dp.keys():
                opt[key] = dp[key]

        if opt['verbose']:
            self.print_this('***********************\n'
                            +'ASA optimiser\n***********************',self.logfile)
            self.print_this('Extracting data products and ASA parameters',self.logfile)

        expected_keys = ['chi2_file','Nsamp','dsig','model_func','model_args','param_mins','param_maxs']
        if not opt['no_cost']:
            expected_keys += ['data','invcov_data','cost_func','cost_args']
        missing_key = False
        for key in expected_keys:
            if key not in dp.keys():
                self.print_this("Key '"+key+"' missing in data_pack used by PICASA.optimize()",self.logfile)
        if missing_key:
            raise KeyError('Missing key(s) detected. Try again.')

        if opt['no_cost']:
            # in this mode, user takes responsibility for passing data products to model_func (using model_args)
            # such that output of model_func() is itself the value of chi2. 
            # length of data then needs to be specified by the key 'Ndata'
            if opt['Ndata'] == 0:
                raise ValueError("If key 'no_cost' is True, key 'Ndata' must be a positive integer.")
            dp['cost_args'] = None
            dp['data'] = np.zeros(opt['Ndata'],dtype=float)
            dp['invcov_data'] = np.zeros_like(dp['data'])
            def cost_func(data,invcov_data,model,cost_args):
                return model
            dp['cost_func'] = cost_func

        if opt['verbose']:
            self.print_this('... done',self.logfile)
            self.print_this('-----------------------\nPerforming coarsely sampled ASA\n-----------------------',self.logfile)
        Nzoom_coarse = 50
        chi2,params = self.asa_chi2min(dp['data'],dp['invcov_data'],dp['param_mins'],dp['param_maxs'],
                                           dp['model_func'],dp['model_args'],dp['cost_func'],dp['cost_args'],
                                           Nsamp=dp['Nsamp'],rng=opt['rng'],eps_conv=100*opt['eps_conv'],
                                           Nzoom=Nzoom_coarse,
                                           write_out=dp['chi2_file'],over_write=True,parallel=opt['parallel'],verbose=opt['verbose'])
        ######################
        # Handling slow convergence... better to restart at this stage than wait for improper GPR training
        ######################
        if chi2.size == dp['Nsamp']*Nzoom_coarse:
            self.print_this('!! WARNING !! Coarse iteration may not have properly converged. '
                            + '{Change rng seed} and/or {decrease starting range and/or increase Nsamp}. Returning zeros.')
            return 0,0,0,0,0,0,0,False
        ######################
        cov_asa,params_best,chi2_min,eigvals,rotate = self.asa_get_bestfit(chi2,params)
        if np.all(np.isfinite(eigvals)):
            if opt['verbose']:
                self.print_this('... rotating coarse result to eigenspace',self.logfile)
            params_best = np.dot(rotate.T,params_best)
            dsig_vals = np.sqrt(eigvals) 
        else:
            if opt['verbose']:
                self.print_this('... keeping coarse result in parameter space (centered on evaluated minimum)',self.logfile)
            rotate = None
            dsig_vals = 0.1*(np.array(dp['param_maxs'])-np.array(dp['param_mins']))
            params_best = params[chi2.argmin()]

        dim = params_best.size

        if opt['verbose']:
            self.print_this('-----------------------\nFocusing on coarse best-fit estimate\n-----------------------',self.logfile)
        param_mins_upd = params_best - dsig_vals 
        param_maxs_upd = params_best + dsig_vals

        Nzoom_minimiser = 100
        Neval_coarse = chi2.size
        Neval_start = 0 if opt['use_coarse'] else Neval_coarse
        chi2,params = self.asa_chi2min(dp['data'],dp['invcov_data'],param_mins_upd,param_maxs_upd,
                                       dp['model_func'],dp['model_args'],dp['cost_func'],dp['cost_args'],rotate=rotate,
                                       Nsamp=dp['Nsamp'],rng=opt['rng'],eps_conv=opt['eps_conv'],
                                       write_out=dp['chi2_file'],over_write=False,Nzoom=Nzoom_minimiser,
                                       parallel=opt['parallel'],verbose=opt['verbose'])
        ######################
        # Handling slow convergence... better to restart at this stage than wait for improper GPR training
        ######################
        if chi2.size == dp['Nsamp']*Nzoom_minimiser:
            self.print_this('!! WARNING !! Minimiser-mode may not have properly converged. '
                            + '{Change rng seed} and/or {decrease starting range and/or increase Nsamp}. Returning zeros.')
            return 0,0,0,0,0,0,0,False
        ######################
        if opt['verbose']:
            if opt['use_coarse']:
                self.print_this('... including coarse sample in quadratic form estimates',self.logfile)
            else:
                self.print_this('... excluding coarse sample from quadratic form estimates',self.logfile)
        cov_asa,params_best,chi2_min,eigvals,rotate = self.asa_get_bestfit(chi2[Neval_start:],params[Neval_start:])
        if np.any(~np.isfinite(eigvals)):
            flag_ok = False
            self.print_this('!! WARNING !!: Diagonalisation failed after minimiser mode',self.logfile)
        else:
            sig_asa = np.sqrt(np.diag(cov_asa))
            imin = chi2[Neval_start:].argmin()
            for p in range(dim):
                if np.fabs(params_best[p] - params[Neval_start:][imin,p]) > sig_asa[p]:
                    flag_ok = False
                    self.print_this('!! WARNING !!: Evaluated minimum is more than 1sigma away'
                                    +' from estimated best fit for parameter a{0:d} in minimiser-mode.'.format(p),self.logfile)

        if not flag_ok:
            self.print_this('Exiting with minimiser-mode output. '
                            + ' Change starting range (param_mins and/or param_maxs) in data_pack.',self.logfile)
        else:
            if opt['verbose']:
                self.print_this('... all seems fine, proceeding with last iterations [excluding first {0:d} samples from setup].'
                                .format(Neval_start),self.logfile)
            Ds = 0.25
            n_last = int(dp['dsig']/Ds)
            s_full = np.linspace(0,dp['dsig'],n_last+1)
            last_iters = list(np.delete(s_full,0)) # ASA_Utilities.asa_chi2min() expects a list
            ###################
            chi2,params = self.asa_chi2min(dp['data'],dp['invcov_data'],param_mins_upd,param_maxs_upd,
                                           dp['model_func'],dp['model_args'],dp['cost_func'],dp['cost_args'],
                                           Nsamp=dp['Nsamp'],rng=opt['rng'],eps_conv=opt['eps_conv'],
                                           last_iters=last_iters,last_frac=opt['last_frac'],burn_in_iter=Neval_start,C_sig=opt['C_sig'],
                                           read_file=dp['chi2_file'],
                                           write_out=dp['chi2_file'],over_write=True,parallel=opt['parallel'],verbose=opt['verbose'])
            cov_asa,params_best,chi2_min,eigvals,rotate = self.asa_get_bestfit(chi2,params)
            if np.any(~np.isfinite(eigvals)):
                flag_ok = False
                self.print_this('!! WARNING !!: Diagonalisation failed after last iterations',self.logfile)
            else:
                sig_asa = np.sqrt(np.diag(cov_asa))
                imin = chi2.argmin()
                for p in range(dim):
                    if np.fabs(params_best[p] - params[imin,p]) > 0.5*sig_asa[p]:
                        flag_ok = False
                        self.print_this('!! WARNING !!: Evaluated minimum is more than 0.5sigma away'
                                        +' from estimated best fit for parameter a{0:d} after all last iterations!'.format(p),self.logfile)

            if opt['verbose']:
                self.print_this('-----------------------\n'
                                +'Optimisation complete.\nReturning {0:d} ASA samples'.format(chi2.size)
                                +' and best fit estimates\n-----------------------',self.logfile)
                self.print_this('Evaluated optimum...',self.logfile)
                self.print_this("... best-fit ( a0,..,a{0:d} ) = ( ".format(dim-1)
                                +','.join(['%.3f' % (pval,) for pval in params[imin]])+" )",self.logfile)
                self.print_this('... chi2 = {0:.4f})'.format(chi2[imin]),self.logfile)
                #
                self.print_this('\nQuadratic form estimate...',self.logfile)
                self.print_this("... best-fit ( a0,..,a{0:d} ) = ( ".format(dim-1)
                                +','.join(['%.3f' % (pval,) for pval in params_best])+" )",self.logfile)
                self.print_this("... errors ( sig_a0,..,sig_a{0:d} ) = ( ".format(dim-1)
                                +','.join(['%.3f' % (pval,) for pval in sig_asa])+" )",self.logfile)
                self.print_this('... chi2 = {0:.4f}'.format(chi2_min),self.logfile)
            if not flag_ok:
                self.print_this('Consider changing starting range (param_mins and/or param_maxs) ' 
                                + 'for some parameters (see warnings above).',self.logfile)

        return chi2,params,cov_asa,params_best,chi2_min,eigvals,rotate,flag_ok

    ############################################################

    ############################################################
    def gpr_train_and_run(self,params_best,cov_asa,chi2_file,chain_dir,max_iter=25,train_size_start=None,
                          rotate=None,robust_convergence=True,cv_thresh=0.01,dsig=5.0,prop_fac=0.1,burn_in=500,
                          exclude_from_start=0,rng=None,mcmc_prior=None,
                          kernel='rbf',verbose=False,latex_symb=None,
                          Rminus1_stop=0.1,Rminus1_cl_stop=0.2,force=True):
        """ Wrapper to train GPR and run Cobaya.
            -- max_iter: maximum number of training iterations.
            -- train_size_start: None or int. starting training size.
            -- exclude_from_start: int. number of parameter points to exclude from start of sample.
            -- robust_convergence: controls whether to use 1,99 (if True) or 16,84 percentiles (if False) of cross-validation
                                                    residuals in judging convergence.
            -- cv_thresh: threshold on max of abs value of cross-validation residual 1,99 or 16,84 percentiles.
            -- see ASAUtilities.create_info_dict() for remaining call signature.
            Returns updated_info,sampler: outputs of run().
        """
        if verbose:
            self.print_this(' GP training and MCMC run',self.logfile)

        chi2,params = self.read_data_from_file(chi2_file)
        if verbose:
            self.print_this('... read {0:d} samples'.format(chi2.size),self.logfile)
            self.print_this('... excluding first {0:d} samples'.format(exclude_from_start),self.logfile)            
        chi2 = chi2[exclude_from_start:]
        params = params[exclude_from_start:]
        Nsample = chi2.size

        # Initialising training iterations
        if train_size_start is None:
            train_size = np.min([1000,int(0.05*Nsample)])
            train_size = np.max([100,train_size]) # smaller of 1000 and 5% of sample, with minimum cap at 100
        else:
            train_size = 1*train_size_start
        bound_fac = 2.0
        Nsamp_hyper = 30 # 50
        kernel_start = None
        cv1,cv99 = -1.0,1.0
        cv16,cv84 = -1.0,1.0
        dont_use = False

        cnt = 0
        while True: 
            conv_stat = np.max(np.fabs([cv1,cv99])) if robust_convergence else np.max(np.fabs([cv16,cv84]))
            if (conv_stat <= cv_thresh):
                break
            if verbose:
                self.print_this('... training with {0:d} of {1:d} samples'.format(train_size,Nsample),self.logfile)
            if train_size >= 0.8*Nsample:
                self.print_this("No convergence for large training set. Try changing the GPR kernel or increase sampling.",
                                self.logfile)
                dont_use = True
                break
            if cnt >= max_iter:
                self.print_this("Maximum number of iterations reached. GPR may not be properly trained",self.logfile)
                break
            info = self.create_info_dict(params_best,cov_asa,chi2_file,chain_dir,
                                         dsig=dsig,prop_fac=0.25,rotate=rotate,
                                         burn_in=0,train_size=train_size,verbose=False,mcmc_prior=None,
                                         kernel=kernel,kernel_start=kernel_start,skip_train=False,
                                         bound_fac=bound_fac,Nsamp_hyper=Nsamp_hyper,rng=rng,
                                         latex_symb=latex_symb,max_samples=5,force=force)

            try:
                updated_info, sampler = run(info)
            except LoggedError:
                pass

            kernel_start = np.loadtxt(self.base_dir + chain_dir + '/hyper.tmp')
            cv_percs = np.loadtxt(self.base_dir + chain_dir + '/crossval.tmp')
            cv1 = cv_percs[0]
            cv16 = cv_percs[1]
            cv84 = cv_percs[3]
            cv99 = cv_percs[4]
            train_size_last = 1*train_size
            train_size = int(1.1*train_size_last) # 1.2
            bound_fac *= 1.1 # 1.25
            Nsamp_hyper = int(Nsamp_hyper*1.1) # 1.15
            tmp_file = self.base_dir + chain_dir + '/train_size.tmp'
            if verbose:
                self.print_this('... ... writing training size to file: ' +tmp_file,self.logfile)
            np.savetxt(tmp_file,np.array([train_size_last]),fmt='%d')
            
            cnt += 1

        if dont_use:
            if verbose:
                self.print_this('\nExiting with last available sampler.',self.logfile)
        else:
            if verbose:
                self.print_this('\nGPR is fully trained at requested accuracy with {0:d} samples!'.format(train_size_last),
                                self.logfile)
                self.print_this('... cross-validation 1,16,84,99 percentiles = {0:.2e},{1:.2e},{2:.2e},{3:.2e}'
                                .format(cv1,cv16,cv84,cv99),self.logfile)
                self.print_this('... running production quality chain with last recorded kernel params.',
                                self.logfile)

            updated_info,sampler = self.run_trained_gpr(params_best,cov_asa,chi2_file,chain_dir,
                                                        rotate=rotate,dsig=dsig,prop_fac=prop_fac,burn_in=burn_in,verbose=verbose,
                                                        kernel=kernel,latex_symb=latex_symb,rng=rng,mcmc_prior=mcmc_prior,
                                                        Rminus1_stop=Rminus1_stop,Rminus1_cl_stop=Rminus1_cl_stop,force=force)

        return updated_info, sampler
    ############################################################


    ############################################################
    def run_trained_gpr(self,params_best,cov_asa,chi2_file,chain_dir,max_restarts=10,count=None,
                        dsig=5.0,prop_fac=0.1,burn_in=500,kernel='rbf',rotate=None,rng=None,mcmc_prior=None,
                        verbose=False,latex_symb=None,Rminus1_stop=0.1,Rminus1_cl_stop=0.2,force=True):
        """ Wrapper to run Cobaya with pre-trained GPR stored in chain_dir/hyper.tmp and chosen training sample size. 
             (Assumes that kernel is compatible with hyper-parameters, so use with care.)
            -- max_restarts: int, number of times to restart in case of stuck chains
            -- count: integer for internal use.
            -- see ASAUtilities.create_info_dict() for remaining call signature.
            Returns updated_info,sampler: outputs of run().
        """
        train_size = int(np.loadtxt(self.base_dir + chain_dir + '/train_size.tmp'))
        kernel_start = np.loadtxt(self.base_dir + chain_dir + '/hyper.tmp')
        info = self.create_info_dict(params_best,cov_asa,chi2_file,chain_dir,
                                     rotate=rotate,dsig=dsig,prop_fac=prop_fac,burn_in=burn_in,verbose=verbose,kernel=kernel,
                                     kernel_start=kernel_start,skip_train=True,train_size=train_size,rng=rng,
                                     mcmc_prior=mcmc_prior,
                                     latex_symb=latex_symb,Rminus1_stop=Rminus1_stop,Rminus1_cl_stop=Rminus1_cl_stop,
                                     force=force)
        cnt = 0 if count is None else count+1
        try:
            updated_info, sampler = run(info)
        except LoggedError:
            if cnt < max_restarts:
                if verbose:
                    self.print_this('... restarting {0:d}th time'.format(cnt+1),self.logfile)
                updated_info,sampler = self.run_trained_gpr(params_best,cov_asa,chi2_file,chain_dir,
                                                            max_restarts=max_restarts,count=cnt,rotate=rotate,rng=rng,
                                                            mcmc_prior=mcmc_prior,
                                                            dsig=dsig,prop_fac=prop_fac,burn_in=burn_in,verbose=verbose,
                                                            kernel=kernel,latex_symb=latex_symb,
                                                            Rminus1_stop=Rminus1_stop,Rminus1_cl_stop=Rminus1_cl_stop,
                                                            force=force)
            else:
                if verbose:
                    self.print_this('... maximum number of restarts ({0:d}) exceeded,'.format(max_restarts)
                                    +' try decreasing prop_fac or Rminus1_stop' 
                                    + ' or re-train GPR with higher accuracy or, if all fails, with a different kernel'
                                    +'. Returning 0,0.',self.logfile)
                return 0,0

        return updated_info, sampler

    ############################################################
    
