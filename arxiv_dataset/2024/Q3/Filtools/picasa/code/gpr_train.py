import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
import sklearn.gaussian_process.kernels as Kernels
from scipy.linalg import cholesky, cho_solve, solve_triangular
import scipy.interpolate as interp
from asa_utilities import ASAUtilities
from time import time
from pathlib import Path
import os,gc

#########################################
class GPRTrainer(ASAUtilities):
    """ Class containing GPR training utilities. """

    #########################################
    def __init__(self,base_dir=None,data_file=None,dim=None,rotate=None,
                 logfile=None,tmp_dir=None,add_noise=False,
                 use_log=None,verbose=False,bound_fac=2.0,Nsamp_hyper=30,data_type='ascii',names=None,
                 absolute_residual=False):
        """ Initialise the following:
            -- base_dir: base directory path used by ASAUtilities
            -- data_file: None or valid path containing data in format 'predicted variable | parameter 1 | parameter 2 | ...'
                                If None (useful with pre-trained GPR), will assume default dummy data file for internal use. 
                                In this case, dim must be integer equal to number of parameters. 
            -- dim: None or integer number of parameters. If None, then parameter dimension will be determined
                        directly from given data. 
            -- rotate: if not None, should contain rotation matrix such that eigen_param = (rotate.T) . params
            -- logfile: valid writeable file for storing verbose output. 
            -- tmp_dir: user-defined path to directory where training output will be temporarily written.
            -- add_noise: boolean. If True, kernel will be defined with additive noise, 
                                                   thus increasing number of hyperparameters by one.
            -- use_log: None or boolean, whether or not to log transform input data. 
                               If None (default), decision will be based on actual dynamic range of data.
                               If data_file is None, then use_log *must* be a boolean value.
            -- verbose: boolean, whether or not to display messages.
            -- bound_fac: float > 1.0, controls starting range of hyperparameters in GPR training
            -- Nsamp_hyper: int > 0, controls ASA sampling of hyperparameters in GPR training
             -- data_type: string describing type of data file. One of ['ascii','npz']. Ignored if data_file is None.
             -- names: None or list of strings giving column names. Needed if data_type == 'npz'.
             -- absolute_residual: boolean, whether to use absolute or relative differences in cross-validation residuals.
                                                (Default False, use relative residuals.)
        """
        self.base_dir = base_dir if base_dir is not None else os.getcwd()+'/'
        self.rotate = rotate
        self.logfile = logfile
        self.tmp_dir = tmp_dir
        self.verbose = verbose
        self.bound_fac = bound_fac
        self.Nsamp_hyper = Nsamp_hyper
        self.names = names
        self.add_noise = add_noise
        self.absolute_residual = absolute_residual

        if data_file is not None:
            self.data_file = data_file
            self.data_type = data_type
            self.GPR_Exists = False
        else:
            if use_log is None:
                raise ValueError('use_log must be True or False if data_file is not specified.')
            self.data_file = self.tmp_dir + '/train_set.tmp'
            self.data_type = 'ascii'
            self.GPR_Exists = True

        self.kernel_names = ['rbf','rq','matern']

        if not os.path.isfile(self.data_file):
            raise ValueError('Expecting data_file '+self.data_file+' to be valid path to data file in GPRTrainer(). ')

        ASAUtilities.__init__(self,base_dir=self.base_dir,logfile=self.logfile)

        if self.verbose:
            self.print_this('Setting up GPRTrainer()...',self.logfile)

        if self.tmp_dir is None:
            self.tmp_dir = self.base_dir # expect user-defined tmp_dir where training output will be temporarily written
        Path(self.tmp_dir).mkdir(parents=True, exist_ok=True)

        if self.verbose:
            self.print_this('... preparing data',self.logfile)
        if (self.data_type == 'npz') & (self.names is None):
            raise ValueError("kwarg 'names' should be list of column names for data_type 'npz'"
                             +" in GPRTrainer.prep_data()")
        pred_var,params,dim,use_log_data = self.prep_data()
        self.pred_var = pred_var
        self.params = params
        self.dim = dim

        if self.GPR_Exists:
            self.use_log = use_log
        else:
            self.use_log = use_log_data if use_log is None else use_log

        if self.verbose:
            self.print_this('... done with setup',self.logfile)
    #########################################

    #########################################
    def prep_data(self):
        """ Convenience function to prepare data for GPR training."""
        if self.data_type == 'ascii':
            data_prods = np.loadtxt(self.data_file).T
        elif self.data_type == 'npz':
            all_arr = np.load(self.data_file)
            col1 = all_arr[self.names[0]]
            data_prods = np.zeros((len(self.names),col1.size),dtype=float)
            for n in range(len(self.names)):
                data_prods[n] = all_arr[self.names[n]]
            del all_arr,col1
            gc.collect()
        pred_var = data_prods[0]
        params = data_prods[1:].T
        if self.rotate is not None:
            params = np.dot(self.rotate.T,params.T).T # params now in eigenspace
        dim = params.shape[1]
        use_log = False
        if np.all(pred_var > 0.0):
            if pred_var.max()/pred_var.min() > 1e6: # hard-coded threshold to switch to log[pred_var]
                use_log = True
                if self.verbose:
                    self.print_this('Predicted variable will be log-transformed before interpolation.',self.logfile)
        del data_prods
        gc.collect()
        return pred_var,params,dim,use_log
    #########################################

    #########################################
    def train_gpr_once(self,rng=None,kernel='rbf',kernel_start=None,skip_train=False,train_size=None,
                       bound_fac=None,Nsamp_hyper=None,verbose=True):
        """ Main workhorse routine to train GPR using given data file.
            -- rng: None or instance of np.random.RandomState()
            -- kernel: name of GPR kernel to use. Currently supports ['rbf','rq','matern'].
            -- kernel_start: None or valid array of hyperparameters for kernel.
            -- skip_train: if True, GPR training is skipped and interpolator constructed using provided 
                                  parameters in kernel_start (in this case, kernel_start cannot be None).
            -- train_size: None or integer number of samples to be used for training GPR.
            -- bound_fac: None or float > 1.0, controls starting range of hyperparameters in GPR training
            -- Nsamp_hyper: None or int > 0, controls ASA sampling of hyperparameters in GPR training
            -- verbose: boolean. Print selected messages if True (separate from self.verbose).
            Returns interpolating function.
        """
        if kernel not in self.kernel_names:
            raise NameError('Only kernels ['+','.join(['%s' % (kern,) for kern in self.kernel_names])
                            +'] supported in GPRTrainer().')

        if rng is None:
            rng_use = np.random.RandomState(seed=1983)
            
        if bound_fac is None:
            bound_fac = self.bound_fac

        if Nsamp_hyper is None:
            Nsamp_hyper = self.Nsamp_hyper

        if skip_train:
            if kernel_start is None:
                raise ValueError('kernel_start must be valid list of hyperparameters if skip_train = True '
                                 +'in GPRTrainer.train_gpr_once()')
        # Gaussian Process Regression
        if verbose:
            self.print_this('... setting up Gaussian Process Regression',self.logfile)
        if kernel == 'rq':
            if kernel_start is None:
                kernel_start_use = np.ones(4,dtype=float) if self.add_noise else np.ones(3,dtype=float)
            else:
                kernel_start_use = np.asarray(kernel_start)
            if verbose:
                self.print_this('... ... using Rational Quadratic kernel',self.logfile)
                print_str = ('... ... starting with (norm,alpha,scale,noise) = (' 
                             if self.add_noise else 
                             '... ... starting with (norm,alpha,scale) = (')
                print_str += ','.join(['%.3f' % (par,) for par in kernel_start_use]) +')'
                self.print_this(print_str,self.logfile)
            Kernel = kernel_start_use[0] * Kernels.RationalQuadratic(alpha=kernel_start_use[1],
                                                                      length_scale=kernel_start_use[2])
        else:
            if kernel_start is None:
                kernel_start_use = np.ones(self.dim+2,dtype=float) if self.add_noise else np.ones(self.dim+1,dtype=float)
            else:
                kernel_start_use = np.asarray(kernel_start)
            if kernel == 'rbf':
                if verbose:
                    self.print_this('... ... using anisotropic Gaussian (RBF) kernel',self.logfile)
                    print_str = ('... ... starting with (norm,scale,noise) = (' 
                                 if self.add_noise else 
                                 '... ... starting with (norm,scale) = (')
                    print_str += ','.join(['%.3e' % (par,) for par in kernel_start_use]) +')'
                    self.print_this(print_str,self.logfile)
                Kernel = kernel_start_use[0] * Kernels.RBF(length_scale=kernel_start_use[1:self.dim+1])
            elif kernel == 'matern':
                if verbose:
                    self.print_this('... ... using anisotropic Matern kernel with nu = 2.5',self.logfile)
                    print_str = ('... ... starting with (norm,scale,noise) = (' 
                                 if self.add_noise else 
                                 '... ... starting with (norm,scale) = (')
                    print_str += ','.join(['%.3e' % (par,) for par in kernel_start_use]) +')'
                    self.print_this(print_str,self.logfile)
                Kernel = kernel_start_use[0] * Kernels.Matern(length_scale=kernel_start_use[1:self.dim+1],nu=2.5)

        if self.add_noise:
            Kernel = Kernel + Kernels.WhiteKernel(noise_level=kernel_start_use[-1])

        gpr = GPR(kernel=Kernel,normalize_y=True,alpha=1e-8)
        gpr.kernel_ = gpr.kernel
        if self.GPR_Exists:
            gpr.X_train_ = self.params
            gpr.y_train_ = np.log(self.pred_var) if self.use_log else self.pred_var
        else:
            TRAIN_SIZE = int(2000*(self.dim/3)**1.5) if (train_size is None) else train_size
            ind_train = rng_use.choice(self.pred_var.size,size=np.min([TRAIN_SIZE,self.pred_var.size]),replace=False)
            ind_cv = np.delete(np.arange(self.pred_var.size),ind_train)
            gpr.X_train_ = self.params[ind_train]
            gpr.y_train_ = np.log(self.pred_var[ind_train]) if self.use_log else self.pred_var[ind_train]

        gpr._y_train_std = gpr.y_train_.std() # 1.0
        gpr._y_train_mean = gpr.y_train_.mean() # 0.0

        self.y_train_mean = 1.0*gpr._y_train_mean
        self.y_train_std = 1.0*gpr._y_train_std

        if gpr.normalize_y:
            gpr.y_train_ -= gpr._y_train_mean
            gpr.y_train_ /= gpr._y_train_std

        if skip_train:
            if verbose:
                data_size = self.pred_var.size if self.GPR_Exists else ind_train.size
                self.print_this('... ... using fixed input hyperparameters with {0:d} training samples'.format(data_size))
            gpr.kernel_.theta = np.log(kernel_start_use)
        else:
            if verbose:
                self.print_this('... ... fitting GP on training set ({0:d} values; ~{1:.0f}% of input data)'
                                .format(ind_train.size,100*ind_train.size/self.pred_var.size),self.logfile)
            start_time = time()
            b_mins = np.asarray(kernel_start_use)/bound_fac
            b_maxs = np.asarray(kernel_start_use)*bound_fac
            bounds = np.log(np.array([b_mins,b_maxs]))
            hyperpred_var,ln_hyperparams = self.asa_chi2min(np.array([1.0]),np.array([1.0]),
                                                        bounds[0],bounds[1],self.hyper_model_func,gpr,self.hyper_cost_func,None,
                                                        Nsamp=Nsamp_hyper,
                                                        eps_close=0.2,eps_conv=0.5,robust_convergence=False,
                                                        rng=rng_use,verbose=self.verbose)
            if verbose:
                self.time_this(start_time)
                self.print_this('... ... setting up Cholesky decomposition',self.logfile)
            gpr.kernel_.theta = ln_hyperparams[hyperpred_var.argmin()] # ln_hyperparams_best

        # # Copied from sklearn/gaussian_process/_gpr.py
        # Precompute quantities required for predictions which are independent
        # of actual query points
        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        GPR_CHOLESKY_LOWER = True
        K = gpr.kernel_(gpr.X_train_)
        K[np.diag_indices_from(K)] += gpr.alpha
        try:
            gpr.L_ = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError as exc:
            exc.args = (
            f"The kernel, {gpr.kernel_}, is not returning a positive "
            "definite matrix. Try gradually increasing the 'alpha' "
            "parameter of your GaussianProcessRegressor estimator.",
                        ) + exc.args
            raise
        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        gpr.alpha_ = cho_solve((gpr.L_, GPR_CHOLESKY_LOWER),gpr.y_train_,check_finite=False)

        gpr.log_marginal_likelihood_value_ = gpr.log_marginal_likelihood(gpr.kernel_.theta)

        if verbose:
            self.print_this('... ... log-marg-like = {0:.2f}'.format(gpr.log_marginal_likelihood_value_),self.logfile)
            self.print_this('... ... best kernel params = ('
                            +','.join(['%.3f' % (par,) for par in gpr.kernel_.theta])+')',self.logfile)
            self.print_this(gpr.kernel_,self.logfile)

        if not self.GPR_Exists:
            tmp_file = self.tmp_dir + '/hyper.tmp'
            if verbose:
                self.print_this('... ... writing best kernel params to file: ' +tmp_file,self.logfile)
            np.savetxt(tmp_file,np.exp(gpr.kernel_.theta)) # theta is ln-hyperparam

            if ind_cv.size:
                if verbose:
                    self.print_this('... ... cross-validating',self.logfile)
                pred_var_CV = gpr.predict(self.params[ind_cv], return_std=False)
                if self.use_log:
                    pred_var_CV = np.exp(pred_var_CV)
                if self.absolute_residual:
                    resid_CV = pred_var_CV - self.pred_var[ind_cv] 
                else:
                    resid_CV = pred_var_CV / self.pred_var[ind_cv] - 1
                cv1 = np.percentile(resid_CV,1)
                cv16 = np.percentile(resid_CV,16)
                cv50 = np.percentile(resid_CV,50)
                cv84 = np.percentile(resid_CV,84)
                cv99 = np.percentile(resid_CV,99)
                if verbose:
                    rel_strng = 'absolute' if self.absolute_residual else 'relative'
                    self.print_this('... ... (1,16,50,84,99) percentiles of '+rel_strng+' error: ({0:.1e},{1:.1e},{2:.1e},{3:.1e},{4:.1e})'
                                    .format(cv1,cv16,cv50,cv84,cv99),self.logfile)
                tmp_file = self.tmp_dir + '/crossval.tmp'
                if verbose:
                    self.print_this('... ... writing cross-validation percentiles to file: ' +tmp_file)
                np.savetxt(tmp_file,np.array([cv1,cv16,cv50,cv84,cv99]))
            else:
                if verbose:
                    self.print_this('... ... no inputs left to cross-validate',self.logfile)

            tmp_file = self.tmp_dir + '/train_set.tmp'
            if verbose:
                self.print_this('... ... writing training set sample to file: ' +tmp_file,self.logfile)
            if self.rotate is not None:
                self.write_data_to_file(tmp_file,self.pred_var[ind_train],np.dot(self.rotate,self.params[ind_train].T).T)
            else:
                self.write_data_to_file(tmp_file,self.pred_var[ind_train],self.params[ind_train])

        if verbose:
            self.print_this('... GPR is ready for use',self.logfile)

        return gpr.predict
    #########################################

    #########################################
    def train_gpr(self,rng=None,kernel='rbf',kernel_start=None,skip_train=False,train_size_start=None,
                  max_iter=25,max_iter_vary=5,cv_thresh=0.01,
                  verbose=True,robust_convergence=True,vary_kernel=True):
        """ Iteratively train GPR using given data file.
            -- rng: None or instance of np.random.RandomState()
            -- kernel: name of GPR kernel to use. Needed if kwarg vary_kernel = False. 
                            Currently supports ['rbf','rq','matern'].
            -- kernel_start: None or valid array of hyperparameters for kernel.
            -- skip_train: if True, GPR training is skipped and interpolator constructed using provided 
                                  parameters in kernel_start (in this case, kernel_start cannot be None).
            -- train_size: None or integer number of samples to be used for training GPR.
            -- max_iter: maximum number of training iterations.
            -- max_iter_vary: maximum number of exploratory training iterations when varying kernel.
                                         Only relevant if vary_kernel = True.
            -- verbose: boolean. Print messages if True (separate from self.verbose).
            -- robust_convergence: controls whether to use 1,99 (if True) or 16,84 percentiles (if False) of cross-validation
                                                    residuals in judging convergence.
            -- cv_thresh: threshold on max of abs value of cross-validation residual 1,99 or 16,84 percentiles.
            Returns interpolating function.
        """
        if skip_train:
            train_size = int(np.loadtxt(self.tmp_dir + '/train_size.tmp'))
            kernel_start_use = np.loadtxt(self.tmp_dir + '/hyper.tmp')
            interpolator = self.train_gpr_once(rng=rng,kernel=kernel,kernel_start=kernel_start_use,
                                               skip_train=True,train_size=train_size,verbose=verbose)
        else:
            if vary_kernel:
                self.print_this('\nVarying kernels to find best option',self.logfile)
                cv1 = -1.0*np.ones(len(self.kernel_names),dtype=float) 
                cv99 = np.ones(len(self.kernel_names),dtype=float)
                cv16 = -1.0*np.ones(len(self.kernel_names),dtype=float)
                cv84 = np.ones(len(self.kernel_names),dtype=float)
                for k in range(len(self.kernel_names)):
                    kernel = self.kernel_names[k]
                    interpolator = self.train_gpr(rng=rng,kernel=kernel,kernel_start=kernel_start,skip_train=False,
                                                  train_size_start=train_size_start,max_iter=max_iter_vary,cv_thresh=cv_thresh,
                                                  verbose=False,vary_kernel=False)
                    cv_percs = np.loadtxt(self.tmp_dir + '/crossval.tmp')
                    cv1[k] = cv_percs[0]
                    cv16[k] = cv_percs[1]
                    cv84[k] = cv_percs[3]
                    cv99[k] = cv_percs[4]
                cv_outer = np.maximum(np.fabs(cv1),np.fabs(cv16))
                cv_inner = np.maximum(np.fabs(cv16),np.fabs(cv84))
                k_choose = np.argmin(cv_outer)
                kernel = self.kernel_names[k_choose]
                self.print_this('\nKernel '+kernel+' is best choice: cv(1pc,99pc) after {0:d} iters = ({1:.2e},{2:.2e})\n'
                                .format(max_iter_vary,cv1[k_choose],cv99[k_choose]),self.logfile)
                interpolator = self.train_gpr(rng=rng,kernel=kernel,kernel_start=kernel_start,skip_train=False,
                                              train_size_start=train_size_start,max_iter=max_iter,verbose=verbose,
                                              cv_thresh=cv_thresh,vary_kernel=False)
            else:
                if kernel not in self.kernel_names:
                    raise NameError('Only kernels ['+','.join(['%s' % (kern,) for kern in self.kernel_names])
                                    +'] supported in GPRTrainer().')

                bound_fac = 1.0*self.bound_fac
                Nsamp_hyper = 1*self.Nsamp_hyper
                if train_size_start is None:
                    train_size = np.min([1000,int(0.05*self.pred_var.size)])
                    train_size = np.max([100,train_size]) # smaller of 1000 and 5% of sample, with minimum cap at 100
                else:
                    train_size = 1*train_size_start
                cv1,cv99 = -1.0,1.0
                cv16,cv84 = -1.0,1.0
                dont_use = False
                cnt = 0
                kernel_start_use = kernel_start
                while True: 
                    conv_stat = np.max(np.fabs([cv1,cv99])) if robust_convergence else np.max(np.fabs([cv16,cv84]))
                    if (conv_stat <= cv_thresh):
                        break
                    if verbose:
                        self.print_this('\n... training with {0:d} of {1:d} samples'.format(train_size,self.pred_var.size),self.logfile)
                        self.print_this('... ... current conv_stat = {0:.2e}; waiting for {1:.2e}'
                                        .format(conv_stat,cv_thresh),self.logfile)
                    if train_size >= 0.8*self.pred_var.size:
                        self.print_this("No convergence for large training set."
                                        +" Try changing the GPR kernel or increase sampling.",self.logfile)
                        dont_use = True
                        break
                    if cnt >= max_iter:
                        self.print_this("Maximum number of iterations reached. GPR may not be properly trained.",self.logfile)
                        break
                    interpolator = self.train_gpr_once(rng=rng,kernel=kernel,kernel_start=kernel_start_use,
                                                       skip_train=False,train_size=train_size,verbose=verbose,
                                                       bound_fac=bound_fac,Nsamp_hyper=Nsamp_hyper)

                    kernel_start_use = np.loadtxt(self.tmp_dir + '/hyper.tmp')
                    cv_percs = np.loadtxt(self.tmp_dir + '/crossval.tmp')
                    cv1 = cv_percs[0]
                    cv16 = cv_percs[1]
                    cv84 = cv_percs[3]
                    cv99 = cv_percs[4]
                    train_size_last = 1*train_size
                    train_size = int(1.1*train_size_last) # 1.2
                    bound_fac *= 1.1 # 1.25
                    Nsamp_hyper = int(Nsamp_hyper*1.1) # 1.15
                    tmp_file = self.tmp_dir + '/train_size.tmp'
                    if verbose:
                        self.print_this('... writing training size to file: ' +tmp_file+'\n',self.logfile)
                    np.savetxt(tmp_file,np.array([train_size_last]),fmt='%d')

                    cnt += 1

                if dont_use:
                    if verbose:
                        self.print_this('\nExiting with last available interpolator.',self.logfile)
                else:
                    if verbose:
                        self.print_this('\nGPR is fully trained at requested accuracy with {0:d} samples!'.format(train_size_last),
                                        self.logfile)
                        self.print_this('... cross-validation 1,16,84,99 percentiles = {0:.2e},{1:.2e},{2:.2e},{3:.2e}'
                                        .format(cv1,cv16,cv84,cv99),self.logfile)

        return interpolator
    #########################################

    #########################################
    def hyper_model_func(self,gpr,ln_hyperparams):
        """ Convenience function used by self.train_gpr_once(). """
        return -2.0*gpr.log_marginal_likelihood(ln_hyperparams)

    def hyper_cost_func(self,data,err_data,model,cost_args):
        """ Convenience function used by self.train_gpr_once(). """
        return model
    #########################################

    #########################################
    def predict(self,params,interpolator):
        """ Use GPR to predict a value at given parameters.
            -- params: array of shape (self.dim,) or (Nparam,self.dim) containing sample in original parameter space.
            -- interpolator: valid GPR interpolator obtained from either self.train_gpr() or self.train_gpr_once().
            Returns scalar or 1-d array of size Nparams.
        """
        if len(params.shape) == 2:
            if params.shape[1] != self.dim:
                raise TypeError('Expecting params.shape[1] to be {0:d} in GPRTrainer.predict()'.format(self.dim))
            params_use = params.copy()
        elif len(params.shape) == 1:
            if params.size != self.dim:
                raise TypeError('Expecting params.size to be {0:d} in GPRTrainer.predict()'.format(self.dim))
            params_use = np.array([params])
        else:
            raise TypeError('Expecting params to be array of shape ({0:d},) or (Nsample,{0:d},) in GPRTrainer.predict()'
                            .format(self.dim))

        if self.rotate is not None:
            params_use = np.dot(self.rotate.T,params_use.T).T # move to eigenspace
        predicted = interpolator(params_use)
        if self.use_log:
            predicted = np.exp(predicted)
        return predicted

    #########################################

