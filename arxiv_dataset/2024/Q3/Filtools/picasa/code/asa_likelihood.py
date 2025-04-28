import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
import sklearn.gaussian_process.kernels as Kernels
from scipy.linalg import cholesky, cho_solve, solve_triangular
import scipy.interpolate as interp
from asa_utilities import ASAUtilities
from cobaya.likelihood import Likelihood
from time import time
from pathlib import Path

#########################################
class ASALike(Likelihood,ASAUtilities):
    data_file = None
    base_dir = None
    tmp_dir = None
    logfile = None
    rotate = None
    estimate = 'gpr' # gpr, rbf
    kernel = 'rbf' # if estimate==gpr, use rq, rbf or matern. if estimate==rbf, use gaussian or cubic
    kernel_start = None # None or list of floats
    skip_train = False # if True, kernel_start must be valid list of hyperparameters
    bound_fac = 10.0 # parameter deciding starting hyperparameter range for GPR training
    Nsamp_hyper = 50 # ASA Nsamp for GPR hyperparameter exploration
    verbose = False
    rng = None
    train_size = None # None or integer number of samples to be used for training GPR.
    #########################################
    def initialize(self):
        ASAUtilities.__init__(self,base_dir=self.base_dir,logfile=self.logfile)
        if self.tmp_dir is None:
            self.tmp_dir = self.base_dir # expect user-defined tmp_dir where training output will be temporarily written
        Path(self.tmp_dir).mkdir(parents=True, exist_ok=True)
        self.data_prods = np.loadtxt(self.data_file).T
        chi2 = self.data_prods[0]
        params = self.data_prods[1:].T
        if self.rotate is not None:
            params = np.dot(self.rotate.T,params.T).T # params now in eigenspace
        dim = params.shape[1]
        self.use_log = False
        if np.all(chi2 >= 0.0):
            if chi2.max()/chi2.min() > 1e6: # hard-coded threshold to switch to log[chi2]
                self.use_log = True
                if self.verbose:
                    self.print_this('Cost function will be log-transformed before interpolation.',self.logfile)
        if self.rng is None:
            self.rng = np.random.RandomState(seed=1983)


        if self.estimate == 'gpr':
            if self.skip_train:
                if self.kernel_start is None:
                    raise ValueError('kernel_start must be valid list of hyperparameters if skip_train = True.')
            # Gaussian Process Regression
            if self.verbose:
                self.print_this('... setting up Gaussian Process Regression',self.logfile)
            if self.kernel == 'rq':
                if self.kernel_start is None:
                    self.kernel_start = np.ones(3,dtype=float)
                else:
                    self.kernel_start = np.asarray(self.kernel_start)
                if self.verbose:
                    self.print_this('... ... using Rational Quadratic kernel',self.logfile)
                    self.print_this('... ... starting with (norm,alpha,scale) = ({0:.3f},{1:.3f},{2:.3f})'
                                    .format(self.kernel_start[0],self.kernel_start[1],self.kernel_start[2]),self.logfile)
                kernel = self.kernel_start[0] * Kernels.RationalQuadratic(alpha=self.kernel_start[1],
                                                                          length_scale=self.kernel_start[2])
            else:
                if self.kernel_start is None:
                    self.kernel_start = np.ones(dim+1,dtype=float)
                else:
                    self.kernel_start = np.asarray(self.kernel_start)
                if self.kernel == 'rbf':
                    if self.verbose:
                        self.print_this('... ... using anisotropic Gaussian (RBF) kernel',self.logfile)
                        self.print_this('... ... starting with (norm,scale) = ('
                                        +','.join(['%.3e' % (par,) for par in self.kernel_start])
                                        +')',self.logfile)
                    kernel = self.kernel_start[0] * Kernels.RBF(length_scale=self.kernel_start[1:])
                elif self.kernel == 'matern':
                    if self.verbose:
                        self.print_this('... ... using anisotropic Matern kernel with nu = 2.5',self.logfile)
                    kernel = self.kernel_start[0] * Kernels.Matern(length_scale=self.kernel_start[1:],nu=2.5)
                else:
                    raise NameError('Only kernels rq, rbf, matern supported for estimate == gpr.')

            b_mins = np.asarray(self.kernel_start)/self.bound_fac
            b_maxs = np.asarray(self.kernel_start)*self.bound_fac
            bounds = np.log(np.array([b_mins,b_maxs]))

            TRAIN_SIZE = int(2000*(dim/3)**1.5) if (self.train_size is None) else self.train_size
            ind_train = self.rng.choice(chi2.size,size=np.min([TRAIN_SIZE,chi2.size]),replace=False)
            ind_cv = np.delete(np.arange(chi2.size),ind_train)

            gpr = GPR(kernel=kernel,normalize_y=True,alpha=1e-8)
            # # ASA works better than default minimizer for high-dim and/or nonlinear problems
            # if self.use_log:
            #     gpr.fit(params[ind_train],np.log(chi2[ind_train])) 
            # else:
            #     gpr.fit(params[ind_train],chi2[ind_train])
            gpr.X_train_ = params[ind_train]
            gpr.y_train_ = np.log(chi2[ind_train]) if self.use_log else chi2[ind_train]
            gpr.kernel_ = gpr.kernel
            gpr._y_train_std = gpr.y_train_.std() # 1.0
            gpr._y_train_mean = gpr.y_train_.mean() # 0.0

            self.y_train_mean = 1.0*gpr._y_train_mean
            self.y_train_std = 1.0*gpr._y_train_std

            if gpr.normalize_y:
                gpr.y_train_ -= gpr._y_train_mean
                gpr.y_train_ /= gpr._y_train_std
            
            if self.skip_train:
                if self.verbose:
                    self.print_this('... ... using fixed input hyperparameters with {0:d} training samples'.format(ind_train.size))
                gpr.kernel_.theta = np.log(self.kernel_start)
            else:
                if self.verbose:
                    self.print_this('... ... fitting GP on training set ({0:d} values; ~{1:.0f}% of input data)'
                                    .format(ind_train.size,100*ind_train.size/chi2.size),self.logfile)
                start_time = time()
                hyperchi2,ln_hyperparams = self.asa_chi2min(np.array([1.0]),np.array([1.0]),
                                                            bounds[0],bounds[1],self.hyper_model_func,gpr,self.hyper_cost_func,None,
                                                            Nsamp=self.Nsamp_hyper,
                                                            eps_close=0.2,eps_conv=0.5,robust_convergence=False,
                                                            rng=self.rng,verbose=self.verbose)
                if self.verbose:
                    self.time_this(start_time)
                    self.print_this('... ... setting up Cholesky decomposition',self.logfile)
                gpr.kernel_.theta = ln_hyperparams[hyperchi2.argmin()] # ln_hyperparams_best

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

            if self.verbose:
                self.print_this('... ... log-marg-like = {0:.2f}'.format(gpr.log_marginal_likelihood_value_),self.logfile)
                self.print_this('... ... best kernel params = ('
                                +','.join(['%.3f' % (par,) for par in gpr.kernel_.theta])+')',self.logfile)
                self.print_this(gpr.kernel_,self.logfile)

            tmp_file = self.tmp_dir + '/hyper.tmp'
            if self.verbose:
                self.print_this('... ... writing best kernel params to file: ' +tmp_file,self.logfile)
            np.savetxt(tmp_file,np.exp(gpr.kernel_.theta)) # theta is ln-hyperparam

            if ind_cv.size:
                if self.verbose:
                    self.print_this('... ... cross-validating',self.logfile)
                chi2_CV = gpr.predict(params[ind_cv], return_std=False)
                if self.use_log:
                    chi2_CV = np.exp(chi2_CV)
                resid_CV = chi2_CV / chi2[ind_cv] - 1
                cv1 = np.percentile(resid_CV,1)
                cv16 = np.percentile(resid_CV,16)
                cv50 = np.percentile(resid_CV,50)
                cv84 = np.percentile(resid_CV,84)
                cv99 = np.percentile(resid_CV,99)
                if self.verbose:
                    self.print_this('... ... (1,16,50,84,99) percentiles of relative error: ({0:.1e},{1:.1e},{2:.1e},{3:.1e},{4:.1e})'
                                    .format(cv1,cv16,cv50,cv84,cv99),self.logfile)
                tmp_file = self.tmp_dir + '/crossval.tmp'
                if self.verbose:
                    self.print_this('... ... writing cross-validation percentiles to file: ' +tmp_file)
                np.savetxt(tmp_file,np.array([cv1,cv16,cv50,cv84,cv99]))
            else:
                if self.verbose:
                    self.print_this('... ... no inputs left to cross-validate',self.logfile)

            tmp_file = self.tmp_dir + '/train_set.tmp'
            if self.verbose:
                self.print_this('... ... writing training set sample to file: ' +tmp_file,self.logfile)
            if self.rotate is not None:
                self.write_data_to_file(tmp_file,chi2[ind_train],np.dot(self.rotate,params[ind_train].T).T)
            else:
                self.write_data_to_file(tmp_file,chi2[ind_train],params[ind_train])

            if self.verbose:
                self.print_this('... GPR is ready for use',self.logfile)
            self.interpolator = gpr.predict
        elif self.estimate == 'rbf':
            if self.verbose:
                self.print_this('... setting up Radial Basis Function interpolator',self.logfile)
            if self.kernel == 'gaussian':
                if self.use_log:
                    self.interpolator = interp.RBFInterpolator(params,np.log(chi2),
                                                               kernel='gaussian',epsilon=0.1*cond_sig_asa.min())
                else:
                    self.interpolator = interp.RBFInterpolator(params,chi2,
                                                               kernel='gaussian',epsilon=0.1*cond_sig_asa.min())
            elif self.kernel == 'cubic':
                self.interpolator = interp.RBFInterpolator(params,np.log(chi2),kernel='cubic')
                self.use_log = True
            if self.verbose:
                self.print_this('... RBF is set up',self.logfile)
        else:
            raise ValueError("Only ['gpr','rbf'] estimate(s) supported for now.")
    #########################################

    #########################################
    def logp(self,**params):
        keys = list(params.keys())
        keys.remove('_derived') # hack for now
        param_vec = np.array([params[keys[i]] for i in range(len(keys))])
        if self.rotate is not None:
            param_vec = np.dot(self.rotate.T,param_vec) # move to eigenspace
        chi2 = self.interpolator([param_vec])
        if chi2 < 0.0:
            return -np.inf
        if self.use_log:
            chi2 = np.exp(chi2)
        return -0.5*chi2
    #########################################

    #########################################
    def get_allow_agnostic(self):
        return True
    #########################################

    #########################################
    # For use with GPR
    #########################################
    def hyper_model_func(self,gpr,ln_hyperparams):
        return -2.0*gpr.log_marginal_likelihood(ln_hyperparams)

    def hyper_cost_func(self,data,err_data,model,cost_args):
        return model
    #########################################


#########################################

