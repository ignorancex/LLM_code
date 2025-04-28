import numpy as np

class MyStuff(object):
    """ This class contains custom model and cost function definitions.
         These could be defined directly as functions, or could be in separate classes, imported from elsewhere.
    """

    def __init__(self,rng=None):
        # Random state
        self.rng = rng if rng is not None else np.random.RandomState(seed=42)
        # Data structure (observables and control variables, here y = f(x)). Could also be loaded from file.
        # ... x values
        self.xvals = np.linspace(-1,1,10)
        # ... inverse covariance matrix
        err = 0.2
        self.invcov_data = (1/err**2)*np.eye(self.xvals.size)
        # ... y values: effectively, f(x) := x
        self.data = self.xvals + err*self.rng.randn(self.xvals.size)


    def my_model(self,x,params):
        """ Polynomial. Expect x to be scalar or array-like, params to be list or 1-d array. 
             Essentially same as ASAUtilities.model_poly().
        """
        return np.sum(np.array([params[i]*x**i for i in range(len(params))]),axis=0)

    def my_cost(self,data,invcov_data,model,cost_args):
        residual = data - model
        chi2 = np.dot(residual,np.dot(invcov_data,residual))
        return chi2


if __name__=="__main__":
    import os
    from picasa import PICASA
    from time import time
    import gc
    # ... following imports needed for plotting and handling Cobaya output
    import matplotlib.pyplot as plt
    from getdist.mcsamples import loadMCSamples
    import getdist.plots as gdplt
    
    rng = np.random.RandomState(seed=42)
    pic = PICASA(base_dir='../') # base_dir should be the 'share/' folder on local system.
    # pic.NPROC = 1 # uncomment this to set no. of processes by hand
    stuff = MyStuff(rng=rng)
    
    # let's fit a quadratic polynomial to the data
    params_true = [0.,1.,0.] # true values, for comparison
    degree = 2
    dim = degree + 1

    # analytical answer
    err_data = 1/np.sqrt(np.diag(stuff.invcov_data))
    apoly,cov_poly = pic.polyfit_custom(stuff.xvals,stuff.data,degree,sig2=err_data)
    sig_apoly = np.sqrt(np.diag(cov_poly))
    print('Best poly fit...')
    print("... values ( a0,..,a{0:d} ) = ( ".format(dim-1)
          +','.join(['%.3f' % (ai,) for ai in apoly])+" )")
    print("... errors ( sig_a0,..,sig_a{0:d} ) = ( ".format(dim-1)
          +','.join(['%.3f' % (sigi,) for sigi in sig_apoly])+" )")
    print("... full cov = \n",cov_poly)

    chi2_analytical = np.sum(((stuff.data - stuff.my_model(stuff.xvals,apoly))/err_data)**2)
    print('... chi2 = {0:.2f}'.format(chi2_analytical))


    ########################
    # PICASA
    # User must define a dictionary (called `data_pack' below) containing some mandatory and optional keys.
    # All keys are listed below, with commented keys showing their default values.
    # For detailed information on the definition and usage of the keys, see the docstring of PICASA.mcmc().

    # empty dictionary, to be modified below.
    data_pack = {}

    ###########################
    # ... keys needed for ASA (and MCMC)
    ###########################
    # ... mandatory
    data_pack['chi2_file'] = 'examples/demo_chi2data.txt'
    data_pack['Nsamp'] = 10 # number of samples at each zoom iteration
    data_pack['param_mins'] = [-4.0]*dim # starting range for 
    data_pack['param_maxs'] = [4.0]*dim # parameter exploration
    data_pack['dsig'] = 5.0 # maximum factor multiplying 1-sigma along each parameter direction for last iterations
    data_pack['data'] = stuff.data
    data_pack['invcov_data'] = stuff.invcov_data
    data_pack['cost_func'] = stuff.my_cost
    data_pack['cost_args'] = None
    data_pack['model_func'] = stuff.my_model
    data_pack['model_args'] = stuff.xvals
    ###########################
    # ... optional
    data_pack['rng'] = rng # Default None
    # data_pack['parallel'] = True # Default False 

    # data_pack['no_cost'] = False # If True, cost_func not used; user should ensure that model_func returns chi2 value.
    # data_pack['Ndata'] = 0 
    # If no_cost is True, user must provide integer length of data vector here. (Or a constant such that dof = Ndata - dim
    # is close to expectation value of chi2_min.)
    data_pack['last_frac'] = 0.25 # Default 0.5
    # data_pack['C_sig'] = 1.0 # Default 1.0
    # data_pack['max_opt_iter'] = 10 # number of times ASA optimization is restarted
    # data_pack['verbose'] = True
    # data_pack['use_coarse'] = True
    # data_pack['eps_conv'] = 0.1
    # data_pack['read_file'] = None

    ###########################
    # ... keys needed for MCMC
    ###########################
    # ... mandatory
    data_pack['chain_dir'] = 'examples/chains/demo'
    ###########################
    # ... optional
    data_pack['mcmc_prior'] = [[-0.3,0.5],[0.5,1.5],[-0.7,0.7]] # Default None
    # data_pack['Rminus1_stop'] = 0.005
    # data_pack['Rminus1_cl_stop'] = 0.2
    # data_pack['kernel'] = 'rbf' 
    # data_pack['robust_convergence'] = True
    # data_pack['cv_thresh'] = 1e-2 
    # data_pack['no_training'] = False
    # data_pack['read_file_mcmc'] =  None # overrides chi2_file
    # data_pack['prop_fac'] = 0.1
    # data_pack['latex'] = None
    # data_pack['burn_in'] = 1000
    data_pack['verbose_training'] = True # Default False
    ###########################


    start_time = time()
    info,sampler,chi2,params,cov_asa,params_best,chi2_min = pic.mcmc(data_pack)
    pic.time_this(start_time)


    # Coming soon...
    # ... demo for using ASA starting last iterations with pre-existing chi2_file.


    gd_sample = loadMCSamples(os.path.abspath(info["output"]))
    gd_sample.label = 'PICASA'
    # samples contain a0..a(dim-1) chi2 | chi2__name
    mcmc_covmat = gd_sample.getCovMat().matrix[:dim, :dim]
    ibest = gd_sample.samples.T[dim].argmin()
    mcmc_best = gd_sample.samples[ibest,:dim]

    print('\n Residuals (percentage) relative to analytical answer...')
    print("... Evaluated best-fit ( a0,..,a{0:d} ) = ( ".format(dim-1)
          +','.join(['%.1e' % (pval,) for pval in np.fabs(params[chi2.argmin()]/apoly - 1.0)*1e2])+" )")
    print("... Quadratic form ( a0,..,a{0:d} ) = ( ".format(dim-1)
          +','.join(['%.1e' % (pval,) for pval in np.fabs(params_best/apoly - 1.0)*1e2])+" )")
    print("... MCMC (final) ( a0,..,a{0:d} ) = ( ".format(dim-1)
          +','.join(['%.1e' % (pval,) for pval in np.fabs(mcmc_best/apoly - 1.0)*1e2])+" )")
    print('\nTotal number of function evaluations = {0:d}'.format(chi2.size))

    gdplot = gdplt.get_subplot_plotter(subplot_size=2.)
    plot_param_list = ['a'+str(d) for d in range(dim)]
    plot_limits_dict = {}
    for d in range(dim):
        plot_limits_dict['a'+str(d)] = (info['params']['a'+str(d)]['prior']['min'],
                                        info['params']['a'+str(d)]['prior']['max'])

    gdplot.triangle_plot([gd_sample], plot_param_list, 
                         filled=True,contour_colors=['crimson'],legend_loc='upper center',
                         param_limits=plot_limits_dict)
    gdplot.add_text('$N_{{\\rm eval,PICASA}} = {0:d}$'.format(chi2.size),
                    x=-0.75,y=2.35,fontsize=14,c='crimson')
    for par_x in range(dim):
        for par_y in range(par_x,dim):
            if par_x != par_y:
                ax = gdplot.get_axes_for_params('a'+str(par_x),'a'+str(par_y))
                ax.axhline(params_true[par_y],c='k',ls=':',lw=1)
                ax.axhline(mcmc_best[par_y],c='crimson',ls='-',lw=1.,alpha=0.6)
                ax.axhline(apoly[par_y],c='b',ls='--',lw=1.,alpha=0.6)

                ax.axvline(params_true[par_x],c='k',ls=':',lw=1)
                ax.axvline(mcmc_best[par_x],c='crimson',ls='-',lw=1.,alpha=0.6)
                ax.axvline(apoly[par_x],c='b',ls='--',lw=1.,alpha=0.6)
            else:
                ax = gdplot.subplots[par_x,par_x]
                ax.axvline(params_true[par_x],c='k',ls=':',lw=1)
                ax.axvline(mcmc_best[par_x],c='crimson',ls='-',lw=1.,alpha=0.6)
                ax.axvline(apoly[par_x],c='b',ls='--',lw=1.,alpha=0.6)
    gdplot.export(fname='demo_results.pdf',adir='../examples/plots/')

    Nsample = gd_sample.samples.shape[0]
    Nboot = 500
    ind = rng.choice(Nsample,size=np.min([Nboot,Nsample]),replace=False)
    model_b = np.zeros((Nboot,stuff.xvals.size),dtype=float)
    print('Bootstrapping...')
    for b in range(Nboot):
        model_b[b] = stuff.my_model(stuff.xvals,gd_sample.samples[ind[b],:dim])
        pic.status_bar(b,Nboot)

    picasa_16pc = np.percentile(model_b,16,axis=0)
    picasa_84pc = np.percentile(model_b,84,axis=0)

    del model_b,ind
    gc.collect()

    plt.figure(figsize=(5,5))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(stuff.xvals,stuff.my_model(stuff.xvals,params_true),'k:',lw=2,label='true')
    plt.errorbar(stuff.xvals,stuff.data,yerr=err_data,capsize=5,elinewidth=0.5,c='purple',ls='none',marker='o')
    plt.plot(stuff.xvals,stuff.my_model(stuff.xvals,mcmc_best),'-',c='crimson',lw=2,label='PICASA')
    plt.fill_between(stuff.xvals,picasa_16pc,picasa_84pc,color='crimson',alpha=0.2)
    plt.plot(stuff.xvals,stuff.my_model(stuff.xvals,apoly),'--',c='b',lw=0.8,label='analytical')
    plt.legend(loc='upper left')
    plt.savefig('../examples/plots/demo_data.pdf',bbox_inches='tight')





        
