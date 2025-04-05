import sys
import numpy as np
from analysis.params import ParamRun
from likelihood.like import Likelihood
from likelihood.sampler import Sampler
from model.data import DataManager
from model.theory import get_theory
from model.power_spectrum import HalomodCorrection, hm_bias
from model.utils import selection_planck_erf, selection_planck_tophat

try:
    fname_params = sys.argv[1]
except IndexError:
    raise ValueError("Must provide param file name as command-line argument")

try:
    jk_region=int(sys.argv[2])
except IndexError:
    raise ValueError("Must provide jackknife region")

p = ParamRun(fname_params)

# Cosmology (Planck 2018)
cosmo = p.get_cosmo()

# Include halo model correction if needed
if p.get('mcmc').get('hm_correct'):
    hm_correction = HalomodCorrection(cosmo)
else:
    hm_correction = None

# Include selection function if needed
sel = p.get('mcmc').get('selection_function')
if sel is not None:
    if sel == 'erf':
        sel = selection_planck_erf
    elif sel == 'tophat':
        sel = selection_planck_tophat
    elif sel == 'none':
        sel = None

par = []
for v in p.get('data_vectors'):
    print(v['name'])

    param_jk_values={}
    print(jk_region)
    
    # Construct data vector and covariance
    d = DataManager(p, v, cosmo, jk_region=jk_region)

    # Theory predictor wrapper
    def th(pars):
        return get_theory(p, d, cosmo,
                          hm_correction=hm_correction,
                          selection=sel,
                          **pars)
    
    # Set up likelihood
    lik = Likelihood(p.get('params'), d.data_vector, d.covar, th,
                     template=d.templates, debug=p.get('mcmc')['debug'])
    
    # Set up sampler
    sam = Sampler(lik.lnprob, lik.p0, lik.p_free_names,
                  p.get_sampler_prefix(v['name'])+"jk%d"%jk_region,
                  p.get('mcmc'))
    
    # Compute best fit and covariance around it
    if not sam.read_properties():
        print(" Computing best-fit and covariance")
        sam.get_best_fit(update_p0=True)
        cov0 = sam.get_covariance(update_cov=True)
        sam.save_properties()

    print(" Best-fit parameters:")
    for nam, val, sig in zip(sam.parnames, sam.p0, np.sqrt(np.diag(sam.covar))):
        param_jk_values[nam]=val
        print("  " + nam + " : %.3lE +- %.3lE" % (val, sig))
        if nam == p.get("mcmc")["save_par"]: par.append(val)
    z, nz = np.loadtxt(d.tracers[0][0].dndz, unpack=True)
    zmean = np.average(z, weights=nz)
    sigz = np.sqrt(np.sum(nz * (z - zmean)**2) / np.sum(nz))
    zarr=np.linspace(zmean-sigz,zmean+sigz,10)
    bg=np.mean(hm_bias(cosmo, 1./(1+zarr),d.tracers[0][0].profile,
                       **(lik.build_kwargs(sam.p0))))
    by=np.mean(hm_bias(cosmo, 1./(1+zarr),d.tracers[1][1].profile,
                       **(lik.build_kwargs(sam.p0))))
    chi2=lik.chi2(sam.p0)
    print("  b_g : %.3lE" % bg)
    print("  b_y : %.3lE" % by)
    print(" chi^2 = %lf" % chi2)
    print(" n_data = %d" % (len(d.data_vector)))
    param_jk_values['b_g']=bg
    param_jk_values['b_y']=by
    param_jk_values['chi2']=chi2

    np.savez(p.get_sampler_prefix(v['name'])+"jk%d_vals"%jk_region,
             M1=np.array(param_jk_values['M1']),
             Mmin=np.array(param_jk_values['Mmin']),
             b_hydro=np.array(param_jk_values['b_hydro']),
             r_corr=np.array(param_jk_values['r_corr']),
             width=np.array(param_jk_values['width']),
             b_g=np.array(param_jk_values['b_g']),
             b_y=np.array(param_jk_values['b_y']),
             chi2=np.array(param_jk_values['chi2']))
