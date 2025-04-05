import sys
import numpy as np
from analysis.params import ParamRun
from likelihood.like import Likelihood
from likelihood.sampler import Sampler
from model.data import DataManager
from model.theory import get_theory
from model.power_spectrum import HalomodCorrection
from model.utils import selection_planck_erf, selection_planck_tophat
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("param_file", help="yaml target parameter file")
parser.add_argument('--use-mpi', default=False, action='store_true',
                    help='Use MPI (default: False)')
parser.add_argument('--jk-region', type=int, default=-1,
                    help='Jackknife region to use (default: no jackknives)')
parser.add_argument('--data-name', type=str, default='none',
                    help='Name of the data vector to analyze')

o = parser.parse_args()

if o.param_file == 'none':
    raise ValueError("Must provide param file name")

p = ParamRun(o.param_file)

# Jackknives
if o.jk_region < 0:
    jk_region = None
else:
    jk_region = o.jk_region

def extract_map_p0(p, v, parnames):
    """Extract the proposal p0 from a specific map."""
    for m in p.get("maps"):
        if m["name"] == v["name"]:
            break

    p0 = [m["model"][k] for k in parnames]
    return p0

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
    # if (v['name'] != o.data_name) and (o.data_name != 'all'):
    #     continue

    print(v['name'])

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

    p0 = extract_map_p0(p, v, lik.p_free_names)  # p0 for particular map
    # Benchmarks
    print(dict(zip(lik.p_free_names, p0)))
    print("chisq:", lik.chi2(p0))
    exit(1)
    # Set up sampler
    sam = Sampler(lik.lnprob, lik.p0, lik.p_free_names,
                  p.get_sampler_prefix(v['name']),
                  p.get('mcmc'))

    # Benchmarks
    # print(dict(zip(lik.p_free_names, lik.p0)))
    # print("chisq:", lik.chi2(lik.p0))

    # Compute best fit and covariance around it
    if not sam.read_properties():
        print(" Computing best-fit and covariance")
        sam.get_best_fit(update_p0=True)
        cov0 = sam.get_covariance(update_cov=True)
        sam.save_properties()

    print(" Best-fit parameters:")
    for n, v, s in zip(sam.parnames, sam.p0, np.sqrt(np.diag(sam.covar))):
        print("  " + n + " : %.3lE +- %.3lE" % (v, s))
        if n == p.get("mcmc")["save_par"]: par.append(v)
    print(" chi^2 = %lf" % (lik.chi2(sam.p0)))
    print(" n_data = %d" % (len(d.data_vector)))

    if sam.nsteps > 0:
        # Monte-carlo
        print(" Sampling:")
        sam.sample(carry_on=p.get('mcmc')['continue_mcmc'], verbosity=1,
                   use_mpi=o.use_mpi)

if len(par) > 0:
    is_jk = str(jk_region) if bool(jk_region) else ""
    fname = p.get_outdir() + "/" + p.get("mcmc")["save_par"] + \
            "_" + p.get("mcmc")["run_name"] + "_" + is_jk
    np.save(fname, np.array(par))
