# move to parent dir
import os, sys
THIS_PATH = os.path.dirname(os.path.realpath(__file__))
NEW_PATH = "/".join(THIS_PATH.split("/")[:-1])
os.chdir(NEW_PATH)
sys.path.append(NEW_PATH)
####
import sys
import numpy as np
from analysis.params import ParamRun
from likelihood.like import Likelihood
from likelihood.sampler import Sampler
from model.data import DataManager
from model.power_spectrum import HalomodCorrection, hm_bias
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.cm import copper
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


try:
    fname_params = sys.argv[1]
    bin_name = sys.argv[2]
except IndexError:
    fname_params = "params_dam_wnarrow.yml"
    bin_name = "wisc3"

p = ParamRun("params_dam_wnarrow.yml")
run_name = p.get('mcmc')['run_name']
cosmo = p.get_cosmo()
# Include halo model correction if needed
if p.get('mcmc').get('hm_correct'):
    hm_correction = HalomodCorrection(cosmo)
else:
    hm_correction = None
for v in p.get('data_vectors'):
    if v['name'] == bin_name:
        d = DataManager(p, v, cosmo)
        z, nz = np.loadtxt(d.tracers[0][0].dndz, unpack=True)
        zmean = np.average(z, weights=nz)
        sigz = np.sqrt(np.sum(nz * (z - zmean)**2) / np.sum(nz))

        # Theory predictor wrapper
        def th(pars):
            return get_theory(p, d, cosmo, return_separated=False,
                              hm_correction=hm_correction,
                              selection=sel,
                              **pars)
        lik = Likelihood(p.get('params'), d.data_vector, d.covar, th,
                         debug=p.get('mcmc')['debug'])
        sam = Sampler(lik.lnprob, lik.p0, lik.p_free_names,
                      p.get_sampler_prefix(v['name']), p.get('mcmc'))
        sam.get_chain()
        figs_ch = lik.plot_chain(sam.chain, save_figure=True,
                                 prefix='notes/paper/')
plt.show()
