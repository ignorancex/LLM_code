import os,sys; sys.path.append(os.path.join((lambda r,f:f[0:f.index(r)+len(r)])('code',os.path.abspath(__file__)),'config')); import config
import numpy as np
config.epimodel()
config.numpy()
config.plot()

import matplotlib.pyplot as plt
from collections import OrderedDict as odict
import modelutils
import system

dt = 0.005
output = 'prevalence'
specs = system.get_specs()
models = {
  'homo': system.get_model(),
  'zeta': system.get_model(),
}
# define true homogeneous model
models['homo'].collapse(['ii'])
# define high zeta model
models['zeta'].params['pe'].update(np.nan)
models['zeta'].params['dur'].update([0.01,0.05,0.8])
models['zeta'].params['zeta'].update(np.nan,ii='H',ip='L')

sims = odict([
  ('Homogeneous',   system.get_simulation(models['homo'],dt=dt)),
  ('High Turnover', system.get_simulation(models['zeta'],dt=dt)),
])
for (name,sim),clr in zip(sims.items(),[[1,0,0],[0,0,1]]):
  sim.init_outputs(system.get_outputs(
    spaces = sim.model.spaces,
    select = sim.model.select,
    t = system.get_t(dt=dt),
    names = [output],
  ))
  for param in ['pe','dur','zeta']:
    print('{} = \n{}'.format(param,np.around(sim.model.params[param],6)),flush=True)
  sim.solve(eqfun=lambda s,t: print(t,flush=True))
  y = modelutils.taccum(sim.outputs[output].iselect(sim.model.select['all']))
  plt.plot(sim.t,y,color=clr)
  sim = None

plt.ylim([0,0.1])
plt.ylabel('Prevalence (overall)')
plt.xlabel('t (years)')
plt.legend(sims.keys())
plt.savefig('infinite-zeta-tmp.eps')
plt.savefig('infinite-zeta-tmp.png')
plt.show()



