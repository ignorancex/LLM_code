import os,sys;
sys.path.append(os.path.join((lambda r,f:f[0:f.index(r)+len(r)])('code',os.path.abspath(__file__)),'config'));
import config
import numpy as np
config.epimodel()
config.plot(tex=True)
import matplotlib.pyplot as plt
from collections import OrderedDict as odict
import modelutils
import elements
import system

def iterdur(min,max,N):
  for i in np.logspace(np.log10(min),np.log10(max),N)[::-1]:
    yield i

def get_sim(type,output,dh=None):
  model = system.get_model()
  if type == 'homo':
    model.collapse(['ii','ip'])
    name = '$G = 1$ (Homogeneous)'
    dt = 0.2
  if type == 'zeta':
    dhmax = 1.0/model.params['mu']
    model.params['dur'].update([dh,min(dh*5,dhmax),min(dh*30,dhmax)])
    # z1 = (1/model.params['dur'].iselect(ii='H',ki='M') - model.params['mu'])/2
    # z2 = (1/model.params['dur'].iselect(ii='M',ki='M') - model.params['mu'])/2
    # z3 = (1/model.params['dur'].iselect(ii='L',ki='M') - model.params['mu'])/2
    # model.params['zeta'].update(z1,ii='H',ip='M')
    # model.params['zeta'].update(z1,ii='H',ip='L')
    # model.params['zeta'].update(z2,ii='M',ip='L')
    # model.params['zeta'].update(z2,ii='M',ip='H')
    # model.params['zeta'].update(z3,ii='L',ip='M')
    # model.params['zeta'].update(z3,ii='L',ip='H')
    model.params['zeta'].update(np.nan)
    # model.params['pe'].update(np.nan)
    name = '$G = 3, \\delta_H = {:.03f}$'.format(model.params['dur'][0,0])
    dt = min(dh*0.8,0.2)
  t = system.get_t(dt=dt,tmin=0,tmax=200)
  sim = system.get_simulation(model,t=t)
  sim.init_outputs(system.get_outputs(
    spaces = sim.model.spaces,
    select = sim.model.select,
    t = sim.t,
    names = [output],
  ))
  return sim,name

def plotfun(sim,name,color,output,**specs):
  selector = sim.model.select['all']
  selector.color = color
  selector.specs.update(**specs)
  sim.solve()
  sim.plot(
    outputs = [output],
    selectors = [selector],
    ylabel = 'Prevalence Overall',
    show = False,
    leg = False,
  )
  return [name]

def make_plot(output,min,max,N,homo=False):
  legend = []
  cmap = elements.Color([1,0,0]).cmap(N,[-0.7 ,+0.7])[::-1]
  durs = iterdur(min,max,N)
  fb = 2.5 if homo else 1.0
  from pprint import pprint
  for dh,c in zip(durs,cmap):
    print(dh,flush=True)
    sim,name = get_sim('zeta',dh=dh,output=output)
    sim.params['beta'] *= fb
    legend += plotfun(sim,name,output=output,color=c)
  if homo:
    sim,name = get_sim('homo',output=output)
    sim.params['beta'] *= fb
    legend += plotfun(sim,name,output=output,color=[0,0,0],linestyle='--')
  plt.legend(legend,loc='lower right',fontsize=8)
  save = 'vary-zeta-{}-[{:0.2f},{:0.2f}].eps'.format(output,min,max)
  plt.savefig(os.path.join(config.path['figs'],'plots','compare',save))
  plt.show()

if __name__ == '__main__':
  # make_plot('prevalence',3,33,5)
  make_plot('prevalence',0.005,0.5,9,homo=True)
