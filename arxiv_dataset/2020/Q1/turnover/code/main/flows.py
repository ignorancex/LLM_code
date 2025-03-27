# load code / config
import os,sys;
sys.path.append(os.path.join((lambda r,f:f[0:f.index(r)+len(r)])('code',os.path.abspath(__file__)),'config'));
import config
config.epimodel()
config.numpy()
config.plot()
# external module
import re
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
# epi-model modules
import utils
import modelutils
from space import Space,Array
from elements import Color
# relative imports
import system
import sensitivity

names = ['S','I','T']
lights = [0.6,0.2,0.4]
selects = ['high','med','low']
# names = ['S','infected']
out = 'X'

def gen_pie_data(sim,label,select,phi):
  SIR = np.array([
    modelutils.taccum(
      sim.outputs[out],
      **sim.model.select[name].union(sim.model.select[select])
    ).islice(t=sim.t[-1])
    for name in names
  ])
  SIR = SIR / SIR.sum()
  SIR[0] = 1-SIR[1:].sum()
  tdir = os.path.join(config.path['data'],'flows',label)
  if config.save:
    utils.makedir(tdir)
    utils.savetxt(os.path.join(tdir,'phi.tex'),np.float(phi))
    for name,value in zip(names,SIR):
      utils.savetxt(os.path.join(tdir,'flow-{}-{}.tex'.format(select,name)),np.float(value))

def make_tikz(label,phi):
  tikzdir = os.path.join(config.path['tikz'],'flows')
  tdir    = os.path.join(config.path['data'],'flows',label)
  flowdir = os.path.join(config.path['figs'],'flows')
  configstr = '\n'.join([
    '\\newcommand{{\\x{}{}}}{{{}}}'.format(name,select,
        np.around(utils.loadtxt(os.path.join(tdir,'flow-{}-{}.tex'.format(select,name))),4)
      ) for name in names for select in selects
  ])+'\n\\newcommand{\\turnover}{'+str(int(10*(phi-0.03)**(1/3)))+'}'
  utils.savetxt(os.path.join(tikzdir,'config.tex'),configstr)
  os.system('cd {} && pdflatex flows.tex >/dev/null && cp flows.pdf {}/{}'.format(
    tikzdir, flowdir, 'flows-{}.pdf'.format(label) ))

def make_legend(sim,d='h'):
  xy = [np.nan*np.ones((3,))]*2
  fs = {'h':(4.5,0.6), 'v':(1.6,1.0)}
  nc = {'h': 3, 'v': 1}
  plt.figure(figsize=fs[d])
  for name,light in zip(names,lights):
    select = sim.model.select[name]
    plt.fill_between(*xy,label='$\\textrm{'+select.label+'}$',color=select.color.lighten(light))
  plt.legend(ncol=nc[d])
  plt.axis('off')
  plt.savefig(os.path.join(config.path['figs'],'flows','flows-legend-{}.pdf'.format(d)))

def run_sims():
  phis = list(sensitivity.iter_phi())
  for label,phi in [
      ('none', phis[config.n4[0]]),
      ('low',  phis[config.n4[1]]),
      ('high', phis[config.n4[2]]),
      ('extr', phis[config.n4[3]])
    ]:
    specs = system.get_specs()
    model = system.get_model()
    sim = sensitivity.get_sim(phi,0.1)
    sim.init_outputs(system.get_outputs(
      spaces = sim.model.spaces,
      select = sim.model.select,
      t = sim.t,
      names = [out]
    ))
    sim.solve()
    for select in selects:
      gen_pie_data(sim,label,select,phi)
    make_tikz(label,phi)
    make_legend(sim)
