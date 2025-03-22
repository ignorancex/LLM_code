import os,sys
sys.path.append(os.path.join((lambda r,f:f[0:f.index(r)+len(r)])('code',os.path.abspath(__file__)),'config'));
import config
config.epimodel()
config.plot()
import numpy as np
config.numpy()
import matplotlib.pyplot as plt
from collections import OrderedDict as odict
from textwrap import wrap
from copy import deepcopy
import utils
import modelutils
import system
import calibration

def fname_fig(compare,output,selector,**params):
  path = os.path.join(config.path['figs'],'compare')
  utils.makedir(path)
  return os.path.join(path,
    '-'.join(
      [config.model,compare,output,selector]+
      ['{}={}'.format(name,value) for name,value in params.items()])+'.pdf'
  )

def fname_fit(name):
  return os.path.join(config.path['data'],'fit',shortname(name)+'.json')

@system.xmodel
def load_fit(name,sim):
  sim = deepcopy(sim)
  model = sim.model
  model.params.fromdict(utils.loadjson(fname_fit(name)))
  sim.init_model(model)
  sim.init_params()
  return sim

def shortname(name):
  return name.lower().replace(' ','-')

def txtsave(sims,output):
  def vfun(sim,output,selector):
    return modelutils.taccum(sim.outputs[output],**sim.model.select[selector]).islice(t=sim.t[-1])
  def fname(name,output,selector):
    return os.path.join(config.path['data'],'values','-'.join([shortname(name),output,selector])+'.txt')
  fmts = {
    'prevalence': lambda x: '{:.1f}\%'.format(100*float(x)),
    'C':          lambda x: '{:.1f}'.format(float(x)),
    'ratio':      lambda x: '{:.1f}'.format(float(x)),
    'tpaf-high':  lambda x: '{:.3f}'.format(float(x)),
    'tpaf-med':   lambda x: '{:.3f}'.format(float(x)),
    'tpaf-low':   lambda x: '{:.3f}'.format(float(x)),
  }
  for name,sim in sims.items():
    utils.savetxt(fname(name,output,'high'),  fmts[output](vfun(sim,output,'high')))
    utils.savetxt(fname(name,output,'med'),   fmts[output](vfun(sim,output,'med')))
    utils.savetxt(fname(name,output,'low'),   fmts[output](vfun(sim,output,'low')))
    utils.savetxt(fname(name,output,'ratio-high-low'), fmts['ratio'](vfun(sim,output,'high') / vfun(sim,output,'low')))
    utils.savetxt(fname(name,output,'ratio-high-med'), fmts['ratio'](vfun(sim,output,'high') / vfun(sim,output,'med')))
    utils.savetxt(fname(name,output,'ratio-med-low'),  fmts['ratio'](vfun(sim,output,'med')  / vfun(sim,output,'low')))

def get_sim(turnover=True,asso=False,t=None):
  specs = system.get_specs()
  model = system.get_model()
  if not turnover:
    model.params['dur'].update(np.nan)
    model.params['phi'].update(np.nan)
  if asso:
    model.params['eps'] = 0.5
  else:
    model.params['eps'] = 0.0
  return system.get_simulation(model,t=t)

def plot_iter(sims,output,selector):
  legend = []
  # colors = [[0.8,0.0,0.0],[1.0,0.6,0.6],[0.8,0.0,0.0],[1.0,0.6,0.6]]
  colors = [[0.0235,0.1608,0.3451]]*2+[[0.5216,0.6588,0.8431]]*2
  linestyles = ['--','-','--','-']
  ylim = {'paper': None, 'isstdr': [0.5, 1.0], 'cshrf': [.30,.82]}[config.context]
  for (name,sim),color,ls in zip(sims.items(),colors,linestyles):
    legend.append(name)
    select = sim.model.select[selector]
    select.color = color
    sim.plot(
      output = output,
      selectors = [sim.model.select[selector]],
      xlabel = 'Time (years)',
      show = False,
      legloc = False,
      linestyle = ls,
      ylim = ylim,
    )
  if config.context in ['paper']:
    plt.legend(legend)
  if config.context == 'isstdr':
    plt.legend(['Turnover','No Turnover'], loc='lower right')
    plt.xlabel(plt.gca().get_xlabel(),fontsize='x-large')
    plt.ylabel('\n'.join(wrap(plt.gca().get_ylabel(),13,break_on_hyphens=False)),
      fontsize='x-large',
      labelpad=40,
      rotation=0,
      va='center',
    )

def run_sim(sim,outputs=None):
  outputs = outputs if outputs is not None else []
  sim.init_outputs(system.get_outputs(
    spaces = sim.model.spaces,
    select = sim.model.select,
    t = sim.t,
    names = outputs
  ))
  return sim.solve()

def exp_run_plot(compare,sims,outputs,selectors,txt=False,**params):
  for sim in sims.values():
    for name,value in params.items():
      if name in sim.model.params:
        sim.model.params[name].update(value)
    sim.update_params(sim.model.params)
    run_sim(sim,outputs)
  figsize = {'paper': (4,3), 'isstdr': (4.5,3), 'cshrf': (4,2.5)}[config.context]
  axespos = {
  'paper': [0.16,0.14,0.82,0.84],
  'isstdr':[0.33,0.16,0.65,0.82],
  'cshrf': [0.16,0.14,0.82,0.84],
  }[config.context]
  for output in outputs:
    for selector in selectors:
      plt.figure(figsize=figsize)
      plot_iter(sims,output,selector)
      plt.gca().set_position(axespos)
      if config.save:
        plt.savefig(fname_fig(compare,output,selector,**params))
        plt.close()
      else:
        plt.show()
    if txt and config.save:
      txtsave(sims,output)

def run_fit():
  t = system.get_t(tmax=500)
  sims = odict([
    ('Turnover',  get_sim('full',t=t)),
    ('No Turnover', get_sim('no-turnover',t=t)),
  ])
  for name,sim in sims.items():
    sim.init_outputs(system.get_outputs(
      spaces = sim.model.spaces,
      select = sim.model.select,
      t = sim.t,
      names = ['prevalence'])
    )
    sim.solve()
    targets = system.get_targets(
      sim.model.spaces,
      sim.model.select,
      sim.outputs,
      t = sim.t,
    )
    if config.model == 'mort':
      sim.model.params['C'].update([25,10,5]) # HACK to get low prev > 0
    calsim = calibration.CalibrationSim(
      name,
      sim = sim,
      targets = targets,
      verbose = True,
    )
    calsim.optimize(ftol=1e-3,plot='tmp.png')
    if config.save:
      utils.savejson(fname_fit(name),calsim.fitted_params().todict())
    else:
      print(dict(calsim.fitted_params().todict()))

def gen_phi_base():
  phi = get_sim().model.params['phi']
  fmt = lambda x: '{:.4f}'.format(x)
  s = '\\left[\\begin{{array}}{{{}}}\n{}\n\\end{{array}}\\right]'.format(
    'c'*phi.shape[1], '\\\\\n'.join([' & '.join([fmt(ij) for ij in i]) for i in phi])
  ).replace(fmt(0),'*')
  fname = os.path.join(config.path['data'],'values','phi-base.tex')
  utils.savetxt(fname,s)

def simple_turnover():
  sims = odict([
    ('Turnover',    get_sim('full')),
    ('No Turnover', get_sim('no-turnover')),
  ])
  for tau in [0.1, 0.2]:
    exp_run_plot('turnover',
      sims      = sims,
      outputs   = ['prevalence','incidence'],
      selectors = ['all','high','med','low'],
      tau       = tau,
      # infect    = [[0.5/3],[2.0/3],[7.5/3]], # TEMP
    )

def exp_tpaf():
  if config.context == 'paper':
    exp_tpaf_paper()
  if config.context == 'isstdr':
    print('TBD')
  if config.context == 'cshrf':
    exp_tpaf_cshrf()

def exp_tpaf_cshrf():
  t = system.get_t(tmax=10)
  mmap = {'main':[0,1],'asso':[2,3],'both':[0,1,2,3]}
  for iname,im in mmap.items():
    sims = [
      ('Turnover [Prop]',      load_fit('Turnover',   get_sim(turnover=True, asso=False,t=t),model='main')),
      ('No Turnover [Prop]',   load_fit('No Turnover',get_sim(turnover=False,asso=False,t=t),model='main')),
      ('Turnover [Assort]',    load_fit('Turnover',   get_sim(turnover=True, asso=True, t=t),model='asso')),
      ('No Turnover [Assort]', load_fit('No Turnover',get_sim(turnover=False,asso=True, t=t),model='asso')),
    ]
    sims = odict([sims[i] for i in im])
    exp_run_plot('tpaf',
      sims      = sims,
      outputs   = ['tpaf-high'],
      selectors = ['all'],
      vs        = iname,
      txt       = False,
    )

def exp_tpaf_paper():
  tmax = {'paper': 50, 'isstdr': 30}[config.context]
  t = system.get_t(tmax=tmax)
  for case in ['raw','fit','both']:
    print(case,flush=True)
    sims = odict([
      ('Turnover',    get_sim(turnover=True, t=t)),
      ('No Turnover', get_sim(turnover=False,t=t)),
    ])
    if case == 'raw':
      pass
    if case == 'fit':
      for name in list(sims.keys()):
        sims.update([(name+' [fit]', load_fit(name,sims.pop(name)))])
    if case == 'both':
      for name in list(sims.keys()):
        sims.update([(name+' [fit]', load_fit(name,sims.get(name)))])
    exp_run_plot('tpaf',
      sims      = sims,
      outputs   = ['tpaf-high'], # TPAF
      selectors = ['all'],
      vs        = case,
      txt       = True,
    )
  # equilibrium prevalence plot
  names = list(sims.keys())
  for name in names:
    sim_eq = sims[name].model.equilibriate(tmax=500,tol=1e-6)
    sims[name]._model.X0 = sim_eq.X.islice(t=sim_eq.teq)
    sims[name].model.params['infect'].update(0)
  exp_run_plot('tpaf',
    sims      = sims,
    outputs   = ['prevalence','C'],
    selectors = ['low','med','high','all'],
    txt       = True,
  )
