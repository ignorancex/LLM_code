import os,sys; sys.path.append(os.path.join((lambda r,f:f[0:f.index(r)+len(r)])('code',os.path.abspath(__file__)),'config')); import config
config.epimodel()
from space import Dimension,Space,Array
from simulation import Simulation, Model
from elements import Selector,Output
import numpy as np
import transmit
import modelutils

def dxfun(X,t,P,dxout={}):
  dX  = X*0
  dX += X*P['nu']
  dX -= X*P['mu']
  mu  = P['mu']+P['f']
  xf0 = X.islice(i=0)*P['f']
  xf1 = X.islice(i=1)*P['f']
  transmit.transfer(dX,src={'i':[0]},dst={'i':[1]},N=xf0)
  transmit.transfer(dX,src={'i':[1]},dst={'i':[0]},N=xf1)
  for v in dxout: dxout.update({v:locals()[v]})
  return dX

def propaccum():
  pass

def propfrom(sim,t,xf1,mu):
  it = list(sim.t).index(t)
  xfp = sim.outputs['propfrom'].islice(t=sim.t[it-1]) * sim.X.islice(i=0,t=sim.t[it-1]) if it else 0
  return (xfp * (1-mu) + xf1) / sim.X.islice(i=0,t=t)

space = Space([Dimension('index','i',[0,1])])
t = np.arange(0,100,0.1)

output = Output(
  name = 'propfrom',
  space = Space([modelutils.tdim(t)]),
  fun = propfrom,
  accum = None,
  calc = 'peri',
  dxout = ['xf1','mu'],
)
params = {
  'nu': 0.03,
  'mu': 0.03,
  'f':  0.03,
}
selectors = {
  str(i): Selector(
    name = str(i),
    title = str(i),
    select = {'i':[i]},
    color = c,
  )
  for i,c in zip([0,1],[[1,0,0],[0,0,1]])
}
X0 = Array([1,1],space)
model = Model(X0=X0,dxfun=dxfun,params=params)
sim = Simulation(model=model,t=t,outputs=[output])
sim.solve()
print(sim.outputs['propfrom'].space)
# sim.plot(
#   output='propfrom',
#   selectors=[selectors['1']],
# )
# sim.plot(
#   selectors=[selectors['0'],selectors['1']]
# )
