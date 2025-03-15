import os

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

save    = True
model   = 'asso'
context = 'cshrf'
N       = 31
n4      = [0, int(N*0.3), int(N*0.7), N-1]

path = {
  'root': root,
  'epi-model': os.path.join(root,'code','epi-model','src'),
  'figs':      os.path.join(root,'outputs',context,model,'figs'),
  'data':      os.path.join(root,'outputs',context,model,'data'),
  'tikz':      os.path.join(root,'outputs','tikz'),
  'specs':     os.path.join(root,'code','main','specs'),
  'model':     os.path.join(root,'code','main'),
  'paper':     os.path.join(root,'docs','paper'),
}

def epimodel():
  import sys
  sys.path.append(path['epi-model'])

def numpy():
  import numpy as np
  np.set_printoptions(linewidth=160)

def plot(tex=False):
  import matplotlib.pyplot as plt
  plt.style.use('seaborn-whitegrid')
  plt.rc('grid',color=[0.95,0.95,0.95])
  plt.rc('legend',frameon=True,framealpha=0.9)
  plt.rc('savefig',directory=path['figs'],format='pdf')
  if tex:
    plt.rc('text',usetex=True)
    # plt.rc('font',family='serif',serif=['computer modern roman'])
