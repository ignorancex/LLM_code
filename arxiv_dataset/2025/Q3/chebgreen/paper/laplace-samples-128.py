from chebgreen.greenlearning.utils import DataProcessor
from chebgreen.greenlearning.model import *
from chebgreen.backend import plt
from chebgreen.utils import runCustomScript
from chebgreen.chebpy2 import Chebfun2, Chebpy2Preferences

def green(x,s):
    g = 0
    g = (x <= s) * (x * (1-s)) + (x > s) * (s * (1-x))
    return g

example = 'laplace'
script = 'generate_example'
theta = None
Nsample = 256
lmbda = 0.01
Nf = 500
Nu = 500
noise_level = 0.0
runCustomScript(script,example,theta,Nsample,lmbda,Nf,Nu,noise_level)

dimension = 1
domain = [0,1,0,1]
layerConfig = [50,50,50,50]
activation = 'rational'
dirichletBC = True

eps = 1e-6
cheb2prefs = Chebpy2Preferences()
cheb2prefs.prefx.eps = eps
cheb2prefs.prefy.eps = eps
g = Chebfun2(green, domain = domain, prefs = cheb2prefs, simplify = True)
gnorm = g.norm()

data = DataProcessor(f"datasets/{example}/data.mat")
data.generateDataset(trainRatio = 0.5)

model = GreenNN()

model.build(dimension, domain, layerConfig, activation, dirichletBC)

print("-------------------------------------------------------------------------------")
print(f"Training greenlearning model for example \'{example}\' with {noise_level*100}% noise")
lossHistory = model.train(data, epochs = {'adam':int(4000), 'lbfgs':int(0)})

xF, xU = data.xF, data.xU
x, y = np.meshgrid(xU, xF)
G = model.evaluateG(x,y)
N = model.evaluateN(data.xF)
GreenNNPath = "temp/"
model.build(dimension = dimension,domain = domain, dirichletBC = dirichletBC, loadPath = GreenNNPath, device = torch.device('cuda:1'))

print("-------------------------------------------------------------------------------")
print(f"Learning a chebfun model for example \'{example}\' with {noise_level*100}% noise")
# ChebGreen
cheb2prefs = Chebpy2Preferences()
Gcheb = Chebfun2(model.evaluateG, domain = model.domain, prefs = cheb2prefs, simplify = True)

cheb2prefs = Chebpy2Preferences()
cheb2prefs.prefx.eps = eps
cheb2prefs.prefy.eps = eps
e = Chebfun2(lambda x,y: np.abs(Gcheb[x,y] - green(x,y)), domain = model.domain, prefs = cheb2prefs, simplify = True)

print(f"Error: {e.norm()/gnorm}")


xx = np.linspace(domain[0],domain[1],2000)
yy = np.linspace(domain[2],domain[3],2000)
x, y = np.meshgrid(xx,yy)
E = np.abs(e[x,y])

vmin = 0
vmax = 0.0036

fig = plt.figure(figsize = (13,10), frameon=False)
plt.axis('off')
plt.gca().set_aspect('equal', adjustable='box')
levels = np.linspace(vmin, vmax, 100, endpoint = True)
plt.contourf(x,y, E, levels = levels, cmap = 'jet', vmin = vmin, vmax = vmax)
# ticks = np.linspace(vmin, vmax, 10, endpoint=True)
# cbar = plt.colorbar(ticks = ticks, fraction = 0.046, pad = 0.04)

savePath = f"plots/"
Path(savePath).mkdir(parents=True, exist_ok=True)
fig.savefig(f'{savePath}/laplace-60-error.png', dpi = fig.dpi, bbox_inches='tight', pad_inches=0)