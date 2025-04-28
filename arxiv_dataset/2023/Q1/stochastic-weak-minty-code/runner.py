#%%
import os
from datetime import datetime
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
from datargs import argsclass, parse
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
import jax.nn as jnn

plt.rcParams['figure.facecolor'] = 'white'


@argsclass
class Args:
    T: int = 10_000
    d: int = 1
    batch_size: int = 1
    gamma: float = 1.0
    alpha0: float = 1/18
    beta0: float = 1
    seed: int = None
    method: str = "BCSEG+"
    decrease : str = "linear"
    decrease_factor: float = 1
    projection: str = None
    radius: int = 1

    problem: str = "quadratic"
    L: float = 1.0
    rho: float = -1/20
    init: Sequence[float] = (0.5, 0.5)
    offset: float = 0.0
    
    noise: float = 0.1
    noise_model: str = "additive" # 

    plot_field: bool = False

    name: str = None

    def setup(self):
        if self.seed is None:
            random_data = os.urandom(4) 
            self.seed = int.from_bytes(random_data, byteorder="big")

        now = datetime.now()
        name = now.strftime("%Y-%m-%d_%Hh%Mm%Ss")
        if self.name is not None:
            name = f"{self.name}({name})"
        
        self.dir = os.path.join("output", name)
        Path(self.dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(self.dir, "args.txt"), "w") as args_file:
            args_file.write(str(self))
        return self

    def path(self, filename):
        return os.path.join(self.dir, filename)


def init_history(keys=[], length=0):
    history = {}
    for key in keys:
        history[key] = jnp.arange(length, dtype="float32")
    return history

def update_history(history, t, values: dict):
    for k,v in values.items():
        if k in history:
            history[k] = history[k].at[t].set(v)
    return history

args = Args()

#%%

if __name__ == '__main__':
    args = parse(Args).setup()

#%%

# Setup
args.setup()

def create_problem(args):
    if args.problem == "lagrangian":
        A = jnp.zeros((args.d,args.d))
        A = A + jnp.diag(jnp.ones(args.d), 0)
        A = A + jnp.diag(-jnp.ones(args.d-1), 1)
        A = jnp.fliplr(A)
        A = 1/4 * A
        H = 2*A.transpose().dot(A)
        b = 1/4*jnp.ones(args.d)
        h = jnp.zeros(args.d)
        h = h.at[args.d].set(1/4)

        def L(x,y):
            return 1/2 * x.dot(H).dot(x) - h.dot(x) - (A.dot(x) - b).dot(y)
    elif args.problem == "logsumexp": # TODO: double check what example this is
        A = jnp.zeros((args.d,args.d))
        A = A + jnp.diag(jnp.ones(args.d), 0)
        A = A + jnp.diag(-jnp.ones(args.d-1), 1)
        A = jnp.fliplr(A)
        A = 1/4 * A
        b = 1/4*jnp.ones(args.d)
        H = A.transpose().dot(A)
        def L(x,y):
            Lips = 1.0
            Z = H.dot(x)
            return jnn.logsumexp(Z, b=Lips, axis=-1) - (A.dot(x) - b).dot(y)
    elif args.problem == "bilinear":
        def L(x,y):
            return (x-args.offset).transpose().dot(y-args.offset)
    elif args.problem == "quadratic":
        a = jnp.sqrt(args.L**2 - args.L**4 * args.rho ** 2)
        b = args.L**2 * args.rho
        def L(x,y):
            x = x-args.offset
            y = y-args.offset
            return a * x.transpose().dot(y) + b/2 * jnp.sum(x**2, axis=-1) - b/2 * jnp.sum(y**2, axis=-1)
    elif args.problem == "GlobalForsaken":
        def L(x,y):
            phi = lambda z: 2*z**6/21 - z**4/3 + z**2/3
            return (x*y + phi(x) - phi(y)).sum(axis=-1)
    elif args.problem == "forsaken?": # TODO: double check what example this is
        def L(x,y):
            #return (6 * x+25 * x**2-30 * x**4+10 * x**6+60 * x.dot(y)-25 * y**2+30 * y**4-10 * y**6).sum()
            return ((5 * x**2)/12 - x**4/2 + x**6/6 - (5 * y**2)/12 + y**4/2 - y**6/6 + x*(0.1 + y)).sum(axis=-1)
    
    return L


def make_F(L):
    Fx = jit(vmap(grad(L, argnums=0)))
    Fy = jit(vmap(grad(L, argnums=1)))
    def F(z):
        x,y = z[:, :args.d],z[:, args.d:]
        return jnp.concatenate([Fx(x,y), -Fy(x,y)], axis=-1)
    return F


L = create_problem(args)
F = make_F(L)

# Stochastic
globalkey = jax.random.PRNGKey(args.seed)


if args.noise_model == "additive":
    def Fhat(z, xi):
        return F(z) + xi


    def sample_xi(globalkey):
        globalkey, subkey = jax.random.split(globalkey)
        noise = args.noise * jax.random.normal(subkey, shape=(args.batch_size, args.d*2))
        return noise, globalkey
elif args.noise_model == "ScaledLips":
    assert args.problem == "bilinear"

    # such that L_F = 1
    # Solve[3*a + 0.5*(1 - a) == 1, {a}]
    p = jnp.array([0.2, 0.8])
    xis = jnp.array([3, 0.5])
    # p = jnp.array([0.05*2, 0.45*2])
    # xis = jnp.array([11/2, 1/2])
    #p = jnp.array([0.005*2, 0.495*2])
    #xis = jnp.array([101/2, 1/2])

    def L1(x,y):
        return xis[0] * (x-1).transpose().dot(y-1)
    F1 = make_F(L1)

    def L2(x,y):
        return xis[1] * (x).transpose().dot(y)
    F2 = make_F(L2)

    def F(z):
        return p[0] * F1(z) + p[1] * F2(z)

    def Fhat(z, xi):
        if xi == xis[0]:
            return F1(z)
        else:
            return F2(z)

    def sample_xi(globalkey):
        globalkey, subkey = jax.random.split(globalkey)
        lips = jax.random.choice(subkey, 
            xis, 
            p=p, 
            shape=(args.batch_size,))
        return lips, globalkey

# Projection
if args.projection == 'linf':
    def proj(z):
        return jnp.clip(z,  -args.radius, args.radius)
elif args.projection is None:
    proj = lambda z: z

# Initialize
if args.d == 1:
    z0 = jnp.array([args.init])
else:
    globalkey, subkey = jax.random.split(globalkey)
    z0 = jax.random.uniform(subkey, (args.batch_size, args.d*2))
z = z0
zprev = z
zbar = z


#%%

if args.decrease == "linear":
    alpha = lambda t: args.alpha0/(t/args.decrease_factor+1)
elif args.decrease == "linear-slow":
    alpha = lambda t: 1.0 if t == 0 else args.alpha0/(t + args.alpha0)
elif args.decrease == "linear-slow-beta":
    alpha = lambda t: args.alpha0/args.beta0 if t == 0 else args.alpha0/(t + args.beta0)
elif args.decrease == "sqrt":
    alpha = lambda t: args.alpha0/jnp.sqrt(t/args.decrease_factor+1)

#beta = lambda t: args.beta0 / args.alpha0 * alpha(t-1)
beta = lambda t: alpha(t)


history = init_history(keys=[
    '|H-Hbar|', 
    '|Fbar|', 
    '|F|', 
    'znorm', 
    'zbarnorm'
], length=args.T)


def loop_body(t, state):
    history, z, zbar, zprev, globalkey = state 

    # Log prior to step to ensure first iterate is logged
    entry = {
        '|H-Hbar|': jnp.linalg.norm(z - args.gamma * F(z) - (zbar - args.gamma * F(zbar))),
        '|Fbar|': jnp.linalg.norm(F(zbar)), 
        '|F|': jnp.linalg.norm(F(z))**4,
        'znorm': jnp.linalg.norm(z),
        'zbarnorm': jnp.linalg.norm(zbar),
        #'step': t,
    }
    if args.d == 1:
        entry['x'] = z[:,0]
        entry['y'] = z[:,1]
    history = update_history(history, t, entry)

    if "BCSEG+" == args.method:
        xi, globalkey = sample_xi(globalkey)
        zbar = z - args.gamma * Fhat(z, xi) + (1 - beta(t)) * (zbar - zprev + args.gamma * Fhat(zprev, xi))
        zprev = z
        xibar, globalkey = sample_xi(globalkey)
        z = z - alpha(t) * args.gamma * Fhat(zbar, xibar)
    elif "P1BCSEG+" == args.method:
        xi, globalkey = sample_xi(globalkey)
        h = z - args.gamma * Fhat(z, xi) + (1 - beta(t)) * (zbar - zprev + args.gamma * Fhat(zprev, xi))
        zbar = proj(h)
        zprev = z
        xibar, globalkey = sample_xi(globalkey)
        z = z - alpha(t) * (h - zbar + args.gamma * Fhat(zbar, xibar))
    elif "SEG+" == args.method:
        xi, globalkey = sample_xi(globalkey)
        zbar = z - args.gamma * Fhat(z, xi)
        xibar, globalkey = sample_xi(globalkey)
        z = z - alpha(t) * args.gamma * Fhat(zbar, xibar)
    elif "P2SEG+" == args.method:
        xi, globalkey = sample_xi(globalkey)
        zbar = proj(z - args.gamma * Fhat(z, xi))
        xibar, globalkey = sample_xi(globalkey)
        z = proj(z - alpha(t) * args.gamma * Fhat(zbar, xibar))
    elif "P2SSEG+" == args.method:
        xi, globalkey = sample_xi(globalkey)
        zbar = proj(z - args.gamma * Fhat(z, xi))
        z = proj(z - alpha(t) * args.gamma * Fhat(zbar, xi))
    elif "P1SEG+" == args.method:
        xi, globalkey = sample_xi(globalkey)
        Hz = z - args.gamma * Fhat(z, xi)
        zbar = proj(Hz)
        xibar, globalkey = sample_xi(globalkey)
        Hzbar = zbar - args.gamma * Fhat(zbar, xibar)
        z = z + alpha(t) * (Hzbar - Hz)
    elif "P2SEG+" == args.method:
        xi, globalkey = sample_xi(globalkey)
        zbar = proj(z - args.gamma * Fhat(z, xi))
        xibar, globalkey = sample_xi(globalkey)
        z = proj(z - alpha(t) * args.gamma * Fhat(zbar, xibar))
    elif "SEG" == args.method:
        xi, globalkey = sample_xi(globalkey)
        zbar = z - alpha(t) * args.gamma * Fhat(z, xi)
        xibar, globalkey = sample_xi(globalkey)
        z = z - alpha(t) * args.gamma * Fhat(zbar, xibar)
    elif "P2SEG" == args.method:
        xi, globalkey = sample_xi(globalkey)
        zbar = proj(z - alpha(t) * args.gamma * Fhat(z, xi))
        xibar, globalkey = sample_xi(globalkey)
        z = proj(z - alpha(t) * args.gamma * Fhat(zbar, xibar))
    elif "EG+" == args.method:
        xi, globalkey = sample_xi(globalkey)
        zbar = z - args.gamma * Fhat(z, xi)
        xibar, globalkey = sample_xi(globalkey)
        z = z - args.alpha0 * args.gamma * Fhat(zbar, xibar)
    elif "P2EG+" == args.method:
        xi, globalkey = sample_xi(globalkey)
        zbar = proj(z - args.gamma * Fhat(z, xi))
        xibar, globalkey = sample_xi(globalkey)
        z = proj(z - args.alpha0 * args.gamma * Fhat(zbar, xibar))
    elif "detEG+" == args.method:
        zbar = z - args.gamma * F(z)
        z = z - args.alpha0 * args.gamma * F(zbar)
    # if (t % 1000 == 0):
    #     print("t=", t)
    
    return (history, z, zbar, zprev, globalkey)

init_state = (history, z, zbar, zprev, globalkey)
state = jax.lax.fori_loop(1, args.T, loop_body, init_state)
(history, z, zbar, zprev, globalkey) = state


for k in history.keys():
    fig, ax = plt.subplots(1, 1)
    ax.plot(history[k])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title(k)
    fig.tight_layout()
    fig.savefig(args.path(f"{k}.png"))
    plt.close(fig)
    if k in ["|F|", "|Fbar|", "|H-Hbar|"]:
        jnp.save(args.path(f"{k}.npy"), history[k])

print("z:", z)
# %%

def plot_vectorfield_with_trajectory(args, history, F):
    assert args.d == 1
    assert args.batch_size == 1
    N = 10
    M = 10
    bounds = [[-1,1], [-1,1]]
    x,y = jnp.meshgrid(
        jnp.linspace(bounds[0][0],bounds[0][1], N),
        jnp.linspace(bounds[1][0],bounds[1][1], M))
    x = x.flatten()[:,jnp.newaxis]
    y = y.flatten()[:,jnp.newaxis]
    Z = jnp.concatenate((x,y), axis=-1)
    FZ = F(Z)
    u, v = FZ[:, 0], FZ[:, 1]
    x = x.reshape(N,M)
    y = y.reshape(N,M)
    u = u.reshape(N,M)
    v = v.reshape(N,M)

    fig, ax = plt.subplots(1, 1)

    # Vectorfield
    # ax.streamplot(x,y,u,v, color="grey")

    # Trajectory
    ax.plot(history['x'], history['y'], color="red", label=args.method)
    ax.scatter(history['x'][0], history['y'][0], color="black")
    #ax.scatter(history['x'], history['y'], color="red")
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(bounds[0][0],bounds[0][1])
    ax.set_ylim(bounds[1][0],bounds[1][1])
    ax.legend(loc='lower left')
    return fig, ax

if args.plot_field:
    if args.d == 1:
        fig, ax = plot_vectorfield_with_trajectory(args, history, F)
        fig.tight_layout()
        fig.savefig(args.path(f"vectorfield.png"))
