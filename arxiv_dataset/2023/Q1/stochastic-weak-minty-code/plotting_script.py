#%% Bilinear 1/sqrt(k)
import plotting
import numpy as np
from plotting import create_plot

experiments={
    "P$_2$SEG+ $(\gamma=0.5)$": "P2SEG+|bilinear|offset=0.9|projection=linf|alpha0=0.055|noise=1.0|gamma=0.5|decrease=sqrt-100|T=500000",
    "P$_1$SEG+ $(\gamma=0.5)$": "P1SEG+|bilinear|offset=0.9|projection=linf|alpha0=0.055|noise=1.0|gamma=0.5|decrease=sqrt-100|T=500000",
    "P$_1$SEG+ $(\gamma=0.1)$": "P1SEG+|bilinear|offset=0.9|projection=linf|alpha0=0.055|noise=1.0|gamma=0.1|decrease=sqrt-100|T=500000",
    "BC-PSEG+ $(\gamma=0.5)$": "P1BCSEG+|bilinear|offset=0.9|projection=linf|alpha0=0.055|noise=1.0|gamma=0.5|decrease=sqrt-100|T=500000",
}

metrics = {
    '|F|': '$\|Fz^k\|$',
    '|Fbar|': '$\|F\\bar z^k\|$',
}

for name,label in metrics.items():
    fig, ax = create_plot(
        experiments=experiments,
        key=name,
        take_every=5,
        transform=lambda x: np.abs(x),
        plot_kwargs=dict(markevery=0.5,alpha=0.5),
    )
    #ax.set_xlim((0,10000))
    ax.set_ylabel(label)
    ax.set_xlabel("Iteration $k$")
    ax.set_yscale('log')
    ax.set_xscale('log')
    fig.savefig(f"figs/monotone_{name}.png", dpi=300)

#%% Bilinear 1/k
import plotting
import numpy as np
from plotting import create_plot

experiments={
    "P$_2$SEG+ $(\gamma=0.5)$": "P2SEG+|bilinear|offset=0.9|projection=linf|alpha0=0.055|noise=1.0|gamma=0.5|decrease=linear-100|T=500000",
    "P$_1$SEG+ $(\gamma=0.5)$": "P1SEG+|bilinear|offset=0.9|projection=linf|alpha0=0.055|noise=1.0|gamma=0.5|decrease=linear-100|T=500000",
    "P$_1$SEG+ $(\gamma=0.1)$": "P1SEG+|bilinear|offset=0.9|projection=linf|alpha0=0.055|noise=1.0|gamma=0.1|decrease=linear-1000|T=500000",
    "BC-PSEG+ $(\gamma=0.5)$": "P1BCSEG+|bilinear|offset=0.9|projection=linf|alpha0=0.055|noise=1.0|gamma=0.5|decrease=linear-100|T=500000",
}

metrics = {
    '|F|': '$\|Fz^k\|$',
    '|Fbar|': '$\|F\\bar z^k\|$',
}

for name,label in metrics.items():
    fig, ax = create_plot(
        experiments=experiments,
        key=name,
        take_every=5,
        transform=lambda x: np.abs(x),
        plot_kwargs=dict(markevery=0.5,alpha=0.5),
    )
    #ax.set_xlim((0,10000))
    ax.set_ylabel(label)
    ax.set_xlabel("Iteration $k$")
    ax.set_yscale('log')
    ax.set_xscale('log')
    fig.savefig(f"figs/monotone_almostsure_{name}.png", dpi=300)

#%% Quadratic 1/sqrt(k)
import plotting
import numpy as np
from plotting import create_plot

experiments={
    "SF-EG+": "EG+|rho=-0.1|alpha0=0.055|beta0=1|noise=0.1|gamma=0.5|decrease=sqrt-100|T=500000",
    "SEG": "SEG|rho=-0.1|alpha0=0.055|beta0=1|noise=0.1|gamma=0.5|decrease=sqrt-100|T=10000",
    "BC-SEG+": "BCSEG+|rho=-0.1|alpha0=0.055|beta0=1|noise=0.1|gamma=0.5|decrease=sqrt-100|T=500000",
}

metrics = {
    '|F|': '$\|Fz^k\|$',
    '|Fbar|': '$\|F\\bar z^k\|$',
}

for name,label in metrics.items():
    fig, ax = create_plot(
        experiments=experiments,
        key=name,
        take_every=5,
        transform=lambda x: np.abs(x),
        plot_kwargs=dict(markevery=0.5,alpha=0.5),
    )
    #ax.set_xlim((0,10000))
    ax.set_ylabel(label)
    ax.set_xlabel("Iteration $k$")
    ax.set_yscale('log')
    ax.set_xscale('log')
    fig.savefig(f"figs/Quadratic_{name}.png", dpi=300)

#%% Quadratic 1/k
import plotting
import numpy as np
from plotting import create_plot

experiments={
    "SF-EG+": "EG+|rho=-0.1|alpha0=0.055|noise=0.1|gamma=0.5|decrease=linear-100|T=500000",
    "SEG": "SEG|rho=-0.1|alpha0=0.055|noise=0.1|gamma=0.5|decrease=linear-100|T=500000",
    "BC-SEG+": "BCSEG+|rho=-0.1|alpha0=0.055|noise=0.1|gamma=0.5|decrease=linear-100|T=500000",
}

metrics = {
    '|F|': '$\|Fz^k\|$',
    '|Fbar|': '$\|F\\bar z^k\|$',
}

for name,label in metrics.items():
    fig, ax = create_plot(
        experiments=experiments,
        key=name,
        take_every=5,
        transform=lambda x: np.abs(x),
        plot_kwargs=dict(markevery=0.5,alpha=0.5),
    )
    #ax.set_xlim((0,10000))
    ax.set_ylabel(label)
    ax.set_xlabel("Iteration $k$")
    ax.set_yscale('log')
    ax.set_xscale('log')
    fig.savefig(f"figs/Quadratic_almostsure_{name}.png", dpi=300)


#%% GlobalForsaken 1/sqrt(k)
import plotting
import numpy as np
from plotting import create_plot

experiments={
    "SF-PEG+": "GlobalForsaken|P2EG+|alpha0=0.055|beta0=1|noise=0.1|gamma=0.33|decrease=sqrt-100|T=500000",
    "PSEG": "GlobalForsaken|P2SEG|alpha0=0.055|beta0=1|noise=0.1|gamma=0.33|decrease=sqrt-100|T=500000",
    "BC-PSEG+": "GlobalForsaken|P1BCSEG+|alpha0=0.055|beta0=1|noise=0.1|gamma=0.33|decrease=sqrt-100|T=500000",
}

metrics = {
    '|Fbar|': '$\|F\\bar z^k\|$',
}

for name,label in metrics.items():
    fig, ax = create_plot(
        experiments=experiments,
        key=name,
        take_every=5,
        transform=lambda x: np.abs(x),
        plot_kwargs=dict(markevery=0.5,alpha=0.5),
    )
    #ax.set_xlim((0,10000))
    ax.set_ylabel(label)
    ax.set_xlabel("Iteration $k$")
    ax.set_yscale('log')
    ax.set_xscale('log')
    fig.savefig(f"figs/GlobalForsaken_{name}.png", dpi=300)

#%% GlobalForsaken 1/k
import plotting
import numpy as np
from plotting import create_plot

experiments={
    "SF-PEG+": "GlobalForsaken|P2EG+|alpha0=0.055|noise=0.1|gamma=0.33|decrease=linear-100|T=500000",
    "PSEG": "GlobalForsaken|P2SEG|alpha0=0.055|noise=0.1|gamma=0.33|decrease=linear-100|T=500000",
    "BC-PSEG+": "GlobalForsaken|P1BCSEG+|alpha0=0.055|noise=0.1|gamma=0.33|decrease=linear-100|T=500000",
}

metrics = {
    '|Fbar|': '$\|F\\bar z^k\|$',
}

for name,label in metrics.items():
    fig, ax = create_plot(
        experiments=experiments,
        key=name,
        take_every=5,
        transform=lambda x: np.abs(x),
        plot_kwargs=dict(markevery=0.5,alpha=0.5),
    )
    #ax.set_xlim((0,10000))
    ax.set_ylabel(label)
    ax.set_xlabel("Iteration $k$")
    ax.set_yscale('log')
    ax.set_xscale('log')
    fig.savefig(f"figs/GlobalForsaken__almostsure_{name}.png", dpi=300)


#%% Quadratic constrained 1/sqrt(k)
import plotting
import numpy as np
from plotting import create_plot

experiments={
    "P$_1$SEG+ $(\gamma=0.5)$": "P1SEG+|rho=-0.1|offset=0.9|projection=linf|alpha0=0.055|beta0=1|noise=0.1|gamma=0.5|decrease=sqrt-100|T=500000",
    "P$_1$SEG+  $(\gamma=0.1)$": "P1SEG+|rho=-0.1|offset=0.9|projection=linf|alpha0=0.055|beta0=1|noise=0.1|gamma=0.1|decrease=sqrt-100|T=500000",
    "P$_1$SEG+  $(\gamma=0.01)$": "P1SEG+|rho=-0.1|offset=0.9|projection=linf|alpha0=0.055|beta0=1|noise=0.1|gamma=0.01|decrease=sqrt-100|T=500000",
    "BC-PSEG+  $(\gamma=0.5)$": "P1BCSEG+|rho=-0.1|offset=0.9|projection=linf|alpha0=0.055|beta0=1|noise=0.1|gamma=0.5|decrease=sqrt-100|T=500000",
}

metrics = {
    '|Fbar|': '$\|F\\bar z^k\|$',
}

for name,label in metrics.items():
    fig, ax = create_plot(
        experiments=experiments,
        key=name,
        take_every=5,
        transform=lambda x: np.abs(x),
        plot_kwargs=dict(markevery=0.5,alpha=0.7),
    )
    ax.set_xlim((10,500000))
    ax.set_ylabel(label)
    ax.set_xlabel("Iteration $k$")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_aspect(0.5)
    fig.savefig(f"figs/QuadraticConstrained_{name}.png", dpi=300)

#%% Quadratic constrained
import plotting
import numpy as np
from plotting import create_plot

experiments={
    "P$_1$SEG+ $(\gamma=0.5)$": "P1SEG+|rho=-0.1|offset=0.9|projection=linf|alpha0=0.055|beta0=1|noise=0.1|gamma=0.5|decrease=linear-1000|T=500000",
    "P$_1$SEG+  $(\gamma=0.1)$": "P1SEG+|rho=-0.1|offset=0.9|projection=linf|alpha0=0.055|beta0=1|noise=0.1|gamma=0.1|decrease=linear-1000|T=500000",
    "P$_1$SEG+  $(\gamma=0.01)$": "P1SEG+|rho=-0.1|offset=0.9|projection=linf|alpha0=0.055|beta0=1|noise=0.1|gamma=0.01|decrease=linear-5000|T=500000",
    "BC-PSEG+  $(\gamma=0.5)$": "P1BCSEG+|rho=-0.1|offset=0.9|projection=linf|alpha0=0.055|beta0=1|noise=0.1|gamma=0.5|decrease=linear-1000|T=500000",
}

metrics = {
    '|Fbar|': '$\|F\\bar z^k\|$',
}

for name,label in metrics.items():
    fig, ax = create_plot(
        experiments=experiments,
        key=name,
        take_every=5,
        transform=lambda x: np.abs(x),
        plot_kwargs=dict(markevery=0.5,alpha=0.7),
    )
    ax.set_xlim((10,500000))
    ax.set_ylabel(label)
    ax.set_xlabel("Iteration $k$")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_aspect(0.5)
    fig.savefig(f"figs/QuadraticConstrained_almostsure_{name}.png", dpi=300)
