import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../multinest")
import re
try:
    import getdist
    import getdist.plots
except ImportError:
    sys.path.append("../bhmf/lib/python3.10/site-packages")
    sys.path.append("../bhmf/lib/python3.11/site-packages")
    import getdist
    import getdist.plots
import numpy as np
from matplotlib.pyplot import *
import interface
import scipy
import warnings
warnings.filterwarnings("ignore")

rc('text', usetex=True)


parameters = {
    "plp+flat+trivial+trivial": [
        ("m_{min}\\ [M_{\\odot}]", "mmin"),
        ("\\delta_{m}\\ [M_{\\odot}]", "dm"),
        ("m_{max}\\ [M_{\\odot}]", "mmax"),
        ("\\mu_{m}\\ [M_{\\odot}]", "mu"),
        ("\\sigma_{m}\\ [M_{\\odot}]", "sigma"),
        ("\\alpha", "alpha"),
        ("\\lambda_{p}", "lp"),
    ],
    "plp+plp+trivial+trivial": [
        ("m_{min}\\ [M_{\\odot}]", "mmin"),
        ("\\delta_{m}\\ [M_{\\odot}]", "dm"),
        ("m_{max}\\ [M_{\\odot}]", "mmax"),
        ("\\mu_{m}\\ [M_{\\odot}]", "mu"),
        ("\\sigma_{m}\\ [M_{\\odot}]", "sigma"),
        ("\\alpha", "alpha"),
        ("\\lambda_{p}", "lp"),
        ("\\beta_{q}", "k")
    ],
    "plp+pow+trivial+trivial": [
        ("m_{min}\\ [M_{\\odot}]", "mmin"),
        ("\\delta_{m}\\ [M_{\\odot}]", "dm"),
        ("m_{max}\\ [M_{\\odot}]", "mmax"),
        ("\\mu_{m}\\ [M_{\\odot}]", "mu"),
        ("\\sigma_{m}\\ [M_{\\odot}]", "sigma"),
        ("\\alpha", "alpha"),
        ("\\lambda_{p}", "lp"),
        ("k", "k")
    ],
    'plp+flat+planck+trivial': [
        ("m_{min}\\ [M_{\\odot}]", "mmin"),
        ("\\delta_{m}\\ [M_{\\odot}]", "dm"),
        ("m_{max}\\ [M_{\\odot}]", "mmax"),
        ("\\mu_{m}\\ [M_{\\odot}]", "mu"),
        ("\\sigma_{m}\\ [M_{\\odot}]", "sigma"),
        ("\\alpha", "alpha"),
        ("\\lambda_{p}", "lp"),
        ("H_0", "h0"),
    ],
    "plp+pow+planck+trivial": [
        ("m_{min}\\ [M_{\\odot}]", "mmin"),
        ("\\delta_{m}\\ [M_{\\odot}]", "dm"),
        ("m_{max}\\ [M_{\\odot}]", "mmax"),
        ("\\mu_{m}\\ [M_{\\odot}]", "mu"),
        ("\\sigma_{m}\\ [M_{\\odot}]", "sigma"),
        ("\\alpha", "alpha"),
        ("\\lambda_{p}", "lp"),
        ("k", "k"),
        ("H_0", "h0"),
    ],
    "plp+plp+planck+trivial": [
        ("m_{min}\\ [M_{\\odot}]", "mmin"),
        ("\\delta_{m}\\ [M_{\\odot}]", "dm"),
        ("m_{max}\\ [M_{\\odot}]", "mmax"),
        ("\\mu_{m}\\ [M_{\\odot}]", "mu"),
        ("\\sigma_{m}\\ [M_{\\odot}]", "sigma"),
        ("\\alpha", "alpha"),
        ("\\lambda_{p}", "lp"),
        ("\\beta_q", "betaq"),
        ("H_0", "h0"),
    ],
    "plp+flat+trivial+beta": [
        ("m_{min}\\ [M_{\\odot}]", "mmin"),
        ("\\delta_{m}\\ [M_{\\odot}]", "dm"),
        ("m_{max}\\ [M_{\\odot}]", "mmax"),
        ("\\mu_{m}\\ [M_{\\odot}]", "mu"),
        ("\\sigma_{m}\\ [M_{\\odot}]", "sigma"),
        ("\\alpha", "alpha"),
        ("\\lambda_{p}", "lp"),
        ("\\alpha", "alpha1"),
        ("\\beta", "beta1"),
    ],
    "plp+plp+trivial+beta": [
        ("m_{min}\\ [M_{\\odot}]", "mmin"),
        ("\\delta_{m}\\ [M_{\\odot}]", "dm"),
        ("m_{max}\\ [M_{\\odot}]", "mmax"),
        ("\\mu_{m}\\ [M_{\\odot}]", "mu"),
        ("\\sigma_{m}\\ [M_{\\odot}]", "sigma"),
        ("\\alpha", "alpha"),
        ("\\lambda_{p}", "lp"),
        ("\\alpha", "alpha1"),
        ("\\beta_q", "betaq"),
        ("\\beta", "beta1"),
    ],
    "plp+pow+trivial+beta": [
        ("m_{min}\\ [M_{\\odot}]", "mmin"),
        ("\\delta_{m}\\ [M_{\\odot}]", "dm"),
        ("m_{max}\\ [M_{\\odot}]", "mmax"),
        ("\\mu_{m}\\ [M_{\\odot}]", "mu"),
        ("\\sigma_{m}\\ [M_{\\odot}]", "sigma"),
        ("\\alpha", "alpha"),
        ("\\lambda_{p}", "lp"),
        ("k", "k"),
        ("\\alpha", "alpha1"),
        ("\\beta", "beta1"),
    ],
    "ppisn+flat+trivial+trivial": [
        ("m_{min}\\ [M_{\\odot}]", "mmin"),
        ("\\delta_{m}\\ [M_{\\odot}]", "dm"),
        ("m_{gap}\\ [M_{\\odot}]", "mgap"),
        ("a", "a"),
        ("b", "b"),
        ("d", "d"),
        ("\\log_{10}\\lambda_{21}", "lam21"),
        ("\\log_{10}\\lambda_{12}", "lam12"),
    ],
    "ppisn+trivial+trivial": [
        ("m_{min}\\ [M_{\\odot}]", "mmin"),
        ("\\delta_{m}\\ [M_{\\odot}]", "dm"),
        ("m_{gap}\\ [M_{\\odot}]", "mgap"),
        ("a", "a"),
        ("b", "b"),
        ("d", "d"),
        ("\\log_{10}\\lambda_{21}", "lam21"),
        ("\\log_{10}\\lambda_{12}", "lam12"),
        ("\\beta_{q}", "bq0"),
    ],
    "ppisn2P+trivial+trivial": [
        ("m_{min}\\ [M_{\\odot}]", "mmin"),
        ("\\delta_{m}\\ [M_{\\odot}]", "dm"),
        ("m_{gap}\\ [M_{\\odot}]", "mgap"),
        ("a", "a"),
        ("b", "b"),
        ("d", "d"),
        ("\\log_{10}\\lambda_{21}", "lam21"),
        ("\\log_{10}\\lambda_{12}", "lam12"),
        ("\\beta_{q}^{(0)}", "bq0"),
        ("\\beta_{q}^{(1)}", "bq1"),
    ],
    "ppisn+planck+trivial": [
        ("m_{min}\\ [M_{\\odot}]", "mmin"),
        ("\\delta_{m}\\ [M_{\\odot}]", "dm"),
        ("m_{gap}\\ [M_{\\odot}]", "mgap"),
        ("a", "a"),
        ("b", "b"),
        ("d", "d"),
        ("\\log_{10}\\lambda_{21}", "lam21"),
        ("\\log_{10}\\lambda_{12}", "lam12"),
        ("\\beta_{q}", "bq0"),
        ("H_0", "h0"),
    ],
    "ppisn+trivial+beta": [
        ("m_{min}\\ [M_{\\odot}]", "mmin"),
        ("\\delta_{m}\\ [M_{\\odot}]", "dm"),
        ("m_{gap}\\ [M_{\\odot}]", "mgap"),
        ("a", "a"),
        ("b", "b"),
        ("d", "d"),
        ("\\log_{10}\\lambda_{21}", "lam21"),
        ("\\log_{10}\\lambda_{12}", "lam12"),
        ("\\beta_{q}", "bq0"),
        ("\\alpha_1", "alpha1"),
        ("\\beta_1", "beta1"),
        ("\\alpha_2", "alpha2"),
        ("\\beta_2", "beta2"),
    ],
    "ppisn+trivial+beta-turnon": [
        ("m_{gap}\\ [M_{\\odot}]", "mgap"),
        ("a", "a"),
        ("b", "b"),
        ("d", "d"),
        ("\\log_{10}\\lambda_{21}", "lam21"),
        ("\\log_{10}\\lambda_{12}", "lam12"),
        ("\\beta_{q}", "bq0"),
        ("\\alpha_1", "alpha1"),
        ("\\beta_1", "beta1"),
        ("\\alpha_2", "alpha2"),
        ("\\beta_2", "beta2"),
    ],
    "ppisn+trivial+1beta-turnon": [
        ("m_{gap}\\ [M_{\\odot}]", "mgap"),
        ("a", "a"),
        ("b", "b"),
        ("d", "d"),
        ("\\log_{10}\\lambda_{21}", "lam21"),
        ("\\log_{10}\\lambda_{12}", "lam12"),
        ("\\beta_{q}", "bq0"),
        ("\\alpha", "alpha1"),
        ("\\beta", "beta1"),
    ],
    "ppisn+trivial+gauss-turnon": [
        ("m_{gap}\\ [M_{\\odot}]", "mgap"),
        ("a", "a"),
        ("b", "b"),
        ("d", "d"),
        ("\\log_{10}\\lambda_{21}", "lam21"),
        ("\\log_{10}\\lambda_{12}", "lam12"),
        ("\\beta_{q}", "bq0"),
        ("\\mu_1", "alpha1"),
        ("\\sigma_1", "beta1"),
        ("\\mu_2", "alpha2"),
        ("\\sigma_2", "beta2"),
    ],
    "ppisn+trivial+1gauss-turnon": [
        ("m_{gap}\\ [M_{\\odot}]", "mgap"),
        ("a", "a"),
        ("b", "b"),
        ("d", "d"),
        ("\\log_{10}\\lambda_{21}", "lam21"),
        ("\\log_{10}\\lambda_{12}", "lam12"),
        ("\\beta_{q}", "bq0"),
        ("\\mu", "alpha1"),
        ("\\sigma", "beta1"),
    ],

    "ppisn+trivial+beta-mass": [
        ("\\alpha_1", "alpha1"),
        ("\\beta_1", "beta1"),
        ("\\alpha_2", "alpha2"),
        ("\\beta_2", "beta2"),
    ],
    "ppisn+trivial+gauss-mass": [
        ("\\mu_1", "alpha1"),
        ("\\sigma_1", "beta1"),
        ("\\mu_2", "alpha2"),
        ("\\sigma_2", "beta2"),
    ],
}

def mergenumbers(values):
    weight = sum(1 / values[:, 1] ** 2)
    value = sum(values[:, 0] / values[:, 1] ** 2 / weight)
    return np.array([value, np.sqrt(1 / weight)])

def parseInfo(root):
    with open(root + ".timing") as fp:
        txt = fp.read()
    model = re.findall("model: (.*)", txt)[0]
    np = len(parameters[model])
    lower = [
        float(i)
        for i in re.findall(
            r"prior range, lower bounds" + r" *([\d\.-]+)"*np,
            txt
        )[0]
    ]
    upper = [
        float(i)
        for i in re.findall(
            r"Prior range, upper bounds" + r" *([\d\.-]+)"*np,
            txt
        )[0]
    ]
    return model, list(zip(lower, upper))



def loadMC(root, model="plp+pow+trivial+trivial"):
    dat = np.loadtxt(root + '.txt')
    samples = getdist.MCSamples(
        samples=dat[:, 2:],
        weights=dat[:, 0],
        names=[j for _, j in parameters[model]],
        labels=[j for j, _ in parameters[model]]
    )
    samples.dat = dat
    return samples


def triangle_plot(samples, lim, model="plp+pow+trivial+trivial"):
    g = getdist.plots.get_subplot_plotter()
    g.triangle_plot(
        samples,
        param_limits={n: l for ((_, n), l) in zip(parameters[model], lim)},
        filled=True
    )
    return gcf()


def best_fit_band(samples, model, n=500, CL=.95, prescale=1):
    dcl = 100 * (1-CL)/2
    paras = np.random.multivariate_normal(best_fit(samples), samples.cov(), size=n)
    x = np.linspace(0, 100, n)
    y = np.array([interface.pyinterface(model, 'm1', i,x) for i in paras])
    y = y[~np.any(y<0,axis=1)]

    print(np.sqrt(np.diag(samples.cov())))
    print(best_fit(samples))

    return x, *np.percentile(prescale * y, [dcl, 100-dcl],axis=0)


def best_fit_band_norms(samples, model, n=500, npara=200, CL=.95, prescale=1):
    m1 = np.linspace(0, 120, n)
    m2 = np.linspace(0, 120, n)
    A, B = np.meshgrid(m1,m2)
    x=np.concatenate((A,B)).flatten()

    def avg(para):
        yy = interface.pyinterface(model, 'm1m2', para,x)
        return np.mean(np.reshape(yy[:n*n],(n,n)), axis=0)

    dcl = 100 * (1-CL)/2
    paras = np.random.multivariate_normal(best_fit(samples), samples.cov(), size=npara)

    y = np.array([avg(i) for i in paras])
    y = y[~np.any(y<0,axis=1)]

    ybf = avg(best_fit(samples))

    return m1, *np.percentile(prescale * y, [dcl, 100-dcl],axis=0), prescale * ybf


def plot_bestfit_m1(samples, model="plp+pow+trivial+trivial", fig=None, prescale=1, col=None, label=None):
    h = 1e-6
    n = 1000

    bestfit = best_fit(samples)

    x = np.linspace(0, 100, n)
    y = prescale * interface.pyinterface(model, 'm1', bestfit, x)

    plots = parameters[model]

    J = np.zeros((n, len(plots)))

    mat = (1 + h * np.identity(len(plots))) * bestfit
    for i in range(len(plots)):
        yy = prescale * interface.pyinterface(model, 'm1', mat[i], x)
        J[:, i] = (yy - y) / h / bestfit[i]

    tot = J @ samples.cov() @ J.T
    err = np.sqrt(tot.diagonal())

    if fig == None:
        fig = figure()
    if label == None:
        label = model

    if col:
        plot(x, y, color=col, label=label)
        fill_between(x, y-err/2, y+err/2, color=col, alpha=0.2)
    else:
        plot(x, y, label=label)
        fill_between(x, y-err/2, y+err/2, alpha=0.2)
    yscale('log')

    ylabel(r"${\rm d}R/{\rm d}m_1$")
    xlabel("$m_1/M_\odot$")
    xlim(0,65)
    ylim(3e-4, 1)
    return fig


def best_fit(samples):
    def best_fit1(para):
        dens = samples.get1DDensity(para).normalize()

        lower, upper = dens.bounds()
        avg, _ = scipy.integrate.quad(
            lambda x: x * dens.Prob(x),
            lower, upper
        )
        return avg

    return np.array([
        best_fit1(name)
        for name in samples.getMargeStats().names
    ])


def central_value(dens):
    lower, upper = dens.bounds()
    avg, _ = scipy.integrate.quad(
        lambda x: x * dens.Prob(x),
        lower, upper
    )
    lvar, _ = scipy.integrate.quad(
        lambda x: (x-avg)**2 * dens.Prob(x),
        lower, avg
    )
    uvar, _ = scipy.integrate.quad(
        lambda x: (x-avg)**2 * dens.Prob(x),
        avg, upper
    )
    return avg, lvar, uvar


def central_values(samples, model):
    for _, para in parameters[model]:
        avg, lvar, uvar = central_value(samples.get1DDensity(para).normalize())
        print(f"{para:8s} {avg:7.2f} +- {np.sqrt(lvar+uvar):7.2f}   -{np.sqrt(2*lvar):7.2f}+{np.sqrt(2*uvar):7.2f}")


def multiintersect(lists):
    if len(lists) == 0:
        return []
    inter = set(lists[0])
    for i in lists[1:]:
        inter = inter.intersection(set(i))
    return list(inter)


def main(roots):
    figc = figure()
    l = []
    logz = {}
    all_samples = {}

    for root in roots:
        print(root)
        model, lim = parseInfo(root)

        samples = loadMC(root, model)
        all_samples[model] = samples

        fig = triangle_plot(samples, lim, model)
        fig.savefig(root + model + "triangle.pdf")

        fig = plot_bestfit_m1(samples, model)
        fig.savefig(root + model + "fitm1.pdf")

        plot_bestfit_m1(samples, model, figc)
        l.append(model)
        l.append('$1\sigma$')

        central_values(samples, model)
        with open(root+'stats.dat') as fp:
            s = fp.read()
        model = root
        logz[model] = [float(i) for i in re.findall(
            '([\d\.E\+-]+) *\+/- *([\d\.E\+-]+)',
            s.splitlines()[0]
        )[0]]

    legend(l)

    for i, ki in enumerate(logz.keys()):
        for j, kj in enumerate(logz.keys()):
            if i < j:
                bf = (
                    logz[ki][0] - logz[kj][0],
                    np.sqrt(logz[ki][1]**2 + logz[kj][1]**2)
                )
                if bf[0] > 0:
                    kf = kj
                else:
                    kf = ki
                print(f"BF {ki} and {kj} = {bf[0]} +- {bf[1]} favours {kf}")

    for param in multiintersect([i.paramNames.list() for i in all_samples.values()]):
        fig = figure()
        for m, samples in all_samples.items():
            density = samples.get1DDensityGridData(param)
            density.normalize()
            plot(density.x, density.P, label=m)
            xlabel("$" + {j:i for i,j in parameters[m]}[param] + "$")
        legend()
        fig.savefig(f"cmp-{param}.pdf")

    return figc


if __name__ == "__main__":
    main(sys.argv[1:])
