import numpy as np, pylab as plt, time
from tqdm.auto import tqdm
from scipy.stats import skewnorm
from scipy import sparse
from tqdm.auto import tqdm
import mendeleev as pt
from otlm.models import OTLM

def read_from_pickle(filename):
    import xopen, pickle
    with xopen.xopen(filename, 'rb') as f:
        return pickle.load(f)   

def astr(x, precision=4):
    return np.array2string(x, precision=precision, max_line_width=100000, formatter={'float_kind': lambda x: f'{x:.4f}'})

def gauss(x, a, μ, σ): 
    y = a*np.exp(-(x-μ)**2/σ**2)
    return y

def gauss_normed(x, a, μ, σ):
    y = gauss(x, 1, μ, σ)
    y = a*y/np.sum(y)
    return y

def gauss_skew_normed(x, a, μ, σ, α):

    loc = μ - σ * np.sqrt(2/np.pi) * (α/np.sqrt(1+α**2))
    y = skewnorm.pdf(x, α, loc=loc, scale=σ)
    y = a*y/np.sum(y)
    return y

def get_demo_dataset_ridge():

    np.random.seed(43)

    σ = 0.2
    x = np.linspace(0, 3, 100)
    μ = [0.5, 1.5, 2.5]
    yt = gauss_skew_normed(x, a=1,    μ=μ[0]-σ*0.4, σ=σ*1.1, α=2) \
       + gauss_skew_normed(x, a=0.08, μ=μ[1]-σ*0.2, σ=σ*0.8, α=-3) \
       + gauss_skew_normed(x, a=0.02, μ=μ[2]-σ*0.2, σ=σ*0.8, α=0) \

    yt = yt + np.abs(np.random.normal(size=len(yt), scale=0.002))
    b0 = gauss_normed(x, a=1., μ=μ[0], σ=σ) 
    b1 = gauss_normed(x, a=1., μ=μ[1], σ=σ)
    b2 = gauss_normed(x, a=1., μ=μ[2], σ=σ)
    X = np.vstack([b for b in [b0, b1, b2]]).T

    C = np.abs(x[:,np.newaxis] - x[np.newaxis,:]) 
    C = C * 0.01
   
    return x, yt, X, C


def get_demo_dataset_lasso():

    np.random.seed(44)

    σ = 0.2
    x = np.linspace(0,3,100)
    μ = [0.5, 1.5, 2.5]
    yt = gauss_skew_normed(x, a=1,    μ=μ[0]-σ*0.4, σ=σ*1.1, α=2) \
       + gauss_skew_normed(x, a=0.08, μ=μ[1]-σ*0.2, σ=σ*0.8, α=-3) \
       + gauss_skew_normed(x, a=0.02, μ=μ[2]-σ*0.2, σ=σ*0.8, α=0) \

    yt = yt + np.abs(np.random.normal(size=len(yt), scale=0.001))
    b0 = gauss_normed(x, a=1., μ=μ[0], σ=σ) 
    b1 = gauss_normed(x, a=1., μ=μ[1], σ=σ)
    b2 = gauss_normed(x, a=1., μ=μ[2], σ=σ)
    X = np.vstack([b for b in [b0, b1, b2]]).T

    C = np.abs(x[:,np.newaxis] - x[np.newaxis,:]) 
    C = C * 0.001
   
    return x, yt, X, C

def get_demo_dataset_tv():

    np.random.seed(43)

    σ = 0.2
    x = np.linspace(0,3,100)
    μ = [0.5, 1.5, 2.5]
    yt = gauss_skew_normed(x, a=1,    μ=μ[0]-σ*0.4, σ=σ*1.1, α=2) \
       + gauss_skew_normed(x, a=0.08, μ=μ[1]-σ*0.2, σ=σ*0.8, α=-3) \
       + gauss_skew_normed(x, a=0.02, μ=μ[2]-σ*0.2, σ=σ*0.8, α=0) \

    b0 = gauss_normed(x, a=1., μ=μ[0], σ=σ) 
    b1 = gauss_normed(x, a=1., μ=μ[1], σ=σ)
    b2 = gauss_normed(x, a=1., μ=μ[2], σ=σ)
    X = np.vstack([b for b in [b0, b1, b2]]).T

    C = np.abs(x[:,np.newaxis] - x[np.newaxis,:]) 
    C = C * 0.01
   
    return x, yt, X, C


def get_demo_dataset_poisson():

    np.random.seed(43)
    amp=1000

    σ = 0.2
    x = np.linspace(0,3,100)
    μ = [0.5, 1.5, 2.5]
    yt = gauss_skew_normed(x, a=amp*1, μ=μ[0]-σ*0.4, σ=σ*1.1, α=2) \
         + gauss_skew_normed(x, a=amp*0.08, μ=μ[1]-σ*0.2, σ=σ*0.8, α=-3) \
         + gauss_skew_normed(x, a=amp*0.02, μ=μ[2]-σ*0.2, σ=σ*0.8, α=0) \

    yt = np.random.poisson(lam=yt) + 1e-2
    b0 = gauss_normed(x, a=amp, μ=μ[0], σ=σ) 
    b1 = gauss_normed(x, a=amp, μ=μ[1], σ=σ)
    b2 = gauss_normed(x, a=amp, μ=μ[2], σ=σ)
    X = np.vstack([b for b in [b0, b1, b2]]).T

    C = np.abs(x[:,np.newaxis] - x[np.newaxis,:]) 
    C = C * 10
   
    return x, yt, X, C


def plot_scenarios_3panels(x, yt, X, otlm, lm, ylim=None, min_plan=1e-6, log_plan=True):

    nx, ny = 1, 1; figsize=2.5; 
    fig1, ax = plt.subplots(nx, ny, figsize=(ny * figsize * 1.5 , nx * figsize), squeeze=False); 
    axl=ax.ravel(); 
    fig1.set_label(' ')

    ls = ['-.', '--', ':', '-.', '--', ':', '-.', '--', ':']
    cmap = plt.cm.gist_heat_r
    alpha_thick = 0.9

    yp_lm = lm.predict(X)
    axl[0].plot(x, yt, label='Target', c='hotpink', lw=3, alpha=alpha_thick)
        
    for i, b in enumerate(X.T):
        axl[0].plot(x, np.ma.array(b*0.5, mask=b<1e-5), label=f'Basis {i}', ls=ls[i], c='black');

    axl[0].plot(x, yp_lm, label='Fitted TV-LM', color='black', lw=2, alpha=1)
    axl[0].legend(framealpha=0)
    axl[0].set(xlabel=r'$x$', ylim=ylim)

    nx, ny = 1, 1; figsize=2.5; 
    fig2, ax = plt.subplots(nx, ny, figsize=(ny * figsize * 1.5 , nx * figsize), squeeze=False); 
    axl=ax.ravel(); 
    fig2.set_label(' ')
    
    yp_src = otlm.predict(X)
    yp_tgt = otlm.predict(X, target=True)
    yp_otlm = np.dot(X, otlm.w)
    axl[0].plot(x, yt, c='hotpink', lw=4, label='Target', zorder=1, alpha=alpha_thick)
    axl[0].plot(x, yp_otlm, label='Fitted OTLM', color='dodgerblue', lw=4, zorder=3, alpha=alpha_thick)
    axl[0].plot(x, yp_tgt, label='Transported', color='red', lw=1, zorder=2, alpha=1)
    axl[0].plot(x, yp_src, label='Source', color='blue', lw=1, zorder=4, alpha=alpha_thick)
    axl[0].set(xlabel=r'$x$', ylim=ylim)
    axl[0].legend(framealpha=0)

    nx, ny = 1, 1; figsize=2.5; 
    fig3, ax = plt.subplots(nx, ny, figsize=(ny * figsize * 1.5 , nx * figsize), squeeze=False); 
    axl=ax.ravel(); 
    fig3.set_label(' ')

    plan = np.ma.array(otlm.Q.T, mask=otlm.Q.T<min_plan)
    plan = np.log(plan) if log_plan else plan
    axl[0].pcolormesh(x, x, plan, cmap=cmap)
    axl[0].grid(True, ls='--')
    axl[0].set_xticks(np.linspace(0, max(x), 6))
    axl[0].set_yticks(np.linspace(0, max(x), 6))
    axl[0].set(xlim=(min(x), max(x)), ylim=(min(x), max(x)))
    axl[0].set(xlabel=r'$x$ source', ylabel=r'$x$ target')
    if log_plan:
        axl[0].annotate(xy=(0.923, 0.22), text=r'ln$Q$', xycoords='subfigure fraction')
    axl[0].set_aspect('equal', 'box')

    return fig1, fig2, fig3


def plot_multiple_scenarios(x, list_otlms, list_lms,list_labels, list_ylim, list_yt, list_X):
   
    alpha_thick = 0.9
    figs = []

    for i, (otlm, lm, label, ylim, yt, X) in enumerate(zip(list_otlms, list_lms, list_labels, list_ylim, list_yt, list_X)):

        nx, ny = 1, 1; 
        figsize=2.5; 
        fig, ax = plt.subplots(nx, ny, figsize=(ny * figsize * 1.5 , nx * figsize), squeeze=False); 
        axl=ax.ravel(); 
        ax = axl[0]      

        yp_lm = lm.predict(X)   
        yp_tgt = otlm.predict(X, target=True)
        yp_otlm = np.dot(X, otlm.w)
        ax.plot(x, yt, c='hotpink', lw=5, zorder=2, alpha=alpha_thick)
        ax.plot(x, yp_lm, label=label[0], color='black', lw=2, zorder=4, alpha=1)
        ax.plot(x, yp_otlm, label=label[1], color='dodgerblue', lw=5, zorder=3, alpha=alpha_thick)
        ax.plot(x, yp_tgt, label=label[1] + '\n Transported', color='red', lw=1, zorder=2)
        ax.set(xlabel=r'$x$', ylim=ylim)
        ax.legend(loc='upper right')
        figs.append(fig)
        

    return figs



def get_scaling_dataset(n_samples=100, n_features=10, sparse_construction=False):

    np.random.seed(43)

    σ_mean = 2
    xμ = np.linspace(0, n_samples, n_features)
    x = np.linspace(0, n_samples, n_samples)
    a = np.abs(np.random.normal(loc=1, scale=0.2, size=n_features)) + 0.01
    μ = np.random.uniform(low=0, high=n_samples, size=n_features)
    σ = np.abs(np.random.normal(loc=σ_mean, scale=0.2, size=n_features))
    α = np.random.normal(loc=0, scale=2, size=n_features)
    
    # generate target
    yt = np.zeros(n_samples)
    for i in tqdm(range(n_features), desc='Generating target'):
        yt += gauss_skew_normed(x, a=a[i], μ=μ[i], σ=σ[i], α=α[i])

    # add noise
    yt = yt + np.abs(np.random.normal(size=len(yt), scale=0.002))

    # generate source
    cols = []
    for i in tqdm(range(n_features), desc='Generating features'):
        col = gauss_normed(x, a=1, μ=xμ[i], σ=σ_mean)
        col = sparse.csc_array(col)
        cols.append(col)
    X = sparse.vstack(cols).T
    X = sparse.csr_array(X)

    # generate cost
    rows = []
    for i in tqdm(range(n_samples), desc='Generating cost'):
        row = np.abs(x[i] - x)
        if sparse_construction:
            row[row>5*σ_mean] = 0 # 5 sigma
        row = row * 0.01
        row = sparse.csr_array(row)
        rows.append(row)
    
    C = sparse.vstack(rows)
    C = C.tocsr()

    if not sparse_construction:
        C = np.array(C.todense())
        X = np.array(X.todense())
        
    return x, yt, X, C

def run_scaling_experiment(n_samples, n_features_per_sample, variant='l1l1', ε=[0.01], α=0.1, λ=1, sparse_construction=False):

    from otlm.solvers import OTLM_L1_LM, OTLM_L1L1_LM, OTLM_L2L2_QP

    from otlm.models import OTLM
    
    times = np.empty(len(n_samples), dtype=object)
    weights = np.empty(len(n_samples), dtype=object)
    otlms = np.empty(len(n_samples), dtype=object)

    print(f'OTLM scaling experiment started {variant}')


    for i, n_samples_ in enumerate(n_samples):

        print(f'================================================> n_samples: {n_samples_}')

        x, yt, X, C = get_scaling_dataset(n_samples=n_samples_, n_features=int(np.ceil(n_features_per_sample*n_samples_)), sparse_construction=sparse_construction)

        C = sparse.coo_array(C)
        X = sparse.coo_array(X)

        times[i] = {}
        weights[i] = {}
        otlms[i] = {}

        kwargs = {'λ':λ, 'α':α, 'max_iter':1000, 'max_iter_mm':1, 'tol':1e-8, 'options':{'disp':True}}

        for ε_ in ε:

            if variant == 'l1l1':
                otlm = OTLM(datafit='tv', ε=ε_, penalty='l1', **kwargs)
            elif variant == 'fixl1':
                otlm = OTLM(datafit='fix', ε=ε_, penalty='l1', **kwargs)
            elif variant == 'l2l2':
                otlm = OTLM(datafit='quadratic', ε=ε_, penalty='l2', **kwargs)

            print(f'============= OTLM scaling fit started ε={ε_:.2e}')

            time_start = time.time()
            otlm.fit(C, X, yt)  
            time_end = time.time()
            tag = f'OTLM ε={ε_:.2e}'
            times[i][tag] = time_end - time_start
            weights[i][tag] = otlm.w
            otlms[i][tag] = otlm
            yp_otlm = otlm.predict(X)
            print(f'n_samples: {n_samples_}, time {tag}: {times[i][tag]:.2f}')


        solver = 'CLARABEL'
        if variant == 'l1l1':
            otlm_solver = OTLM_L1L1_LM(alpha=α, lam=λ)
            tag = f'LP solver'
        elif variant == 'fixl1':
            otlm_solver = OTLM_L1_LM(alpha=α)
            tag = f'LP solver'
        elif variant == 'l2l2':
            otlm_solver = OTLM_L2L2_QP(alpha=α, lam=λ, solver=solver)
            tag = f'QP solver'

        print(f'============= OTLM solver fit started')
        time_start = time.time()
        otlm_solver.fit(C, X, yt)
        time_end = time.time()
        
        times[i][tag] = time_end - time_start
        weights[i][tag] = otlm_solver.coef_
        otlms[i][tag] = otlm_solver
        yp_solver = otlm_solver.predict(X)  
        print(f'n_samples: {n_samples_}, time {tag}: {times[i][tag]:.2f}')

        nx, ny = 1, 1; figsize=4; fig, ax  = plt.subplots(nx, ny, figsize=(ny * figsize * 20 , nx * figsize), squeeze=False); axl=ax.ravel(); axc=ax[0,0];  fig.set_label(' ')
        for j, tag in enumerate(times[i].keys()):
            if j == 0:
                axc.scatter(x, yt, label='target', c='hotpink', lw=3)
            axc.plot(x, X.dot(weights[i][tag]), label=tag, lw=3)
        axc.legend();

    return times, weights, otlms



def plot_time_scaling(n_samples, times, weights, ε, title='', ylabel=True, ylim=None):

    nx, ny = 1, 1; figsize=2.5; fig, ax  = plt.subplots(nx, ny, figsize=(ny * figsize * 1.2 , nx * figsize), squeeze=False); axl=ax.ravel(); axc=ax[0,0];  fig.set_label(' ')

    color_solver = 'black'
    colors_otlm = plt.cm.tab10.colors

    keys = list(times[0].keys())
    tags = [r'$\epsilon$=' + f'{ε_:1.0e}' for ε_ in ε] + [r'$\epsilon$=0 ' + '(' + keys[-1] + ')']
    print(tags)
    tags = tags[::-1]
    keys = keys[::-1]

    x = n_samples
    for j, (key, tag) in enumerate(zip(keys, tags)):
        y = [times[i][key] for i in range(len(n_samples))]
        if key.lower().endswith('solver'):
            color_ = color_solver
        else:
            color_ = colors_otlm[j-1]
        axl[0].loglog(x, y, label=tag, lw=3, c=color_)

    axl[0].legend();
    axl[0].set(xlabel='Problem size N', ylabel='Time (s)' if ylabel else None, ylim=ylim)
    axl[0].set_title(title)
    if ylabel is None:
        axl[0].set_yticklabels([])
    axl[0].grid(False)
    plt.show()

    return fig

def plot_w_diff(ε, weights, weigths_ref, labels, title=''):

    nx, ny = 1, 1; figsize=2.5; fig, ax  = plt.subplots(nx, ny, figsize=(ny * figsize * 1.2 , nx * figsize), squeeze=False); axl=ax.ravel(); axc=ax[0,0];  fig.set_label(' ')

    for i, (w, w_ref, label) in enumerate(zip(weights, weigths_ref, labels)):

        w_diff = []
        for j in range(len(w)):
            w_diff_ = np.sqrt(np.mean((w[j]-w_ref)**2))
            w_diff.append(w_diff_)

        w_diff = np.array(w_diff)
        axc.plot(ε, w_diff, label=label, lw=3)

    axc.legend();
    axc.set(xlabel=r'Entropic regularization $\epsilon$', ylabel=r'RMSE $\mathbf{w}^{\epsilon} - \mathbf{w}$', xscale='log', yscale='linear')
    axc.set_title(title)
    axc.grid(False)
    plt.show()

    return fig

###############################################
###############################################
##
##
## Real data analysis
##
##
###############################################
###############################################


###############################################
##
## Muonic X-ray spectra
##
###############################################


def get_samurai_data(fname = './otlm_real_data_samurai.pkl', sample_name='CRM_Ag-Cu_alloy_p45_dp2', basis_name='basis', min_frac=0.01, verb=True):

    samples_fitted = read_from_pickle(fname)    
    sample_fitted = samples_fitted[sample_name]
    X = sample_fitted[basis_name]
    x = sample_fitted['x']
    elements_true = sample_fitted['elements_true']
    proportions_true = sample_fitted['proportions_true']
    muon_capture_prob = sample_fitted['muon_capture_prob']
    
    y = sample_fitted['yb'] - sample_fitted['bl']
    y = np.where(y<1e-6, 1e-6, y)

    if verb:
        print(sample_fitted['proportions_true'])
        print(sample_fitted['elements_true'])
        print('x', x.shape, 'y', y.shape)
        print('elements_true', elements_true)
        print('muon_capture_prob', muon_capture_prob)


    mask_elements = [e for e in elements_true if proportions_true[e] > min_frac]
    elemid = sample_fitted[f'{basis_name}_elemid']
    mask_elemid = [e for e in elemid if e in mask_elements]
    mask_cols = [i for i, e in enumerate(elemid) if e in mask_elements]
    X = X[:,mask_cols]
    
    if verb:
        print(f'mask {min_frac:.2f}', elements_true, '->', mask_elements)

    mudirac = sample_fitted['mudirac4']
    mudirac = mudirac[mask_cols]

    return X, y, x, mask_elemid



def plot_mixe_data_fit_singlepanel(otlm, x, y, X, xlim, tag='', ylim=None):

    nx, ny = 1, 1; 
    figsize=2.5; 
    fig, ax  = plt.subplots(nx, ny, figsize=(ny * figsize * 1.5 , nx * figsize), squeeze=False); 
    fig.set_label(' ')
    
    a = ax[0,0]
    select = (x>xlim[0]) & (x<xlim[1])
    a.plot(x[select], y[select] * otlm.norm_factor, c='hotpink', lw=5, zorder=1, alpha=0.5, label='Target')
    a.plot(x[select], otlm.predict(X, target=True)[select] * otlm.norm_factor, c='red', lw=1, zorder=2, label='OTLM target')
    a.plot(x[select], otlm.predict(X)[select] * otlm.norm_factor, c='dodgerblue', lw=1,  zorder=4, alpha=1, label='OTLM source')
    a.plot(np.arange(xlim[0], xlim[1]), np.zeros(xlim[1]-xlim[0]), c='black', lw=0, zorder=5)
    a.text(0.95, 0.95, tag, transform=a.transAxes, ha='right', va='top')
    a.get_yaxis().get_major_formatter().set_powerlimits((0, 0))
    a.set(ylabel='Calibrated spectrum', xlabel='Energy [keV]', ylim=ylim)
    a.legend(framealpha=0, loc='upper left')

    return fig

def plot_mixe_data_mcmc_singlepanel(samples, x, y, xlim, tag='', ylim=None):

    model='4components' 
    num_tests=100
    components=True
    title=''
    y_background=None

    P = 4
    N = samples.shape[1]//P

    amp = samples[:num_tests,0]
    mean = samples[:num_tests,1]
    scale = samples[:num_tests,2]
    skew = samples[:num_tests,3]

    models_components = skew_gaussian_multi_batch_modeparam(x, amp, mean, scale, skew, background=np.zeros_like((N,)))
    models = models_components.sum(axis=1)

    if y_background is not None:
        models = models + y_background

    plt.figure(figsize=(1.5*2.5, 2.5))
    plt.plot(x, y, 'hotpink', label='Target', lw=5, zorder=1, alpha=0.5)
    plt.gca().get_yaxis().get_major_formatter().set_powerlimits((0, 0))

    for i, model in enumerate(models):
        plt.plot(x, model, alpha=1, color='red', zorder=10, label='Bayesian mixture posterior' if i==0 else None)

    if components:
        colors = plt.get_cmap('tab10').colors
        for i in range(models_components.shape[1]):
            for j in range(models_components.shape[0]):
                m = models_components[j, i, :] if y_background is None else models_components[j, i, :] + y_background
                plt.plot(x, m, alpha=0.1, color=colors[i])

    plt.gca().text(0.95, 0.95, tag, transform=plt.gca().transAxes, ha='right', va='top')
    plt.gca().set(ylabel='Calibrated spectrum', xlabel='Energy [keV]')
    plt.legend(framealpha=0)
    plt.grid(True)
    plt.gca().set(xlim=xlim, ylim=ylim, title=title)
    
    return plt.gcf()


def skewnorm_pdf_modeparam(x, alpha, gamma, omega):
    """
    Given:
      - alpha: skew parameter
      - gamma: desired mode of the skew-normal
      - omega: scale parameter (> 0)
      
    We compute the location xi so that the distribution
    SkewNormal(alpha, xi, omega) has mode = gamma (approximately).

    Then we return the skew-normal logPDF evaluated at x.
    """

    def m0(alpha):
        """
        Approximate formula for the 'standardized' mode of a skew-normal
        with shape parameter alpha (loc=0, scale=1).
        
        m0(alpha) ≈ sqrt(2/pi)*delta 
                - (1 - pi/4) * [(sqrt(2/pi)*delta)^3 / (1 - (2/pi)*delta^2)]
                - 0.5 * sgn(alpha) * exp(-2*pi / |alpha|)
        where delta = alpha / sqrt(1 + alpha^2).
        
        This formula is an approximation, but is often quite accurate.
        """
        
        delta = alpha / np.sqrt(1.0 + alpha**2)
        
        # First piece: sqrt(2/pi) * delta
        part1 = np.sqrt(2.0/np.pi) * delta
        
        # Second piece: (1 - pi/4)* [ (sqrt(2/pi)*delta)^3 / (1 - (2/pi)*delta^2 ) ]
        numerator   = (np.sqrt(2.0/np.pi)*delta)**3
        denominator = 1.0 - (2.0/np.pi)*delta**2
        part2 = (1.0 - np.pi/4.0) * (numerator / denominator)
        
        # Third piece: (sgn(alpha)/2) * exp(-2pi / |alpha|)
        part3 = 0.5 * np.sign(alpha) * np.exp(-2.0*np.pi / abs(alpha))
        
        return part1 - part2 - part3


    # 1) Compute the approximate 'standard mode' from alpha
    m0_val = m0(alpha)
    
    # 2) Solve for xi (the location), given the target mode = gamma
    #    gamma = xi + omega * m0(alpha)  =>  xi = gamma - omega*m0(alpha)
    xi = gamma - omega * m0_val
    
    # 3) Evaluate the log-PDF of the skew-normal at x
    return skewnorm.pdf(x, alpha, loc=xi, scale=omega)

def skew_gaussian_multi_batch_modeparam(x, amplitude, mode, scale, skew, background):
    x = x[np.newaxis, np.newaxis, :]  # Reshape x to (1, 1, M) for broadcasting
    y = skewnorm_pdf_modeparam(x, alpha=skew[:,:,np.newaxis], gamma=mode[:,:,np.newaxis], omega=scale[:,:,np.newaxis])
    y = y / np.sum(y, axis=-1, keepdims=True)
    background_ = background.reshape(-1, background.shape[-1], 1)
    return amplitude[:, :, np.newaxis] * y + background_

    
def get_mixe_fit_comparison_data(posteriors, otlm, samp_name):

    lines = []

    elemid = np.array(posteriors[samp_name]['elemid'])
    elems = np.unique(elemid)

    Zs = [pt.element(e).atomic_number for e in elems]
    Zs = np.sort(Zs)

    print(f'------------- {samp_name}: {elems}')
    mean_err = []
    for Z in Zs:
        
        elem = pt.element(int(Z)).symbol
        
        post = posteriors[samp_name][elem][0]
        mudirac = posteriors[samp_name][elem][2]
        print(mudirac.shape)

        w_tot = post[:,0,:].sum(axis=-1)
        w_tot_mean = np.mean(w_tot)
        w_tot_std = np.std(w_tot)
        w_otlm = otlm.coef_[elemid==elem] * otlm.norm_factor

        print(f'total {elem}   : {w_tot_mean: .4e} +/- {w_tot_std: .4e} frac_err {w_tot_std/w_tot_mean: .4e}    w_otlm {w_otlm.sum(): .4e} diff {(w_otlm.sum()-w_tot_mean)/w_tot_std: .4e}')

        for i in range(post.shape[2]):
            w_mean = np.mean(post[:,0,i])
            w_std = np.std(post[:,0,i])
            pos_mean = np.mean(post[:,1,i])

            print(f'      {elem} {i:>2d}: {w_mean: .4e} +/- {w_std: .4e} frac_err {w_std/w_mean: .4e} detect_sig  {w_mean/w_std: .4e}   w_otlm {w_otlm[i]: .4e} diff {(w_otlm[i]-w_mean)/w_std: .4e}    pos_mean {pos_mean: .4e}  pos_mudirac {mudirac["E [eV]"][i]/1000: .4e}')

            kalpha_spec = 'K$\\alpha_1$' if mudirac['QS_i'][i] == 'L3' else 'K$\\alpha_2$'
            line = [mudirac["Z"][i], mudirac["A"][i], kalpha_spec, w_mean, w_std, w_otlm[i]]
            lines.append(line)


            frac_err = np.abs(w_mean-w_otlm[i])/w_std
            if np.isfinite(frac_err):
                mean_err.append(frac_err)

    lines = np.array(lines)
    print('mean_err', np.mean(mean_err))
    return lines

def fit_mixe_single_sample(samp_name, ε, α, ρ, λ, max_dist, reg_type, datafit, penalty, size_scale=0.8, plot=True, min_element_frac=0.05, fname='./data/otlm_real_data_samurai_channelall_downsample100_basis4_nsamp3.pkl'):

    
    
    print('=============================== {} ==============================='.format(samp_name))

    X, y_counts, x, mask_elemid = get_samurai_data(fname=fname, sample_name=samp_name, basis_name='basis4', min_frac=min_element_frac)

    y_counts = np.where(y_counts<1e-6, 1e-6, y_counts)
    norm_factor = np.max(y_counts)
    y = y_counts/norm_factor 
    dist = np.abs(x[:,None] - x[None,:])
    C = ρ * dist**2 / x[None,:]**2 + 1e-20
    if max_dist is not None:
        C[dist>max_dist] = 0
    C = sparse.coo_array(C)

    otlm = OTLM(datafit=datafit, penalty=penalty, λ=λ, ε=ε, α=α, reg_type=reg_type, max_iter=1000, max_iter_mm=100, tol=1e-8, options={'disp': True})    
    otlm.fit(C, X, y)
    print('otlm.coef_', astr(otlm.coef_), np.sum(otlm.coef_))

    mask_elemid = np.array(mask_elemid)
    otlm.norm_factor = norm_factor
    for elem in np.unique(mask_elemid):
        print(elem, astr(otlm.coef_[mask_elemid==elem]))

    return otlm, y, x, X

def plot_mixe_fit_comparison(lines, samp_name, ymin=1e1, title=None, figsize=(6,3)):

    ymm = lines[:,3].astype(float)
    ymm_err = lines[:,4].astype(float)
    yot = lines[:,5].astype(float)
    xp = np.arange(len(lines))

    plt.figure(figsize=figsize)
    
    # select = yp>ymin
    select_ot = yot>ymin
    select_mm = ymm>ymin
    
    plt.plot(xp[select_ot], yot[select_ot], 'x', color='red')
    plt.errorbar(xp[select_mm], ymm[select_mm], yerr=ymm_err[select_mm], fmt='.')
    plt.plot(xp[~select_ot], np.full(sum(~select_ot), 1.5*ymin), 'v', color='red', alpha=0.5)
    plt.plot(xp[~select_mm], np.full(sum(~select_mm), 1.5*ymin), 'v', color='blue', alpha=0.5)


    plt.ylabel('Weight value') 
    tags = [r'$_{%s}$' % l[1] + f'{pt.element(int(l[0])).symbol} ' + f'{l[2]}' for l in lines]
    plt.xticks(np.arange(len(lines)), tags, rotation=90)
    plt.gca().set(yscale='log', ylim=[ymin, None])
    plt.grid(True)
    plt.title(title)
    
    return plt.gcf()


###############################################
##
## Infrared Spectroscopy
##
###############################################

# TODO