import numba
import numpy as np
import scipy.interpolate as inte
import hmc_inclination as hinc
import hmc_tapered as hr
import gmm_anyk as ga
import SimplexHMC as shmc

H_bins = np.arange(5.05, 9.05, 0.1)


def sample_inverse_transform(nsamp, x, y):
    cumulative = np.cumsum(y) * np.diff(x)[0]
    cumulative -= cumulative[0]
    cumulative /= cumulative[-1]
    inv_cdf = inte.interp1d(cumulative, x, "cubic")
    r = np.random.uniform(size=nsamp)
    return inv_cdf(r)


def sample_powerlaw(nsamp, slope, Hmin, Hmax):
    un = np.random.uniform(size=nsamp)
    return np.log(
        10 ** (slope * Hmin) + un * (10 ** (Hmax * slope) - 10 ** (Hmin * slope))
    ) / (slope * np.log(10))


def sample_beta(nsamp, Abar, s):
    alpha = Abar * np.power(10, s)
    beta = alpha / Abar - alpha
    return np.random.beta(alpha, beta, size=nsamp)


def sample_vonmises(nsamp, coeff):
    return None


def resample_H_distribution(
    n_obj,
    f,
    gmm,
    params_H=None,
    sel_H=None,
    params_LCA=None,
    Hmin=5.5,
    Hmax=8.2,
):
    sample_f = np.random.choice(len(f), size=n_obj, p=f)

    comp, un = np.unique(sample_f, return_counts=True)

    col_sample = []

    if params_H is not None:
        H_sample = []
    else:
        H_sample = None

    if params_LCA is not None:
        LCA_sample = []
    else:
        LCA_sample = None

    for i, j in zip(comp, un):
        colors = gmm.sampleComponent(j, i % gmm.K_best)
        col_sample.append(colors)
        if params_H is not None:
            pH = hinc.singleslope(H_bins, params_H[i % gmm.K_best])
            msk = np.logical_and(H_bins > Hmin, H_bins < Hmax)
            pHps = pH * sel_H[i]
            Hs = sample_inverse_transform(j, H_bins[msk], pHps[msk])
            H_sample.append(Hs)
        if params_LCA is not None:
            LCAs = sample_beta(
                j,
                params_LCA[2 * (i % gmm.K_best)],
                params_LCA[2 * (i % gmm.K_best) + 1],
            )
            LCA_sample.append(LCAs)

    col_sample = np.vstack(col_sample)

    if params_H is not None:
        H_sample = np.hstack(H_sample)
    if params_LCA is not None:
        LCA_sample = np.hstack(LCA_sample)

    return col_sample, H_sample, LCA_sample


def slide_chain(
    chain_color,
    mean_color,
    chain_H,
    chain_LCA,
    samples_color,
    samples_H=None,
    samples_LCA=None,
):
    # I will be assuming that the chains are sorted by H

    if samples_H is not None:
        s = np.argsort(samples_H)

        samples_color = samples_color[s]
        samples_H = samples_H[s]
        if samples_LCA is not None:
            samples_LCA = samples_LCA[s]

    color_sampled_chain = []
    mean_color_sampled = []
    H_sampled_chain = []
    LCA_sampled_chain = []
    for i in range(len(samples_color)):
        current_color = chain_color[i]
        s = np.random.randint(len(current_color))
        c = samples_color[i] + current_color - current_color[s]
        color_sampled_chain.append(c)
        mean_color_sampled.append(mean_color[i] + samples_color[i] - current_color[s])
        if samples_H is not None:
            current_H = chain_H[i]
            H = samples_H[i] + current_H - current_H[s]
            H_sampled_chain.append(H)
        if samples_LCA is not None:
            current_LCA = chain_LCA[i]
            LCA = np.power(current_LCA, np.log(samples_LCA[i] / current_LCA[s]))
            LCA_sampled_chain.append(LCA)

    return color_sampled_chain, mean_color_sampled, H_sampled_chain, LCA_sampled_chain


# reimplementing this, as the only place this lives is inside the LogLikeGMMSingleSlope class and its children
def compute_gmm_chain(gmm, chains_color):
    S_obj_noamp = []
    logdet = np.linalg.slogdet(gmm.cov_best)[1]
    for x in chains_color:
        S_obj_noamp.append(
            ga.log_multigaussian_no_off(x, gmm.mean_best, gmm.cov_best)
            - logdet / 2
            - np.log(2 * np.pi) * 3 / 2
        )
    return S_obj_noamp


def resample_refit_gmm(
    gmm,
    color_chain,
    mean_colors,
    sigma_colors,
    f,
    n_reject=0,
    Hmin=5.5,
    Hmax=8.2,
):
    col_sample, _, _ = resample_H_distribution(
        len(color_chain), f, gmm, None, None, None, Hmin, Hmax
    )

    sampled_colors, mean_sampled_color, _, _ = slide_chain(
        color_chain, mean_colors, None, None, col_sample, None, None
    )

    mean_sampled_color = np.array(mean_sampled_color)
    

    newgmm = ga.GMMNoise(2, 3)

    newgmm.mean = gmm.mean_best
    newgmm.cov = gmm.cov_best
    newgmm.amp = gmm.amp_best

    newgmm.fit(mean_sampled_color, 2, 0.0, miniter=50, tolerance=1e-5, maxiter=1000, offset=sigma_colors, initialize=False)

    if n_reject > 0:
        z = ga.log_multigaussian(mean_sampled_color, newgmm.mean_best, newgmm.cov_best, sigma_colors)
        zz = np.amax(z,axis=1)
        mask = np.argsort(zz)[n_reject:]
        mean_sampled_color = mean_sampled_color[mask]
        sampled_colors = [sampled_colors[i] for i in mask]
        sigma_colors = sigma_colors[mask]
        newgmm.fit(mean_sampled_color, 2, 0.0, miniter=50, tolerance=1e-5, maxiter=1000, offset=sigma_colors, initialize=False)
        
    jacobian = []
    for i in range(len(mean_sampled_color)):
        jacobian.append(hinc.jacobian_color(sampled_colors[i]))

    S_obj = compute_gmm_chain(newgmm, sampled_colors)
    Rialpha = hinc.Rialpha_gmm(2, len(mean_sampled_color), numba.typed.List(jacobian), numba.typed.List(S_obj))
    
    return -np.sum(np.log(np.sum(Rialpha * newgmm.amp_best, axis=1)))


def resample_refit_singleslope_gmm(gmm, color_chain, H_chain, mean_colors,  sigma_colors, H_sel, f, H_params, Hmin = 5.5, Hmax = 8.2, n_reject = 0):
    
    col_sample, H_sample, _ = resample_H_distribution(len(color_chain), f, gmm, H_params, H_sel, None, Hmin, Hmax)
    
    sampled_colors, mean_sampled_color, sampled_H, _ = slide_chain(color_chain, mean_colors, H_chain, None, col_sample, H_sample, None)
    
    if n_reject > 0:
        z = ga.log_multigaussian(mean_sampled_color, gmm.mean_best, gmm.cov_best, sigma_colors)
        zz = np.amax(z,axis=1)
        mask = np.argsort(zz)[n_reject:]
        mean_sampled_color = mean_sampled_color[mask]
        sampled_colors = [sampled_colors[i] for i in mask]
        sigma_colors = sigma_colors[mask]
        sampled_H = sampled_H[mask]

    
    newlike = hinc.LogLikeGMMSingleSlope(gmm, sampled_colors, sampled_H, H_sel, len(f))
    
    hmc = shmc.SimplexGeneralHMC(f, H_params, newlike.likelihood, newlike.gradient_f, newlike.gradient_theta, 
                                 massTheta=10000*np.identity(2), dt=0.1, massF=10000.*np.identity(2))

    hmc.descend(debug=0, learning_rate=0.00001, maxSteps=1000, minimumShift=0.01, firstPEStep=10.)
    
    return newlike.likelihood(hmc.f, hmc.theta)

    
    

def resample_refit_singleslope(H_chain, H_sel, slope, Hmin = 5.5, Hmax = 8.2, n_reject = 0):
    
    
    pH = hinc.singleslope(H_bins, slope)
    msk = np.logical_and(H_bins > Hmin, H_bins < Hmax)
    pHps = pH * H_sel[0]
    H_sample = sample_inverse_transform(len(H_chain), H_bins[msk], pHps[msk])
    
    s = np.argsort(H_sample)
    H_sample = H_sample[s]
    H_sampled_chain = []
    for i in range(len(H_chain)):
        s = np.random.choice(len(H_chain[i]))
        current_H = H_chain[i]
        H = H_sample[i] + current_H - current_H[s]
        H_sampled_chain.append(H)

    
    newlike = hinc.LogLikeSingleSlope(H_sampled_chain, H_sel, 1)
    
    hmc = shmc.GeneralHMC(np.array([slope]), newlike.likelihood,newlike.gradient_theta, massTheta=10000*np.identity(1), dt=0.1)

    hmc.descend(debug=0, learning_rate=0.00001, maxSteps=1000, minimumShift=0.01, firstPEStep=10.)
    
    return newlike.likelihood(hmc.theta)

def resample_refit_rolling(H_chain, H_sel, slope1, slope2, Hmin = 5.5, Hmax = 8.2):
    
    
    pH = hr.rolling_slope(H_bins, slope1, slope2)
    msk = np.logical_and(H_bins > Hmin, H_bins < Hmax)
    pHps = pH * H_sel[0]
    H_sample = sample_inverse_transform(len(H_chain), H_bins[msk], pHps[msk])
    
    s = np.argsort(H_sample)
    H_sample = H_sample[s]
    H_sampled_chain = []
    for i in range(len(H_chain)):
        s = np.random.choice(len(H_chain[i]))
        current_H = H_chain[i]
        H = H_sample[i] + current_H - current_H[s]
        H_sampled_chain.append(H)

    
    newlike = hr.LogLikeRolling(H_sampled_chain, H_sel, 1)
    
    hmc = shmc.GeneralHMC(np.array([slope1, slope2]), newlike.likelihood,newlike.gradient_theta, massTheta=10000*np.identity(2), dt=0.1)

    hmc.descend(debug=0, learning_rate=0.00001, maxSteps=1000, minimumShift=0.01, firstPEStep=10.)
    
    return newlike.likelihood(hmc.theta)

def resample_refit_rolling_gmm(gmm, color_chain, H_chain, mean_colors,  sigma_colors, H_sel, f, H_params, Hmin = 5.5, Hmax = 8.2, n_reject = 0):
    
    col_sample, H_sample, _ = resample_H_distribution(len(color_chain), f, gmm, H_params, H_sel, None, Hmin, Hmax)
    
    sampled_colors, mean_sampled_color, sampled_H, _ = slide_chain(color_chain, mean_colors, H_chain, None, col_sample, H_sample, None)
    
    if n_reject > 0:
        z = ga.log_multigaussian(mean_sampled_color, gmm.mean_best, gmm.cov_best, sigma_colors)
        zz = np.amax(z,axis=1)
        mask = np.argsort(zz)[n_reject:]
        mean_sampled_color = mean_sampled_color[mask]
        sampled_colors = [sampled_colors[i] for i in mask]
        sigma_colors = sigma_colors[mask]
        sampled_H = sampled_H[mask]

    
    newlike = hinc.LogLikeGMMDynamicsRolling(gmm, sampled_colors, sampled_H, H_sel, len(f))
    
    hmc = shmc.SimplexGeneralHMC(f, H_params, newlike.likelihood, newlike.gradient_f, newlike.gradient_theta, 
                                 massTheta=10000*np.identity(2), dt=0.1, massF=10000.*np.identity(2))

    hmc.descend(debug=0, learning_rate=0.00001, maxSteps=1000, minimumShift=0.01, firstPEStep=10.)
    
    return newlike.likelihood(hmc.f, hmc.theta)

def resample_refit_singleslope(H_chain, H_sel, slope, Hmin = 5.5, Hmax = 8.2, n_reject = 0):
    
    
    pH = hinc.singleslope(H_bins, slope)
    msk = np.logical_and(H_bins > Hmin, H_bins < Hmax)
    pHps = pH * H_sel[0]
    H_sample = sample_inverse_transform(len(H_chain), H_bins[msk], pHps[msk])
    
    s = np.argsort(H_sample)
    H_sample = H_sample[s]
    H_sampled_chain = []
    for i in range(len(H_chain)):
        s = np.random.choice(len(H_chain[i]))
        current_H = H_chain[i]
        H = H_sample[i] + current_H - current_H[s]
        H_sampled_chain.append(H)

    
    newlike = hinc.LogLikeSingleSlope(H_sampled_chain, H_sel, 1)
    
    hmc = shmc.GeneralHMC(np.array([slope]), newlike.likelihood,newlike.gradient_theta, massTheta=10000*np.identity(1), dt=0.1)

    hmc.descend(debug=0, learning_rate=0.00001, maxSteps=1000, minimumShift=0.01, firstPEStep=10.)
    
    return newlike.likelihood(hmc.theta)

def resample_ks(H_chain, H_sample):

	s = np.argsort(H_sample)
	H_sample = H_sample[s]
	H_sampled_chain = []
	for i in range(len(H_chain)):
		s = np.random.choice(len(H_chain[i]))
		current_H = H_chain[i]
		H = H_sample[i] + current_H - current_H[s]
		H_sampled_chain.append(H)


	data_chain = []
	chain_wgt = []
	for j in H_sampled_chain:
		data_chain.append(j)
		chain_wgt.append(len(j) * np.ones_like(j))
	data_chain = np.hstack(data_chain)
	chain_wgt = np.hstack(chain_wgt)

	return data_chain, chain_wgt 



def resample_refit_twoslopes(H_chain, H_sel, slope1, slope2, Hmin = 5.5, Hmax = 8.2, Htransition = 7.0):
    
    
    pH = hinc.singleslope(H_bins, slope)
    msk = np.logical_and(H_bins > Hmin, H_bins < Hmax)
    pHps = pH * H_sel[0]
    H_sample = sample_inverse_transform(len(H_chain), H_bins[msk], pHps[msk])
    
    s = np.argsort(H_sample)
    H_sample = H_sample[s]
    H_sampled_chain = []
    for i in range(len(H_chain)):
        s = np.random.choice(len(H_chain[i]))
        current_H = H_chain[i]
        H = H_sample[i] + current_H - current_H[s]
        H_sampled_chain.append(H)

    
    newlike = hinc.LogLikeSingleSlope(H_sampled_chain, H_sel, 1)
    
    hmc = shmc.GeneralHMC(np.array([slope]), newlike.likelihood,newlike.gradient_theta, massTheta=10000*np.identity(1), dt=0.1)

    hmc.descend(debug=0, learning_rate=0.00001, maxSteps=1000, minimumShift=0.01, firstPEStep=10.)
    
    return newlike.likelihood(hmc.theta)
