import numpy as np
import numpy.random as rd
import scipy
import os
import os.path
import rebound
from celmech import Andoyer
import mr_forecast as mr
# Dan's stability model packages
import random
import dill
import sys
import time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')
import scipy
from scipy.optimize import curve_fit
#import radvel

sys.path.append("../spock")
from spock import StabilityClassifier

seconds_p_day = 86400
days_p_year = 365.25
meters_p_AU = 149597870700
earth_mass_p_solar_mass = 333000
year_p_reboundtime = 1 / (2 * np.pi)
AU_p_RS = 0.00465047
Nplanets = 3

def build_sim(Ps, ms, es, incs, Mstar, Rstar):
    
    assert Nplanets==len(ms)==len(es)==len(Ps)==len(incs)
        
    Ws=2 * np.pi * rd.sample(Nplanets)
    ws=2 * np.pi * rd.sample(Nplanets)
    Ms=2 * np.pi * rd.sample(Nplanets)

    radii = np.zeros(Nplanets)
    for i in range(Nplanets):
        radii[i] = np.cbrt(Ps[i] * Ps[i] * ms[i] / earth_mass_p_solar_mass / Mstar / 3) / 2

    #set up simulation
    sim = rebound.Simulation()

    #add star
    sim.add(m=Mstar, r=Rstar)

    for i in range(Nplanets):
        sim.add(m=ms[i] / earth_mass_p_solar_mass, P=Ps[i] / year_p_reboundtime, e=es[i], inc=incs[i], Omega=Ws[i], omega=ws[i], M=Ms[i], r=radii[i]) #G=1 units!
    sim.move_to_com()
    
    sim.collision = 'line'
    sim.collision_resolve = collision
    sim.ri_whfast.keep_unsynchronized = 1
    sim.ri_whfast.safe_mode = 0
    
    sim.integrator = "whfast"
    sim.dt = np.sqrt(2) / 40 * sim.particles[1].P  # ~0.035355

    return sim

def rebound_rvs(times, sim):
    
    assert np.all(np.diff(times)>=0), "times passed aren't sorted"
    
    times_rb = times / year_p_reboundtime
    
    rvs = np.zeros(len(times))
    for i in range(len(times)):
        sim.integrate(times_rb[i])
        rvs[i] = sim.particles[0].vx
    rvs = rvs * meters_p_AU / year_p_reboundtime / days_p_year / seconds_p_day
    return rvs
    
def a_normal(mean, upper_sigma, lower_sigma):
    draw = rd.normal()
    if draw > 0:
        draw *= upper_sigma
    else:
        draw *= lower_sigma
    return draw + mean

def valid_system(system):
    assert system in ["HR858", "K431", "TOI270", "L98-59", "K23"]
    
# def draw_masses(system, sample=int(1e4)):
    
#     if system == "HR858":
#         R, Rerr = [2.085, 1.939, 2.164], [0.066, 0.069, 0.085]
        
#     if system == "K431":
#         R, Rerr = [1.088, 1.072, 1.307], [0.146, 0.117, 0.16]
    
#     if system == "TOI270":
#         R, Rerr = [1.247, 2.42, 2.13], [0.086, 0.13, 0.12]
        
#     if system == "L98-59":
#         R, Rerr = [0.8, 1.35, 1.57], [0.05, 0.08, 0.14]
    
#     #Berger et al. 2018
#     if system == "K23":
#         R, Rerr = [1.728, 3.075, 2.175], [0.086, 0.136, 0.105]
    
#     assert Nplanets==len(R)==len(Rerr)
#     mb = scipy.stats.gaussian_kde(mr.Rstat2M(R[0], Rerr[0], return_post=True, sample_size=sample, grid_size=sample))
#     mc = scipy.stats.gaussian_kde(mr.Rstat2M(R[1], Rerr[1], return_post=True, sample_size=sample, grid_size=sample))
#     md = scipy.stats.gaussian_kde(mr.Rstat2M(R[2], Rerr[2], return_post=True, sample_size=sample, grid_size=sample))
#     return mb, mc, md

# def build_forecasted_system(system, mb, mc, md):
    
#     incs=-np.ones(3)

#     if system == "HR858":
#         Ps = np.array([a_normal(3.58599, 0.00015, 0.00015), a_normal(5.97293, 0.00060, 0.00053), a_normal(11.2300, 0.0011, 0.0010)])
#         Mstar = a_normal(1.145, 0.074, 0.08)
#         Rstar = a_normal(1.31, 0.024, 0.022)
#         incs = np.array([a_normal(85.5, 1.5, 0.5), a_normal(86.23, 0.26, 0.26), a_normal(87.43, 0.18, 0.19)])
        
#     if system == "K431":
#         Ps = Ps = np.array([a_normal(6.80252171, 7.931e-05, 7.931e-05), a_normal(8.70337044, 9.645e-05, 9.645e-05), a_normal(11.9216214, 0.0001182, 0.0001182)])
#         Mstar = a_normal(1.150, 0.087, 0.059)
#         Rstar = a_normal(1.41, 0.275, 0.24)
    
#     if system == "TOI270":
#         Ps = np.array([a_normal(3.36008, 0.000065, 0.000070), a_normal(5.660172, 0.000035, 0.000035), a_normal(11.38014, 0.00011, 0.00010)])
#         Mstar = a_normal(0.40, 0.02, 0.02)
#         Rstar = a_normal(0.38, 0.02, 0.02)
#         incs = np.array([a_normal(88.65, 0.85, 1.40), a_normal(89.53, 0.30, 0.42), a_normal(89.69, 0.16, 0.12)])
        
#     if system == "L98-59":
#         Ps = np.array([a_normal(2.25314, 0.00002, 0.00002), a_normal(3.690621, 0.000013, 0.000014), a_normal(7.45086, 0.00004, 0.00005)])
#         Mstar = a_normal(0.313, 0.014, 0.014)
#         Rstar = a_normal(0.312, 0.014, 0.014)
#         incs = np.array([a_normal(88.7, 0.8, 0.7), a_normal(89.3, 0.4, 0.5), a_normal(88.5, 0.2, 0.5)])
        
#     #Holczer et al. 2016 and Morton 2016
#     if system == "K23":
#         Ps = np.array([a_normal(7.10697755, 0.00001048, 0.00001048), a_normal(10.74244253, 0.00000349, 0.00000349), a_normal(15.27458613, 0.00000510, 0.00000510)])
#         Mstar = a_normal(1.100, 0.038, 0.030)
#         Rstar = a_normal(1.548, 0.048, 0.048)
        
#     Rstar *= AU_p_RS
#     Ps /= days_p_year
    
#     As = np.cbrt((Ps*(2*np.pi))**2 * Mstar)
#     if all(incs==-np.ones(3)):
#         incs = np.array([rd.uniform(high=0.9 * Rstar * AU_p_RS / As[i]) for i in range(Nplanets)])
#     else:
#         incs = incs * np.pi / 180 - np.pi/2
    
#     es = np.array([0,0,0])
#     e0_max = max_e_inner(As[0], As[1])
#     e1_max = np.minimum(max_e_inner(As[1], As[2], es[2]), max_e_outer(As[1], As[0]))  # ~0.15
#     e2_max = max_e_outer(As[2], As[1])
#     es = rd.rand(3) * np.array([e0_max, e1_max, e2_max])
# #     bad_es = True
# #     while bad_es:
# #         es = rd.rand(3) * np.array([e0_max, e1_max, e2_max])
# #         bad_es = not check_es(As[0], es[0], As[1], es[1], As[2], es[2])
        
#     bad_ms = True
#     while bad_ms:
#         ms = np.array([mb.resample(1)[0][0], mc.resample(1)[0][0], md.resample(1)[0][0]])
#         bad_ms = any([m < 0 for m in ms])

#     return build_sim(Ps, ms, es, incs, Mstar, Rstar)

def build_Hadden_system(system, logm=False, loge=False):
    
    incs=-np.ones(3)
        
    #Hadden 2017 Table 1
    if system == "K23":
        Ps = np.array([7.107, 10.742, 15.274])
        Mstar = a_normal(1.00, 0.1, 0.1)
        Rstar = a_normal(1.548, 0.048, 0.048) # Morton 2016
        R = np.array([1.8, 3.2, 2.3])
#         Rerr = [0.086, 0.136, 0.105]
#         R = [a_normal(R[0], Rerr[0], Rerr[0]), a_normal(R[1], Rerr[1], Rerr[1]), a_normal(R[2], Rerr[2], Rerr[2])]
        
    Rstar *= AU_p_RS
    Ps /= days_p_year
    
    As = np.cbrt((Ps*(2*np.pi))**2 * Mstar)
    if all(incs==-np.ones(3)):
        incs = np.array([rd.uniform(high=0.9 * Rstar / As[i]) for i in range(Nplanets)])
    else:
        incs = incs * np.pi / 180 - np.pi/2
    
#     es = np.array([0,0,0])
#     e0_max = max_e_inner(As[0], As[1])
#     e1_max = np.minimum(max_e_inner(As[1], As[2], es[2]), max_e_outer(As[1], As[0]))  # ~0.15
#     e2_max = max_e_outer(As[2], As[1])
    
    lowm = 0.0544143  # Mearth/Rearth^3 for 0.3 g/cc
    highm = 5.44143  # Mearth/Rearth^3 for 30 g/cc
    
    if logm:
        ms = loguniform(lowm, highm, 3) * (R*R*R)
    else:
        ms = rd.uniform(lowm, highm, 3) * (R*R*R)

    if loge:
    #     e1min = ((ms[1] / ((As[0]-As[1])**2)) + (ms[2] / ((As[0]-As[2])**2))) / (sf.earth_mass_p_solar_mass * (Mstar / (As[0]**2)))
    #     e2min = ((ms[0] / ((As[1]-As[0])**2)) + (ms[2] / ((As[1]-As[2])**2))) / (sf.earth_mass_p_solar_mass * (Mstar / (As[1]**2)))
    #     e3min = ((ms[0] / ((As[2]-As[0])**2)) + (ms[1] / ((As[2]-As[1])**2))) / (sf.earth_mass_p_solar_mass * (Mstar / (As[2]**2)))
        emin = 0.001
        emax = 0.9
        es = loguniform(low=emin, high=emax, size=3)
    else:
        es = rd.rand(3) * 0.9
#     bad_es = True
#     while bad_es:
#         es = rd.rand(3) * np.array([e0_max, e1_max, e2_max])
#         bad_es = not check_es(As[0], es[0], As[1], es[1], As[2], es[2])
        
#     bad_ms = True
#     while bad_ms:
#         ms = np.array([mb.resample(1)[0][0], mc.resample(1)[0][0], md.resample(1)[0][0]])
#         bad_ms = any([m < 0 for m in ms])

    

    return build_sim(Ps, ms, es, incs, Mstar, Rstar)

# # https://arxiv.org/abs/1605.02825 Morton et al. 2016
# def build_chosen_K431(ms, es, Ps, return_extra=False):
#     sim = build_sim(Ps, ms, Mstar=a_normal(1.071, 0.059, 0.037), es=es, Rstar=1.092)
#     if return_extra:
#         return sim, Ps, ms, es
#     else:
#         return sim

def max_e_inner(a_in, a_out, e_out=0):
    return np.minimum(a_out / a_in * (1 - e_out) - 1, 1)

def max_e_outer(a_out, a_in, e_in=0):
    return 1 - a_in / a_out * (1 + e_in)

# As = np.array([6.80252171, 8.70337044, 11.9216214]) ** (2./3)

# e0_max = max_e_inner(As[0], As[1])
# e1_max = np.minimum(max_e_inner(As[1], As[2]), max_e_outer(As[1], As[0]))
# e2_max = max_e_outer(As[2], As[1])

def check_es(a0, e0, a1, e1, a2, e2):
    return (0 <= e0 <= 1) and (0 <= e1 <= 1) and (0 <= e2 <= 1) and (e0 <= max_e_inner(a0, a1, e1)) and (e1 <= max_e_inner(a1, a2, e2)) and (e1 <= max_e_outer(a1, a0, e0)) and (e2 <= max_e_outer(a2, a1, e1))

# def build_K431(mb, mc, md, es=np.array([rd.rand() * e0_max, rd.rand() * e1_max, rd.rand() * e2_max]), Ps = np.array([a_normal(6.80252171, 7.931e-05, 7.931e-05), a_normal(8.70337044, 9.645e-05, 9.645e-05), a_normal(11.9216214, 0.0001182, 0.0001182)]) / days_p_year, return_extra=False):
    
#     ms = np.array([mb.resample(1)[0][0], mc.resample(1)[0][0], md.resample(1)[0][0]])  # earth masses
#     return build_chosen_K431(ms, es, Ps, return_extra=return_extra)


#if a collision occurs, end the simulation
def collision(reb_sim, col):
    reb_sim.contents._status = 5 # causes simulation to stop running and have flag for whether sim stopped due to collision
    return 0

def replace_snapshot(sim, filename):
    if os.path.isfile(filename):
        os.remove(filename)
    sim.simulationarchive_snapshot(filename)
    
def VSA(P, m_star, m_p, e, i):
    m_planet = m_p / earth_mass_p_solar_mass
    comb_mass = m_star + m_planet
    return 2 * np.pi * np.sin(i) * m_planet / (np.sqrt(1 - (e * e)) * np.cbrt(comb_mass * comb_mass * P))* meters_p_AU / seconds_p_day / days_p_year

def mass_from_VSA(P, m_star, VSA, e, i):
    return 1 / (2 * np.pi * np.sin(i) / VSA / (np.sqrt(1 - (e * e)) * np.cbrt(m_star * m_star * P))* meters_p_AU / seconds_p_day / days_p_year) * earth_mass_p_solar_mass

def loguniform(low=0.001, high=1, size=None):
    return np.exp(rd.uniform(np.log(low), np.log(high), size))

def init_process():
    global model
    model = StabilityClassifier()

def gaussian(x, a, mean, sigma):
    return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))

def pred(sim_names, nsim):
    sim = rebound.SimulationArchive(sim_names + "_%d.bin"%nsim)[0]
#     sim.move_to_com()
#     sim.integrator="whfast"
#     sim.dt = 0.07*sim.particles[1].P
#     prob=model.predict(sim)
    prob=model.predict(sim, copy=False)
    return prob

def add_k_cols(df):
    df['k'] = 0.
    df['h'] = 0.
    df['Z12'] = 0.
    df['Zcom12'] = 0.
    df['Z23'] = 0.
    df['Zcom23'] = 0.
    df['e1'] = 0.
    df['e2'] = 0.
    df['e3'] = 0.
    df['m1'] = 0.
    df['m2'] = 0.
    df['m3'] = 0.
    df['probstability'] = 0.
    return df

def get_k(sim_names, row):
    sim = rebound.SimulationArchive(sim_names + "_%d.bin"%(row[0]))[0]
#     print(sim)
    p2 = sim.particles[2]
    row['h'] = p2.e*np.sin(p2.pomega)
    row['k'] = p2.e*np.cos(p2.pomega)
#     avars = Andoyer.from_Simulation(sim, a10=sim.particles[1].a, j=3, k=1, i1=1, i2=2, average=False)
    avars = Andoyer.from_Simulation(sim, j=3, k=1, i1=1, i2=2)
    row['Z12'] = avars.Z
    row['Zcom12'] = avars.Zcom
#     avars = Andoyer.from_Simulation(sim, a10=sim.particles[2].a, j=7, k=2, i1=2, i2=3, average=False)
    avars = Andoyer.from_Simulation(sim, j=7, k=2, i1=2, i2=3)
    row['Z23'] = avars.Z
    row['Zcom23'] = avars.Zcom
    row['e1'] = sim.particles[1].e
    row['e2'] = sim.particles[2].e
    row['e3'] = sim.particles[3].e
    row['m1'] = sim.particles[1].m
    row['m2'] = sim.particles[2].m
    row['m3'] = sim.particles[3].m
    return row

def quantile_1D(data, weights, quantile):
    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]
    Sn =  np.array(np.cumsum(sorted_weights))
    Pn = (Sn-0.5*sorted_weights)/Sn[-1]
    return np.interp(quantile, Pn, sorted_data)

def create_stab_hist(system, df, label, show_quantiles=True, label2="", xlabel="", nbins=50):
    if label2 == "":
        label2=label
    plt.figure(figsize=(8,4.5))
    plt.hist(df[label], density=True, bins=nbins, alpha=0.6)
    df2 = df[df["probstability"] != 0]
    n, bins, patches = plt.hist(df2[label], density=True, bins=nbins, alpha=0.6, weights=df2["probstability"])
    # plt.hist(df[label], n_bins, density=True, histtype='step', cumulative=True)
    # plt.hist(df[label], n_bins, density=True, histtype='step', cumulative=True, weights=df["probstability"])
    plt.title(system + " " + label2, size=30)
    plt.xlabel(xlabel, size=20)
    
    left, right = plt.xlim()  # return the current xlim
    left = 0
    plt.xlim(left, right)
    print("\n")
    std1 = np.std(df[label])
    weight_mean = np.average(df[label], weights=df["probstability"])
    std2 = np.sqrt(np.average((df[label]-weight_mean)**2, weights=df["probstability"]))
    print("std before: %f"%(std1))
    print("std after: %f"%(std2))
    print("factor of %f smaller"%(std1/std2))
    
    quant1 = np.quantile(df[label], 0.16)
    quant2 = np.quantile(df[label], 0.84)
    quant3 = quantile_1D(df2[label], df2["probstability"], 0.16)
    quant4 = quantile_1D(df2[label], df2["probstability"], 0.84)
    sigma1 = (quant2 - quant1) / 2
    sigma2 = (quant4 - quant3) / 2
    print("\"sigma\" before: %f"%(sigma1))
    print("\"sigma\" after: %f"%(sigma2))
    print("factor of %f smaller"%(sigma1/sigma2))
    
    if show_quantiles:
        plt.axvline(x=quant1, color="C0")
        plt.axvline(x=quant2, color="C0")
        plt.axvline(x=quant3, color="C1")
        plt.axvline(x=quant4, color="C1")
        
        kde = scipy.stats.gaussian_kde(df[label])
        xs = np.linspace(left, right, 1000)
        ys = kde(xs) + kde(-xs)
        popt1, pcov = curve_fit(gaussian, xs, ys, [0.3, 1, 3])
        print("Gaussian fit std before: ", popt1[2])
#         plt.plot(xs, ys, color="C0")
        plt.plot(xs, gaussian(xs, *popt1), color="C0")
        
        kde = scipy.stats.gaussian_kde(df2[label], weights=df2["probstability"])
        ys = kde(xs) + kde(-xs)
        popt2, pcov = curve_fit(gaussian, xs, ys, [0.3, 1, 3])
        print("Gaussian fit std after: ", popt2[2])
#         plt.plot(xs, ys, color="C1")
#         plt.plot(xs, gaussian(xs, *popt2), color="C1")
        
        print("factor of %f smaller\n"%(popt1[2]/popt2[2]))
    return n, bins
    
def create_mcmc_hist(system, err_ind, df, df2, label, show_quantiles=True, label2="", xlabel="", max_x=np.inf):
    
    print("working on " + label + "\n")
    if label2 == "":
        label2=label
    plt.figure(figsize=(8,4.5))
    n_bins=50
    plt.hist(df[label], density=True, bins=n_bins, alpha=0.6)
    plt.hist(df2[label][df2[label] < max_x], density=True, bins=n_bins, alpha=0.6)
    # plt.hist(df[label], n_bins, density=True, histtype='step', cumulative=True)
    # plt.hist(df[label], n_bins, density=True, histtype='step', cumulative=True, weights=df["probstability"])
    plt.title(system + " " + label2, size=30)
    plt.xlabel(xlabel, size=20)
    left, right = plt.xlim()  # return the current xlim
    left = 0
    right = np.minimum(right, max_x)
    plt.xlim(left, right)
    print("\n")
    std1 = np.std(df[label])
    std2 = np.std(df2[label])
    print("std before: %f\n"%(std1))
    print("std after: %f\n"%(std2))
    print("factor of %f smaller\n"%(std1/std2))
    
    quant1 = np.quantile(df[label], 0.16)
    quant2 = np.quantile(df[label], 0.84)
    quant3 = np.quantile(df2[label], 0.16)
    quant4 = np.quantile(df2[label], 0.84)
    sigma1 = (quant2 - quant1) / 2
    sigma2 = (quant4 - quant3) / 2
    print("\"sigma\" before: %f\n"%(sigma1))
    print("\"sigma\" after: %f\n"%(sigma2))
    print("factor of %f smaller\n"%(sigma1/sigma2))
    if show_quantiles:
        plt.axvline(x=quant1, color="C0")
        plt.axvline(x=quant2, color="C0")
        plt.axvline(x=quant3, color="C1")
        plt.axvline(x=quant4, color="C1")
        
        kde = scipy.stats.gaussian_kde(df[label])
        xs = np.linspace(left, right, 1000)
        ys = kde(xs) + kde(-xs)
        popt1, pcov = curve_fit(gaussian, xs, ys, [0.3, 1, 3])
        print("Gaussian fit std before: ", popt1[2], "\n")
#         plt.plot(xs, ys, color="C0")
#         plt.plot(xs, gaussian(xs, *popt1), color="C0")
        
        inds = np.sort(np.random.choice(range(len(df2)), size=5000, replace=False))
        kde = scipy.stats.gaussian_kde(df2[label][inds])
        ys = kde(xs) + kde(-xs)
        popt2, pcov = curve_fit(gaussian, xs, ys, [0.3, 1, 3])
        print("Gaussian fit std after: ", popt2[2], "\n")
#         plt.plot(xs, ys, color="C1")
#         plt.plot(xs, gaussian(xs, *popt2), color="C1")
        
        print("factor of %f smaller\n"%(popt1[2]/popt2[2]))

    plt.savefig("figs/" + system + "_%d_"%(err_ind) + label + "_mcmc_hist.png", bbox_inches="tight")
    
def plot_radvel_results(like, t):
    ti = np.linspace(np.min(t), np.max(t), 1000)
    fig = plt.gcf()
    plt.errorbar(
        like.x, like.model(t)+like.residuals(), 
        yerr=like.yerr, fmt='o'
        )
    plt.plot(ti, like.model(ti))
    plt.xlabel('Time')
    plt.ylabel('RV')
    plt.draw()

def myPriorFunc(labels, Mstar, prior_ms_small, prior_es_small, inp_list):

    prob = 1

    for i in range(3):
        secosw = inp_list[labels.index("secosw%d"%(i+1))]
        sesinw = inp_list[labels.index("sesinw%d"%(i+1))]
        e = secosw * secosw + sesinw * sesinw
        if e > 1:
            return -np.inf
        P = inp_list[labels.index("per%d"%(i+1))] / days_p_year
        k = inp_list[labels.index("k%d"%(i+1))]
        m = mass_from_VSA(P, Mstar, k, e, np.pi/2)
        prob *= (prior_ms_small[i](m) + prior_ms_small[i](-m)) * (prior_es_small[i](e) + prior_es_small[i](-e))

    return np.log(prob)
