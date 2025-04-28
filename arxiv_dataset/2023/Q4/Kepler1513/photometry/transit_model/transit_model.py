import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def determine_cadence(times):
    time_gaps = {}
    for ii in range(1, len(times)):
        time_gap = np.round(times[ii]-times[ii-1], 4)
        if time_gap in time_gaps.keys():
            time_gaps[time_gap] += 1
        else:
            time_gaps[time_gap] = 1
            
    #find the key that corresponds to the most data gaps, this is the cadence
    cadence = max(time_gaps, key=time_gaps.get)
    return cadence
            
        

import matplotlib 
matplotlib.rc('xtick', labelsize=27) 
matplotlib.rc('ytick', labelsize=27) 

# load in the t0s
trans_inds = [0,1,3,4,5,6,7,8,29,31]
t0 = 277.504085
period = 160.88414
duration = 10.7536
depth = 0.006 #approx



t0s = []
for epoch in trans_inds:
    t0s.append(t0 + period*epoch)
    
t0s = np.array(t0s)    

print(t0s)

Kepler = pd.read_csv('mm_kipping_kepler.dat', delimiter='\t', header=None, names=['time', 'flux', 'flux_err'])

TESS = pd.read_csv('mm_kipping_tess.dat', delimiter=' ', header=None, names=['time', 'flux', 'flux_err'])
#BARON = pd.read_csv('mm_kipping_baron.dat', delimiter='\t', header=None, names=['time', 'flux', 'flux_err'])
BARO = pd.read_csv(
    'Kepler1513_BARO_20220715_no-detrending.csv', 
    delimiter=',', skiprows=1, names=['time', 'flux', 'flux_err', 'airmass'])
BARON = pd.read_csv(
    'Kepler1513_BARON_20220715_no-detrending.csv', 
    delimiter=',', skiprows=1, names=['time', 'flux', 'flux_err', 'airmass'])



LCO = pd.read_csv(
    'June2obs/TIC394177315-11_20230602_LCO-Teid-1m0_ip_5px_KC_bjd-flux-err-fwhm-sky-detrended.dat', 
    delimiter='\t', skiprows=1, names=['time', 'flux', 'flux_err'])
Whitin = pd.read_csv(
    'June2obs/TIC394177315-11_UT20230602_Whitin_R.ap7_datasubset.dat',
    delimiter='\t', skiprows=1, names=['time','flux','flux_err','airmass'])



BARON['time'] -= 2454833
BARO['time'] -= 2454833
TESS['time']+= 2400000-2454833

#From Kim McLeod at Whitin: 
#For the final fit, it might be worth cutting out the data at the end where the 
#positions begin to shift due to the rotator hitting its limit, 
#and then at the very end twilight appears in the sky counts.

plt.figure(figsize=[18,10])
plt.errorbar(Whitin['time']-2454833, Whitin['flux'], yerr=Whitin['flux_err'], marker='o', color='k', ls='')
plt.axvline(5264.84, 0, 1, color='r')

plt.figure(figsize=[18,10])
plt.errorbar(Whitin['time']-2454833, Whitin['flux'], yerr=Whitin['flux_err'], marker='o', color='k', ls='')
plt.axvline(5264.84, 0, 1, color='r')
plt.xlim(5264.76, 5264.86)

#cutoff Whitin data after 2460097.84
#df.drop(df[df.score < 50].index, inplace=True)

Whitin_cut = Whitin.drop(Whitin[Whitin['time']>2460097.84].index)



from scipy.optimize import curve_fit

def line(x, A, B): # this is your 'straight line' y=f(x)
    return A*x + B

plt.errorbar(TESS['time'], TESS['flux'], TESS['flux_err'], color='b', ls='', marker='o')
#plt.errorbar(BARON['time'], BARON['flux'], BARON['flux_err'], color='k', ls='', marker='o')


ingress = 4942.91
plt.axvline(ingress, color='r')

plt.xlim(4942.75, 4943.25)

plt.show()



BARON_pre_ingress = BARON.loc[BARON['time'] < ingress]


plt.errorbar(BARON['airmass'], BARON['flux'], BARON['flux_err'], color='k', ls='', marker='o')
plt.show()



baron_line,_ = curve_fit(line, BARON_pre_ingress['airmass'], BARON_pre_ingress['flux'], sigma=BARON_pre_ingress['flux_err']) 


x_plot = np.linspace(np.min(BARON['airmass']), np.max(BARON['airmass']), 100)
x_plot_pre_ingress = np.linspace(np.min(BARON_pre_ingress['airmass']), np.max(BARON_pre_ingress['airmass']), 100)

line_plot = line(x_plot, baron_line[0], baron_line[1])
line_plot = line(x_plot_pre_ingress, baron_line[0], baron_line[1])

line_fit = line(BARON['airmass'], baron_line[0], baron_line[1])


plt.errorbar(BARON_pre_ingress['airmass'], BARON_pre_ingress['flux'], BARON_pre_ingress['flux_err'], color='k', ls='', marker='o')
plt.plot(x_plot_pre_ingress, line_plot, 'g')


plt.show()

plt.errorbar(BARON['airmass'], BARON['flux'], BARON['flux_err'], color='k', ls='', marker='o')
plt.plot(x_plot, line_plot, 'g')


plt.show()
plt.figure()
baron_flux_detrended_airmass = BARON['flux']/line_fit
plt.errorbar(BARON['airmass'], baron_flux_detrended_airmass, BARON['flux_err'], color='k', ls='', marker='o')



plt.show()



plt.errorbar(BARON['time'], baron_flux_detrended_airmass, yerr=BARON['flux_err'], marker='o', color='k', ls='')

plt.errorbar(TESS['time'], TESS['flux'], yerr=TESS['flux_err'], color='b', ls='', marker='o')

plt.axvline(ingress, color='r')

plt.xlim(4942.75, 4943.25)
plt.show()





plt.errorbar(LCO['time'], LCO['flux'], yerr=LCO['flux_err'], color='b', ls='', marker='o')
plt.axvline(2460097.69, 0, 1, color='r')
plt.show()

plt.errorbar(Whitin_cut['airmass'], Whitin_cut['flux'], Whitin_cut['flux_err'], color='k', ls='', marker='o')
plt.show()

whitin_pre_ingress = Whitin_cut.loc[Whitin_cut['time'] < 2460097.69]

whitin_line,_ = curve_fit(line, whitin_pre_ingress['airmass'], whitin_pre_ingress['flux'], sigma=whitin_pre_ingress['flux_err']) 


x_plot = np.linspace(np.min(Whitin_cut['airmass']), np.max(Whitin_cut['airmass']), 100)

line_plot = line(x_plot, whitin_line[0], whitin_line[1])

line_fit = line(Whitin_cut['airmass'], whitin_line[0], whitin_line[1])


plt.errorbar(Whitin_cut['airmass'], Whitin_cut['flux'], Whitin_cut['flux_err'], color='k', ls='', marker='o')
plt.plot(x_plot, line_plot, 'g')


plt.show()

plt.figure()
whitin_flux_detrended_airmass = Whitin_cut['flux']/line_fit
plt.errorbar(Whitin_cut['airmass'], whitin_flux_detrended_airmass, Whitin_cut['flux_err'], color='k', ls='', marker='o')



plt.show()


plt.errorbar(Whitin_cut['time'], whitin_flux_detrended_airmass, yerr=Whitin_cut['flux_err'], marker='o', color='k', ls='')

plt.errorbar(LCO['time'], LCO['flux'], yerr=LCO['flux_err'], color='b', ls='', marker='o')


#plt.xlim(5262, 5264.91676768)
plt.show()







x_kepler, y_kepler, yerr_kepler = Kepler['time'].values+2400000-2454833, Kepler['flux'].values-1, Kepler['flux_err'].values

x_tess, y_tess, yerr_tess = TESS['time'].values, TESS['flux'].values-1, TESS['flux_err'].values
#x_baron, y_baron, yerr_baron = BARON['time'].values, BARON['flux'].values-1, BARON['flux_err'].values
x_baron, y_baron, yerr_baron = BARON['time'].values, baron_flux_detrended_airmass.values-1, BARON['flux_err'].values

x_lco, y_lco, yerr_lco = LCO['time'].values - 2454833., LCO['flux'].values-1, LCO['flux_err'].values
x_whitin, y_whitin, yerr_whitin = Whitin_cut['time'].values - 2454833., whitin_flux_detrended_airmass.values-1, Whitin_cut['flux_err'].values



from collections import OrderedDict

texp_kepler = determine_cadence(x_kepler)

texp_baron = determine_cadence(x_baron)
texp_tess = determine_cadence(x_tess)

texp_lco = determine_cadence(x_lco)
texp_whitin = determine_cadence(x_whitin)


datasets = OrderedDict(
    [
        ("Kepler", [x_kepler, y_kepler, yerr_kepler, texp_kepler]),
        ("BARON", [x_baron, y_baron, yerr_baron, texp_baron]),
        ("TESS", [x_tess, y_tess, yerr_tess, texp_tess]),
        ("LCO Teide", [x_lco, y_lco, yerr_lco, texp_lco]),
        ("Whitin", [x_whitin, y_whitin, yerr_whitin, texp_whitin])
    ]
)

print(t0s[0])
datasets

for n, (name, (x, y, yerr, texp)) in enumerate(datasets.items()):
    df = pd.DataFrame(
        {'x': x,
         'y': y+1,
         'yerr': yerr
        })
    
    df.to_csv('./detrended_lcs/' + name + '.dat', index=False, header=False, sep='\t')

t0s

#find reasonable limb darkening params for ground based obs
#𝑇eff = 5491 ± 100 [K]
#logg = 4.46 ± 0.10 [cgs]
#[Fe/H] = 0.17 ± 0.06

limb_darkening = pd.read_csv('claret_limb_darkening.tsv', delimiter='\t', skiprows=39)


#limit by logg
limb_darkening = limb_darkening.loc[limb_darkening['logg [cm/s2]'] > 4.46-.1]
limb_darkening = limb_darkening.loc[limb_darkening['logg [cm/s2]'] < 4.46+.1]

#limit by Teff
limb_darkening = limb_darkening.loc[limb_darkening['Teff [K]'] > 5491-100]
limb_darkening = limb_darkening.loc[limb_darkening['Teff [K]'] < 5491+100]
limb_darkening

#limit by fe/h
limb_darkening = limb_darkening.loc[limb_darkening['Z [Sun]'] > 0.17-0.06]
limb_darkening = limb_darkening.loc[limb_darkening['Z [Sun]'] < 0.17+0.06]
limb_darkening

#BARON observed in R band
BARON_limb_darkening = limb_darkening.loc[limb_darkening['Filt'] == 'i*']
BARON_u1 = float(BARON_limb_darkening['a'].values)
BARON_u2 = float(BARON_limb_darkening['b'].values)
print('BARON')
print(BARON_u1, BARON_u2)
print('')

#LCO observed in ip band
LCO_limb_darkening = limb_darkening.loc[limb_darkening['Filt'] == 'i*']
LCO_u1 = float(LCO_limb_darkening['a'].values)
LCO_u2 = float(LCO_limb_darkening['b'].values)
print('LCO')
print(LCO_u1, LCO_u2)
print('')

#Whitin observed in R band
Whitin_limb_darkening = limb_darkening.loc[limb_darkening['Filt'] == 'R ']
Whitin_u1 = float(Whitin_limb_darkening['a'].values)
Whitin_u2 = float(Whitin_limb_darkening['b'].values)
print('Whitin')
print(Whitin_u1, Whitin_u2)
print('')


import pymc3 as pm
import pymc3_ext as pmx
import exoplanet as xo
import aesara_theano_fallback.tensor as tt
from functools import partial
from celerite2.theano import terms, GaussianProcess
import theano




# Find a reference transit time near the middle of the observations to avoid
# strong covariances between period and t0
x_min = min(np.min(x) for x, _, _, _ in datasets.values())
x_max = max(np.max(x) for x, _, _, _ in datasets.values())
x_mid = 0.5 * (x_min + x_max)

mid_epoch = np.round((x_mid) / period)
t0_ref = period * np.round((x_mid - t0s[0]) / period)


with pm.Model() as model:

    # Shared orbital parameters --> impact parameter, transit times, stellar density
    ###########################
    ###########################
    ###########################
    #stellar density
    log_rho_star = pm.Uniform('log_rho_star', lower=-3, upper=3)
    rho_star = pm.Deterministic('rho_star', tt.exp(log_rho_star))
    
    # impact parameter
    b = pm.Uniform("b", lower=0, upper=2)

    
    # Now we have a parameter for each transit time for each planet:
    transit_times = []
    for i in range(1):
        transit_times.append(
            pm.Uniform(
                "tts_{0}".format(i),
                lower=t0s-1,
                upper=t0s+1,
                shape=len(t0s),
            )
        )

        
    # Now we have a parameter for each transit time for each planet:
    transit_inds = pm.Deterministic("transit_inds", tt.constant(trans_inds))
    
    
    
    
    # Set up an orbit for the planet
    orbit = xo.orbits.TTVOrbit(b=b, transit_times=transit_times, 
                               rho_star=rho_star, transit_inds=[transit_inds])
    
    #rp_over_rstar 
    ror = pm.Uniform('ror', lower=0, upper=1)
    
    #fix r_star = 1, then can solve for rp/rstar after
    r_star = 1.
    r_pl = pm.Deterministic("r_pl", ror * r_star)
    
    
    
    # It will be useful later to track some parameters of the orbit
    pm.Deterministic("t0", orbit.t0)
    pm.Deterministic("period", orbit.period)
    pm.Deterministic("ttvs_{0}".format(i), orbit.ttvs[i])
    
    




    # not shared parameters --> depth and limb-darkening
    # Loop over the instruments
    parameters = dict()
    lc_models = dict()
    for n, (name, (x, y, yerr, texp)) in enumerate(datasets.items()):
        # We define the per-instrument parameters in a submodel so that we
        # don't have to prefix the names manually
        with pm.Model(name=name, model=model):
            
            #if groundbased, assume a claret limb-darkening
            if name == 'BARON':
                u1, u2 = BARON_u1, BARON_u2
            elif name == 'LCO Teide':
                u1, u2 = LCO_u1, LCO_u2
            elif name == 'Whitin':
                u1, u2 = Whitin_u1, Whitin_u2
            
            # Else, use the limb darkening from equations 15-19 in Kipping 2013
            # https://arxiv.org/pdf/1308.0009.pdf
            else:
                q1 = pm.Uniform('q1', lower=0., upper=1.)
                q2 = pm.Uniform('q2', lower=0., upper=1.)
            
            
            
                u1 = pm.Deterministic('u1', 2*tt.sqrt(q1)*q2)
                u2 = pm.Deterministic('u2', tt.sqrt(q1)*(1-(2*q2)))

            star = xo.LimbDarkLightCurve(u1=u1, u2=u2)
            

            
            #jitter term
            #med_yerr = np.median(yerr)
            #std = np.std(y)
            #jitter = pm.InverseGamma(
            #    "jitter",
            #    testval=med_yerr,
            #    **pmx.estimate_inverse_gamma_parameters(
            #        med_yerr, 0.5 * std
            #    ),
            #)
                
                
            #add TESS blend facotr
            if name == 'TESS':
                log_blend = pm.Uniform('log_blend', lower=tt.log(1), upper=tt.log(10), testval=tt.log(1.01))
                blend = pm.Deterministic('blend', tt.exp(log_blend))
                y = (y+1)/blend - 1
            
    

            # Keep track of the parameters for optimization
            if name == 'TESS':
                parameters[name] = [q1, q2, r_pl, blend]
            elif name == 'Kepler':
                parameters[name] = [q1, q2, r_pl]
            else:
                parameters[name] = [r_pl]
            

        # The light curve model
        def lc_model(star, r_pl, texp, t):
            return pm.math.sum(
                star.get_light_curve(orbit=orbit, r=r_pl*r_star, t=t, texp=texp),
                axis=-1,
            )

        lc_model_partial = partial(lc_model, star, r_pl, texp)
        lc_models[name] = lc_model_partial
        
        lc_model_obs = lc_model(star, r_pl, texp, x)
        pm.Normal(f"{name}_obs", mu=lc_model_obs, sd=np.sqrt(yerr**2.), observed=y)



    # Optimize the model
    map_soln = model.test_point
    #for name in ['Kepler', 'TESS']:
    #    map_soln = pmx.optimize(map_soln, parameters[name])
    for name in ['Kepler']:
        map_soln = pmx.optimize(map_soln, parameters[name] + [b])
    map_soln = pmx.optimize(map_soln)

transit_times_map = map_soln['tts_0']
blend_tess_map = map_soln['TESS_blend']
period_map = map_soln['period']

print(transit_times_map)
print(blend_tess_map)
print(period_map)


#print('')
#print('')
#print('')
#for n, (name, (x, y, yerr, texp)) in enumerate(datasets.items()):
#    print(name)
#    print('median yerr: ' + str(np.median(yerr)))
#    print('jitter: ' + str(map_soln[name+'_jitter']))
#    print('')


map_soln['t0']

import matplotlib
matplotlib.rc('xtick', labelsize=27) 
matplotlib.rc('ytick', labelsize=27) 
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)

dt = np.linspace(-.5, .5, 500)


#colors = ["#2E2252", "#ab0048", "#723c87", '#2E5090', '#5164ff']
#colors = ["#b30000", "#4421af", "#7c1158", "#1a53ff", "#0d88e6"]
colors = ["#9350a1", "#697ed5", "#6fac5d", "#bc7d39", "#b94663"]
fig, ax = plt.subplots(2, figsize = [18,13], gridspec_kw={'height_ratios': [7, 2]}, sharex=True)

index = 0

shifts = [-0.05, 0., -0.025, 0.025, 0.05]
#trans_inds = [0,1,3,4,5,6,7,8,29,31]
epochs_labels = ['epochs 1,\,2,\,4,\,5,\,6,\,7,\,8,\,9', 
                 'epoch 30', 'epoch 30', 'epoch 32', 'epoch 32']
with model:
    
    
    for n, (name, (x, y, yerr, texp)) in enumerate(datasets.items()):
        phase_curve = pmx.eval_in_model(
                lc_models[name](transit_times_map[0] + dt), map_soln)
        
        phase_curve_for_resid = pmx.eval_in_model(
                lc_models[name](x), map_soln)
        
        t_warp = pmx.eval_in_model(orbit._warp_times(x), map_soln)
        
        
        #jitter = map_soln[name+'_jitter']
        #error = np.sqrt(yerr**2.+jitter**2.)
        
        if name == 'TESS':
                y = (y+1)/blend_tess_map - 1

        # Get the map period for plotting purposes
        p = period_map

        # Plot the folded data
        x_fold = (t_warp + 0.5 * p) % p - 0.5 * p
        ax[0].errorbar(
            x_fold, y+shifts[index], yerr=yerr, ms=9,
            marker='o', ls='', color=colors[index], zorder=-1000, alpha=0.7
        )


        ax[0].text(-0.49, shifts[index]+.005, name, fontsize = 27, color=colors[index])
        ax[0].text(0.49, shifts[index]+.005, epochs_labels[index], 
                 fontsize = 27, color=colors[index], horizontalalignment= 'right')




        ax[0].plot(dt, phase_curve+shifts[index], color='k', lw=2)
        
        
        
        if name == 'Whitin':
            ax[1].errorbar(
                x_fold, (y-phase_curve_for_resid)*1000, yerr=yerr*1000., ms=9,
                marker='o', ls='', color=colors[index], zorder=-5000, alpha=0.5
            )
        else:
            ax[1].errorbar(
                x_fold, (y-phase_curve_for_resid)*1000, yerr=yerr*1000., ms=9,
                marker='o', ls='', color=colors[index], zorder=-1000, alpha=0.7
            )

        ax[1].axhline(0, 0, 1, color='k', lw=2)

        index+=1
    
    
ax[0].axhline(0.072, 0.74, 0.78, color = 'k', lw=2)   
ax[0].text(0.47, 0.07, 'MAP transit model', fontsize = 27, color='k', horizontalalignment= 'right')
ax[0].set_xlim(-0.5, 0.5)
ax[0].set_ylim(-0.075, 0.08)
ax[0].set_ylabel("relative intensity", fontsize = 36)


#handles, labels = fig.gca().get_legend_handles_labels()
#by_label = dict(zip(labels, handles))
#ax[0].legend(by_label.values(), by_label.keys(), fontsize=23, loc=1)





ax[1].set_xlim(-0.5, 0.5)

ax[1].set_ylabel("residuals [ppt]", fontsize = 36)
ax[1].set_xlabel(r"time from $\textrm{T}_0$ [days]", fontsize = 36)


fig.tight_layout()
fig.savefig('transit_model_new.pdf')
    

plt.show()

with model:
    trace = pmx.sample(
        tune=2000,
        draws=2000,
        start=map_soln,
        cores=2,
        chains=2,
        target_accept=.99,
        return_inferencedata=True,
    )