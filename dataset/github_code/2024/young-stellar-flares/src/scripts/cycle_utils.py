import numpy as np
import matplotlib.pyplot as plot
from altaipony.ffd import FFD
from astropy import units

__all__ = ['calculate_fr', 'calculate_xi', 'cumulative_ffd', 'sine_wave']

def calculate_xi(table, time, lim, cmap, ax=None, color=None):
    tot = (len(time[(time < lim[1]) & (time > lim[0])] * 2)*units.min).to(units.day)
    midtime = np.nanmedian(time[(time < lim[1]) & (time > lim[0])])

    if ax is None:
        return np.log10(np.nansum(table['ed'])/2.0), midtime
    else:
        ax.plot(midtime,
                np.log10(np.nansum(table['ed'])/2.0),
                'o', ms=12, c=cmap[color], markeredgecolor='k')
        return np.log10(np.nansum(table['ed'])/2.0), midtime

def calculate_fr(table, time, lim, cmap, ax=None, color=None):
    tot = (len(time[(time < lim[1]) & (time > lim[0])] * 2)*units.min).to(units.day)
    midtime = np.nanmedian(time[(time < lim[1]) & (time > lim[0])])
    if ax is None:
        return np.log10(np.nansum(table['prob'])/tot.value), midtime
    else:
        ax.plot(midtime,
                np.log10(np.nansum(table['prob'])/tot.value),
                'o', ms=12, c=cmap[color], markeredgecolor='k')
        return np.log10(np.nansum(table['prob'])/tot.value), midtime

def cumulative_ffd(table, ed_key='ed', eng_key='flare_energy'):
    newtab = table.copy()

    try:
        tot = table['total_obs_time'][0]
        newtab.rename_column('prob', 'recovery_probability')
    except:
        tot = 365.0

    newtab.rename_column(eng_key, 'ed_corr')
    newtab.rename_column(ed_key, 'ed_rec')
    newtab = newtab.to_pandas()
    simple_ffd = FFD(f=newtab, tot_obs_time=tot)

    if ed_key != 'dur':
        simple_ffd.ID = 'Target_ID'

        ed, freq, counts = simple_ffd.ed_and_freq(energy_correction=True,
                                                  multiple_stars=True,
                                                  recovery_probability_correction=True)
    else:
        ed, freq, counts = simple_ffd.ed_and_freq(energy_correction=True)
    return ed, freq

def sine_wave(t, A, P, phi, B):
    f = 1.0/P
    return A * np.sin(2*np.pi*f*t + phi) + B
