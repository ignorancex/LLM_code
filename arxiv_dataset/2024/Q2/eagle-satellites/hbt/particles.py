import h5py
from icecream import ic
from matplotlib import pyplot as plt, ticker
from matplotlib.collections import LineCollection
from multiprocessing import Pool
import numpy as np
import os
import sys
from time import time
from tqdm import tqdm
import warnings

from plottery.plotutils import savefig, update_rcParams
update_rcParams()

from HBTReader import HBTReader
from hbtpy import hbt_tools
from hbtpy.hbt_tools import save_plot
from hbtpy.simulation import Simulation
from hbtpy.subhalo import HostHalos, Subhalos
from hbtpy.track import Track, HaloTracks
from hbtpy.helpers.plot_auxiliaries import (
    get_axlabel, get_label_bincol, plot_line)
from hbtpy.helpers.plot_definitions import xbins


"""
TO-DO:
    -Find a couple galaxies in each bin
    -Load their particles, calculate (stellar) mass profiles over time
"""

def main():
    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation, args.root)
    reader = HBTReader(sim.path)

    isnap = -1
    subs = Subhalos(
        reader.LoadSubhalos(isnap), sim, isnap, logMmin=9,
        logM200Mean_min=13, logMstar_min=9)
    ic(np.sort(subs.colnames))

    test = 100
    ic(subs[['TrackId','Nbound','Ndm','Nstar','Ngas',
             'Mbound','Mdm','Mstar','Mgas','M200Mean']].iloc[test])
    ndm = subs['Ndm'].iloc[test]
    nstar = subs['Nstar'].iloc[test]
    ngas = subs['Ngas'].iloc[test]
    # GetParticleProperties doesn't work
    particles = reader.LoadParticles(subindex=subs['TrackId'][test])
    ic(particles, particles.min()/1e10, particles.max()/1e10, particles.size)
    # three structured arrays to store data for each particle type
    dm = np.array(
        np.zeros((ndm,3)),
        dtype=[('x', float), ('y', float), ('z', float)])
    ic(dm)

    ppath = os.path.join(
        os.environ['EAGLE'], sim.name.split('/')[1], 'simu',
        'particledata_028_z000p000')
    while ((dm['x'] > 0) < ndm):
    for pfile in range():
        ic(pfile)
        file = os.path.join(
            ppath, f'eagle_subfind_particles_028_z000p000.{pfile}.hdf5')
        with h5py.File(file, 'r') as f:
            if pfile == 0:
                for key in f.keys():
                    ic(key, f[key].keys())
            ic(f['PartType4']['Coordinates'][...])
            for i in '0145':
                pid = f[f'PartType{i}']['ParticleIDs'][...]
                found = np.isin(particles, pid)
                ic(i, pid.size/1e6, pid.min()/1e10, pid.max()/1e10, found.sum())
        print()
    return

    event = 'history:first_infall'
    stellar_mass_gained = subs[f'Mstar/{event}:Mstar']
    fig, ax = plt.subplots(constrained_layout=True)
    gain_bins = np.logspace(-2, 1.7, 50)
    ax.hist(stellar_mass_gained, gain_bins, color='C8', log=True,
            histtype='stepfilled')
    # same as Fig 8
    mbound_infall_bins = np.logspace(10.5, 12.5, 5)
    ic(mbound_infall_bins)
    mbins = mbound_infall_bins
    mcol = f'{event}:Mbound'
    for i in range(1, mbins.size):
        j = (subs[mcol] >= mbins[i-1]) & (subs[mcol] < mbins[i])
        label = f'{get_label_bincol(mcol)} bin {i}'
        ax.hist(stellar_mass_gained[j], gain_bins, color=f'C{i}', lw=2,
                histtype='step', label=label)
    ax.legend()
    ax.set(xlabel=get_axlabel(f'Mstar/{event}:Mstar', 'mean'),
           ylabel='N', xscale='log')
    output = os.path.join(sim.plot_path, 'particles')
    os.makedirs(output, exist_ok=True)
    output = os.path.join(output, 'hist_stellar_mass_gained.pdf')
    savefig(output, fig=fig, tight=False)


main()
