"""
All plots that test the application of the two-scale method to simulations, with the figure sizes modified to suit ApJ.
"""

import sys
import pathlib
root = pathlib.Path(__file__).parent.parent
sys.path.append(str(root))

import matplotlib.pyplot as plt
import matplotlib as mpl
import pencil as pc
import numpy as np
import scipy.fft
import os

from spectrum import calc_spec_G2 as calc_spec, signed_loglog_plot
from utils import fig_saver, real, rebin

class SpecFromSim:
	def __init__(self, sim, double_domain=False, shift_onesided=0, var_file=''):
		if not hasattr(sim, 'var'):
			sim.var = pc.read.var(
				sim=sim,
				var_file=var_file,
				trimall=True,
				quiet=True,
				magic=['bb'],
				)
		
		var = sim.var
		grid = sim.grid
		
		res = []
		for ix in np.round(np.linspace(0,sim.dim.nx-1,7))[:-1].astype(int):
			Bvec = var.bb[:,:,:,ix]
			Bvec = np.swapaxes(Bvec, 1,2)
			
			if double_domain:
				if not sim.dim.nz%2 == 0:
					raise RuntimeError("nz is odd.")
				nzb2 = int(sim.dim.nz/2)
				Bvec = np.concatenate((Bvec[..., nzb2:], Bvec, Bvec[..., :nzb2]), axis=2) #double along the z direction
				Bvec = np.concatenate((Bvec, Bvec), axis=1) #double along the y direction
			
			Bvec_fft = scipy.fft.fft2(Bvec, norm='forward', axes=(-2,-1))
			
			L = [grid.Ly, grid.Lz]
			k, E0, H0 = calc_spec(
				Bvec_fft,
				K=np.array([0,0]),
				L=L,
				)
			
			if double_domain:
				K = 2
			else:
				K = 1
			
			_, _, H1 = calc_spec(
				Bvec_fft,
				K=np.array([0,K]),
				L=L,
				shift_onesided=shift_onesided,
				)
			
			res.append({'k':k, 'E0': E0, 'H0': H0, 'H1': H1})
		
		
		H0av = np.average(np.array([d['H0'] for d in res]), axis=0)
		H1av = np.average(np.array([d['H1'] for d in res]), axis=0)
		E0av = np.average(np.array([d['E0'] for d in res]), axis=0)
		k = res[0]['k']
		
		if double_domain:
			k_old = k
			dk = k_old[1] - k_old[0] #assumes k are equispaced.
			k = k_old[::2]
			bin_bounds = np.linspace(-dk,k_old[-1]+dk,len(k)+1)
			
			H0av = rebin(k_old, H0av, bin_bounds)
			H1av = rebin(k_old, H1av, bin_bounds)
			E0av = rebin(k_old, E0av, bin_bounds)
		
		self.k = k
		self.H0av = H0av
		self.H1av = H1av
		self.E0av = E0av

def plot(
	sim,
	figname,
	H_getter,
	H_label,
	saver,
	**kwargs,
	):
	"""
	Arguments:
		sim: Pencil simulation object
		figname: str
		H_getter: Function that takes in a SpecFromSim object and returns the part of the helicity spectrum that should be plotted
		H_label: label to use for the helicity spectrum in the plot
		saver: utils.fig_saver instance
		double_domain: bool
		shift_onesided: int
	"""
	spec = SpecFromSim(sim, **kwargs)
	
	av = sim.av
	grid = sim.grid
	
	fig,axs = plt.subplots(nrows=2, gridspec_kw={'height_ratios':[1,4]})
	
	axs[0].plot(grid.z, av.xy.abmz[-1])
	axs[0].set_xlabel(r"$z$")
	axs[0].set_ylabel(r"$\left< \vec{A}\cdot\vec{B} \right>_{xy}$")
	axs[0].set_xlim(min(grid.z), max(grid.z))
	axs[0].axhline(0, ls=':', c='k')
	
	handles = []
	handles.extend(signed_loglog_plot(
		spec.k,
		H_getter(spec),
		axs[1],
		{'label':H_label},
		))
	handles.extend(axs[1].loglog(
		spec.k,
		real(spec.E0av),
		label=r"$\widetilde{E}(k,0)$",
		))
	axs[1].legend(handles=handles)
	axs[1].set_xlabel("k")
	
	fig.set_size_inches(3,3.5) #NOTE: ApJ linewidth is 3.3 inches in twocolumn layout.
	saver(fig, figname)

if __name__ == "__main__":
	mpl.style.use(root/"kishore_apj.mplstyle")
	varname = "var.h5" #Name of snapshot to use
	saver = fig_saver(
		savefig=True,
		savedir=root/"plots/check_helspec_from_sims", #Where to save plots
		)
	sim_1 = pc.sim.get(quiet=True, path="1")
	sim_2 = pc.sim.get(quiet=True, path="2")
	
	for sim in [sim_1, sim_2]:
		sim.av = pc.read.aver(quiet=True, datadir=sim.datadir, simdir=sim.path)
		sim.var = pc.read.var(sim=sim, var_file=varname, trimall=True, quiet=True, magic='bb')
	
	#no sign flip of helicity
	plot(
		sim_2,
		"2.pdf",
		lambda spec: spec.k*np.real(spec.H0av),
		r"$\mathrm{Re}(k\,\widetilde{H}(k,0))$",
		saver,
		var_file=varname,
		)
	
	#sign flip of helicity, BPS17 method
	plot(
		sim_1,
		"1_shift_by_one.pdf",
		lambda spec: -spec.k*np.imag(spec.H1av),
		r"$-\mathrm{Im}(k\,\widetilde{H}(k,K_1))$",
		saver,
		shift_onesided=1,
		var_file=varname,
		)
	
	#sign flip of helicity with domain doubling
	plot(
		sim_1,
		"1_doubled.pdf",
		lambda spec: -spec.k*np.imag(spec.H1av),
		r"$-\mathrm{Im}(k\,\widetilde{H}(k,K_1))$",
		saver,
		double_domain=True,
		var_file=varname,
		)
