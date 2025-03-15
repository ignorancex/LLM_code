import sys
sys.path.append('/Users/pedro/Dropbox/DES/Science/gmm_hmc/')

import os
import gmm_anyk as ga
import hmc_lca as hl
import hmc_physical as hp
import hmc_tapered as hr
import glob
import numpy as np 
import astropy.table as tb 

import SimplexHMC as shmc


selections = {'detached' : lambda x : x[x['CLASS'] == 'Detached'], 'ckbo' : lambda x : x[(x['CLASS'] == 'Classical') & (x['i_free'] < 5)], \
			'hkbo' : lambda x : x[(x['CLASS'] == 'Classical') & (x['i_free'] > 5)], 'classical': lambda x: x[x['CLASS'] == 'Classical'],\
			'scattering' : lambda x : x[x['CLASS'] == 'Scattering'], 'resonant' : lambda x : x[x['CLASS'] == 'Resonant'],\
			'plutino' : lambda x : x[x['Class'] == 'Resonant 3:2'], 'twotino' : lambda x : x[x['Class'] == 'Resonant 2:1'],  \
			'trojan' : lambda x : x[x['Class'] == 'Resonant 1:1'], 'fivetwo' : lambda x : x[x['Class'] == 'Resonant 5:2'],\
			'all' : lambda x : x, 'noncold'  : lambda x : x[np.logical_not((x['CLASS'] == 'Classical') & (x['i_free'] < 5))],\
            'nonscat' : lambda x : x[x['CLASS'] != 'Scattering'], 'nondet' : lambda x : x[x['CLASS'] != 'Detached'],\
			'nonscatdet' : lambda x : x[(x['CLASS'] != 'Detached') & (x['CLASS'] != 'Scattering')]}



def load_eff(obj, area, filepath):
    S0 = np.zeros(70)
    S1 = np.zeros(70)
    S2 = np.zeros(70)
    N = 0.
    for i in obj:
        a = np.load(f'{filepath}/{i}.npz')
        S0 += a['S0']#*a['N']
        S1 += a['S1']#*a['N']
        S2 += a['S2']
        N += a['N']
    return area*np.array([S0/N, S1/N, S2/N])

def load_stuff(tno, gmm_path, chains_path):
	gmm = ga.AdaptiveGMMNoise.read(f'{gmm_path}')

	f = glob.glob(f'{chains_path}/ob*.hdf5')
	data = {}
	for i in f:
	    a = tb.Table.read(i)
	    name = a[0]['MPC']
	    a = tb.Table.read(i, 'samples')
	    a = a[a['lca'] < 1]
	    data[name] = a


	chain_H = []
	chain_col = []
	chain_lca = []

	for i in tno['MPC']:
	    data[i]['g-r'] = -2.5*np.log10(data[i]['flux_g'] / data[i]['flux_r'])
	    data[i]['r-i'] = -2.5*np.log10(data[i]['flux_r'] / data[i]['flux_i'])
	    data[i]['r-z'] = -2.5*np.log10(data[i]['flux_r'] / data[i]['flux_z'])
	    data[i] = data[i][np.isfinite(data[i]['g-r'])]
	    data[i] = data[i][np.isfinite(data[i]['r-i'])]
	    data[i] = data[i][np.isfinite(data[i]['r-z'])]
	    data[i]['Hr'] = -2.5*np.log10(data[i]['flux_r']) + 30 - 10 * np.log10(30)
	    
	    chain_H.append(np.array(data[i]['Hr']))
	    chain_col.append(np.array([data[i]['g-r'], data[i]['r-i'], data[i]['r-z']]).T)
	    chain_lca.append(np.array(data[i]['lca']))

	return gmm, chain_H, chain_col, chain_lca 

def create_likelihood(tno, subsets, maxH, minH,gmm_path, chains_path, eff_path):
	
	cut = {i:selections[i](tno) for i in subsets}
	
	for i in cut:
		cut[i] = cut[i][cut[i]['H_r'] < maxH]
		cut[i] = cut[i][cut[i]['H_r'] > minH]
		cut[i]['subset'] = i


	area = {i : 2 * np.pi * (np.sin(np.max(cut[i]['i']) * np.pi/180)) * 180 * 180 / (np.pi * np.pi)	for i in cut}

	allcuts = tb.vstack([*cut.values()])

	gmm, chain_H, chain_col, chain_lca = load_stuff(allcuts, gmm_path, chains_path)

	eff = {i : load_eff(cut[i]['MPC'], area[i], eff_path) for i in cut}

	ncomp = 3 * len(subsets)

	dynclasses = {}
	j = 0
	for i in subsets:
		dynclasses[i] = j
		j+= 1 
	
	physical = []
	for i in allcuts:
		cl = dynclasses[i['subset']]
		physical.append([3*cl, 3*cl+1, 3*cl+2])
	
	joint_eff = []
	for i in subsets:
		joint_eff.append(eff[i][0])
		joint_eff.append(eff[i][1])
		joint_eff.append(eff[i][2])
	
	prior = lambda x : 0 
	prior_deriv = lambda x : np.zeros(12)

	like = hr.LogLikeEigenGMMDynamicsRolling(gmm, chain_col, chain_H, chain_lca, joint_eff, ncomp, physical, pref=-1, nphys = 3, prior = prior, prior_deriv = prior_deriv)

	for i in range(ncomp):
		like.selection[i] = like.selection[i][20:-9]
		like.selection[i][like.H_bins > maxH] = 0	
		like.selection[i][like.H_bins < minH] = 0
    
	return like, dynclasses


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--subsets', action='store', dest='subsets',type=str, nargs='*',)
	parser.add_argument('--chains_path', type=str)
	parser.add_argument('--gmm_path', type=str)
	parser.add_argument('--tno_file', type=str)
	parser.add_argument('--eff_path', type=str)	
	parser.add_argument('--save', type=str)
	parser.add_argument('--nsamps', type=int)
	parser.add_argument('--minH', type=float)
	parser.add_argument('--maxH', type=float)
	parser.add_argument('--minI', type=float, default=0)
	parser.add_argument('--maxI', type=float, default=90)


	args = parser.parse_args()
	print(args.subsets)
	tno = tb.Table.read(args.tno_file)

	like, dynclass = create_likelihood(tno, args.subsets, args.maxH, args.minH, args.gmm_path, args.chains_path, args.eff_path)

	f = np.random.dirichlet(like.ncomp * [1])
	slope1 = np.random.uniform(0.5, 1.2,)
	roll1 = np.random.uniform(-0.3, 0.3)
	Abar1 = np.random.uniform(0.03, 0.2,)
	s1 = np.random.uniform(0.7, 1.9,)
	slope2 = np.random.uniform(0.5, 1.2,)
	roll2 = np.random.uniform(-0.3, 0.3)
	Abar2 = np.random.uniform(0.03, 0.2,)
	s2 = np.random.uniform(0.7, 1.9,)
	slope3 = np.random.uniform(0.5, 1.2,)
	roll3 = np.random.uniform(-0.3, 0.3)
	Abar3 = np.random.uniform(0.03, 0.2,)
	s3 = np.random.uniform(0.7, 1.9,)

	massTheta=10000*np.identity(12)
	hmc = shmc.SimplexGeneralHMC(f, np.array([slope1, roll1, Abar1, s1, slope2,roll2, Abar2, s2, slope3, roll3, Abar3, s3]), 
                             like.likelihood, like.gradient_f, like.gradient_theta, 
                             massTheta=massTheta, dt=0.1, massF=10000.*np.identity(like.ncomp))
	print('Descending')
	hmc.descend(debug=1, learning_rate=0.00001, maxSteps=1000, minimumShift=0.01, firstPEStep=10.)
	print('First 50 steps')
	for i in range(50):
		print(i)
		hmc.sample(debug=1)
	#hmc.descend(learning_rate=0.00001, maxSteps=50, minimumShift=0.01, firstPEStep=10)
	f = []
	theta = []
	for i in range(args.nsamps):
	    st = hmc.sample(debug=0)[0]
	    print(i, hmc.f, hmc.theta)
	    if st == 0:
	        f.append(hmc.f.copy())
	        theta.append(hmc.theta.copy())
	#print(args.save)
	index = int(os.getenv('SLURM_ARRAY_TASK_ID'))
	save = f"{args.save}/{''.join(args.subsets)}_{index}.npz"
	np.savez(save, f = f, theta = theta)
