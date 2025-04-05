#!/usr/bin/env python
import numpy as np
import h5py
import csv
import random as rd


class Static_Initial_Sample:
	def __init__(self, fonll_file_path, rapidity = 0.):
		self.pT = []
		self.dsigma = []
		self.y = rapidity

		file = open(fonll_file_path, 'r')
		reader = csv.reader(file, delimiter='\t')
		for row in reader:
			self.pT.append(float(row[0]))				# GeV
			self.dsigma.append(float(row[1]))			# mb, milli barn
		file.close()
		
		self.num = len(self.pT)
		self.pTmin = self.pT[0]
		self.pTmax = self.pT[self.num-1]
		self.d_pT = (self.pTmax - self.pTmin)/(self.num-1.)
		self.dsigmamax = max(self.dsigma)
	
	
	### ---------------- heavy quark sampling --------------- ###
	def p_HQ_sample(self, mass):	# mass in GeV
		while True:
			pT_try = rd.uniform(self.pTmin, self.pTmax)
			z_pT, i_pT = np.modf( (pT_try - self.pTmin)/self.d_pT )
			dsigma_try = rd.uniform(0.0, self.dsigmamax)
			if dsigma_try < self.dsigma[int(i_pT)]*(1.-z_pT) + self.dsigma[int(i_pT)]*z_pT:
				break
		p_try = pT_try		# GeV
		E = np.sqrt(p_try**2+mass**2)
		cos_theta = rd.uniform(-1.0, 1.0)
		sin_theta = np.sqrt(1.0-cos_theta**2)
		phi = rd.uniform(0.0, 2.0*np.pi)
		cos_phi = np.cos(phi)
		sin_phi = np.sin(phi)
		return np.array([ E, p_try*sin_theta*cos_phi, p_try*sin_theta*sin_phi, p_try*cos_theta ])
	
	
	
	### ---------------- quarkonium_sampling ---------------- ###
	def p_U1S_sample(self, mass_Q, mass_quarkonium):
		p1 = self.p_HQ_sample(mass_Q)[1:]
		p2 = self.p_HQ_sample(mass_Q)[1:]
		p = p1 + p2
		E = np.sqrt( mass_quarkonium**2 + np.sum(p**2) )
		return np.append(E, p)