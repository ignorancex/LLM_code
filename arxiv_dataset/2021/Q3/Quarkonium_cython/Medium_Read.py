#!/usr/bin/env python
import h5py
import numpy as np


def interpolate_3d(table, r1, r2, r3, i1, i2, i3, max1, max2, max3):
	# 1 = tau, 2 = x, 3 = y
	# r = where in between, i = index, max = max index
	w1 = [1.0-r1, r1]
	w2 = [1.0-r2, r2]
	w3 = [1.0-r3, r3]
	interpolation = 0.0
	for i in range(2):
		for j in range(2):
			for k in range(2):
				interpolation += table[(int(i1)+i)%max1][(int(i2)+j)%max2][(int(i3)+k)%max3] *w1[i]*w2[j]*w3[k]
	return interpolation




class hydro_reader:
	def __init__(self, hydro_file_path):
		hydrofilename = hydro_file_path
		if hydrofilename == None:
			raise ValueError("Need hydro history file")
		else:
			self.file = h5py.File(hydrofilename, 'r')

		self.step_keys = list(self.file['Event'].keys())	# at every tau step, tau^2=t^2-z^2
		self.Nstep = len(self.step_keys)
		self.info_keys = list(self.file['Event'][self.step_keys[0]].keys())
		self.Nx = self.file['Event'].attrs['XH'][0] - self.file['Event'].attrs['XL'][0] + 1
		self.Ny = self.file['Event'].attrs['YH'][0] - self.file['Event'].attrs['YL'][0] + 1
		self.dx = self.file['Event'].attrs['DX'][0]
		self.dy = self.file['Event'].attrs['DY'][0]
		self.dtau = self.file['Event'].attrs['dTau'][0]
		self.tau0 = self.file['Event'].attrs['Tau0'][0]
		self.tauf = self.tau0 + (self.Nstep-1.) * self.dtau
		self.xmin = self.file['Event'].attrs['XL'][0]*self.dx
		self.xmax = self.file['Event'].attrs['XH'][0]*self.dx
		self.ymin = self.file['Event'].attrs['YL'][0]*self.dy
		self.ymax = self.file['Event'].attrs['YH'][0]*self.dy
		self.T = []
		self.Vx = []
		self.Vy = []
		
		for i in range(self.Nstep):
			self.T.append( self.file['Event'][self.step_keys[i]]['Temp'].value )
			self.Vx.append( self.file['Event'][self.step_keys[i]]['Vx'].value )
			self.Vy.append( self.file['Event'][self.step_keys[i]]['Vy'].value )
		self.file.close()	
		
		
	
	def cell_info(self, time, position_vector):
		x = position_vector[0]
		y = position_vector[1]
		z = position_vector[2]
		#print time, z
		vz = z/time
		gamma_inverse = np.sqrt(1.0 - vz**2)
		tau = time * gamma_inverse
		
		if tau < self.tau0:
			return np.array([0.0, 0.0, 0.0, 0.0])
		else:
			tau = np.min( [ np.max([tau, self.tau0]), self.tauf ] )
			x = np.min( [ np.max([x, self.xmin]), self.xmax ] )
			y = np.min( [ np.max([y, self.ymin]), self.ymax ] )
			
			r_tau, i_tau = np.modf( (tau - self.tau0)/self.dtau )	# fractional and integer parts
			r_x, i_x = np.modf( (x-self.xmin)/self.dx )
			r_y, i_y = np.modf( (y-self.ymin)/self.dy )
			temp = interpolate_3d(self.T, r_tau, r_x, r_y, i_tau, i_x, i_y, self.Nstep, self.Nx, self.Ny)
			vx = interpolate_3d(self.Vx, r_tau, r_x, r_y, i_tau, i_x, i_y, self.Nstep, self.Nx, self.Ny)
			vy = interpolate_3d(self.Vy, r_tau, r_x, r_y, i_tau, i_x, i_y, self.Nstep, self.Nx, self.Ny)
		
			vx = gamma_inverse * vx
			vy = gamma_inverse * vy
			
			return np.array([temp, vx, vy, vz])
			
	
			
			
			
			
			
			
			
		