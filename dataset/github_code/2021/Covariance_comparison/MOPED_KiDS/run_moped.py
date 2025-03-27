import os
import numpy as np
from scipy.linalg import null_space
import configparser


class MOPED(object):
	def __init__(self, transform, parameters, delta):
		self.input_dir = 'moped_input/scale_cuts_output/'
		self.values_dir = 'moped_values.ini'
		self.transform = transform
		self.fiducial_theory = (np.loadtxt(self.input_dir+'theory.txt')).T
		self.delta = delta
		self.parameters = parameters
		self.config = configparser.RawConfigParser()
		self.config.read(self.values_dir)
		self.data, self.cov, self.cov_inverse = self.get_data()
		self.weight = self.get_weight()
		self.cov_c, self.y_d = self.get_transformed()

	def get_params(self):
		param_ini = {}
		for sec in self.parameters.keys():
			for i in range(len(self.parameters[sec])):
				param_ini[sec+'-'+self.parameters[sec][i]] = \
					self.config[sec][self.parameters[sec][i]]
		return param_ini

	def set_params(self, **kwargs):
		for sec in self.parameters.keys():
			for sp in kwargs.keys():
				if sec in sp:
					self.config[sec][sp[len(sec)+1:]] = str(kwargs[sp])
		with open(self.values_dir, 'w') as configfile:
			self.config.write(configfile)

	def run_pipeline(self):
		der_vals = {}
		self.params = self.get_params()
		for n in self.params.keys():
			param_value = float(self.params[n])
			delta = np.abs(self.delta*param_value)
			values = [param_value-delta, param_value+delta]
			der_fid = []
			for v in values:
				perturbed_values = {n: str(v)}
				self.set_params(**perturbed_values)
				os.system('cosmosis moped_params.ini')
				der_fid.append(np.loadtxt(self.input_dir+'theory.txt'))
			der_fid = np.asarray(der_fid)
			der_vals[n] = der_fid
			self.set_params(**self.params)
		return der_vals

	def derivative(self):
		varied_params = self.run_pipeline()
		deriv = {}
		for key in varied_params:
			delta = np.abs(self.delta*float(self.params[key]))
			deriv[key] = (varied_params[key].T[:, 1] -
							varied_params[key].T[:, 0])/(delta*2.)
		sort = sorted([[k, v] for k, v in deriv.items()], key=lambda x: x[0])
		derivatives = np.asarray([x[1] for x in sort])
		return derivatives

	def get_weight(self):
		self.derivatives = self.derivative()
		comp_weight = []
		for i in range(len(self.derivatives)):
			numerator = np.dot(self.derivatives[i], self.cov_inverse)
			numerator /= np.linalg.norm(numerator)
			comp_weight.append(numerator)
		comp_weight = np.array(comp_weight)
		if self.transform == 'trans':
			W = null_space(self.derivatives).T
			W /= np.linalg.norm(W)
			weight = np.concatenate((comp_weight, W))
		else:
			weight = comp_weight
		return weight

	def get_transformed(self):
		cov = (self.weight).dot(self.cov).dot((self.weight).T)
		yd = np.dot(self.weight, self.data)
		return cov, yd

	def get_data(self):
		data = np.loadtxt(self.input_dir+'data.txt')
		cov = np.loadtxt(self.input_dir+'covariance.txt')
		cov_inv = np.loadtxt(self.input_dir+'inv_covariance.txt')
		print("\n Behold, you are about to obtain a compressed \
				covariance matrix and a compressed data vector. \n \n ")
		return data, cov, cov_inv


def output(transform, parameters, delta=0.005):
	run = MOPED(transform, parameters, delta)
	data = run.y_d
	covariance = run.cov_c
	weight = run.weight
	return weight, data, covariance
