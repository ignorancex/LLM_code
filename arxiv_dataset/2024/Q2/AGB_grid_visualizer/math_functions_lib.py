#!/usr/bin/env python3
####################################################
#
# Author: M Joyce
#
####################################################
import numpy as np
import re
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy.stats import norm

from scipy.interpolate import CubicSpline

import sys
sys.path.append('/home/mjoyce/MESA/py_mesa_reader/')
import mesa_reader as mr


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx],idx


def search_string_with_regex(file_path, search_string):
    with open(file_path, 'r') as file:
        file_contents = file.read()
        match = re.search(search_string, file_contents)
        return match is not None


def grab_age(history_file):
	md = mr.MesaData(history_file)

	#model_number = md.model_number
	#star_mass = md.star_mass
	star_age = md.star_age/1e6
	#log_LH = md.log_LH
	#log_Teff = md.log_Teff
	#log_L = md.log_L
	#log_R = md.log_R
	#log_g = md.log_g
	#phase_of_evolution = md.phase_of_evolution

	min_age = star_age.min()
	max_age = star_age.max()

	value = (float(min_age) + float(max_age)) / 2.0

	return value


def cost_function_5dof(theory_value1, theory_value2, theory_value3, theory_value4, theory_value5,\
			           obs_value1, obs_value2, obs_value3, obs_value4, obs_value5,\
			   		   obs_sigma1, obs_sigma2, obs_sigma3, obs_sigma4, obs_sigma5,\
			    	   *args, **kwargs):

	term1 = ( (float(theory_value1)-obs_value1)/obs_sigma1 )**2.0
	term2 = ( (float(theory_value2)-obs_value2)/obs_sigma2 )**2.0
	term3 = ( (float(theory_value3)-obs_value3)/obs_sigma3 )**2.0
	term4 = ( (float(theory_value4)-obs_value4)/obs_sigma4 )**2.0
	term5 = ( (float(theory_value5)-obs_value5)/obs_sigma5 )**2.0

	rank = (term1 + term2 + term3 + term4 + term5) 
	return rank

def scaled_kde(xdata, ydata):
	cs_PO1  = CubicSpline(xdata, ydata)
	## resample to increase resolution
	sample_at = 0.01/1e6 
	resampled_star_age_theory = np.arange(xdata.min(), xdata.max(), sample_at)

	spline_ydata = 	cs_PO1(resampled_star_age_theory)

	## make a new kde representing the MESA model
	kde_model = stats.gaussian_kde(spline_ydata) ## the actual periods from MESA, splined
	x_values = np.linspace(min(spline_ydata), max(spline_ydata), 1000)

	kde = kde_model(x_values)

	# ## build an x-data array the same size as resampled y_values
	# x_values = resampled_star_age_theory

	## scale the kde by the number of stellar ages in our sample (91)
	scaled_kde = kde*len(x_values)

	return x_values, scaled_kde



def compute_wrmse(theory_vector, obs_vector, obs_err_vector):
	def weight(sigma):
		w = 1.0/(sigma**2.0)
		return w

	n = []
	#d = []
	for i in range(len(theory_vector)):
		n_i = weight(obs_err_vector[i])*(obs_vector[i] - theory_vector[i])**2.0
		n.append(n_i)

	n = np.array(n)
	d = weight(obs_err_vector)

	numerator 	=  sum(n)
	denominator =  sum(d)

	wrmse = np.sqrt(numerator/denominator)

	return wrmse 

def get_FeH_dict(): ## 37 entries
	FeH_dict={"0.0001":-2.2,\
		  "0.0005":-1.53,\
		  "0.0010":-1.2,\
		  "0.0013":-1.105,\
		  "0.0018":-1.02,\
		  "0.0020":-0.93,\
		  "0.0025":-0.835,\
		  "0.0030":-0.75,\
		  "0.0036":-0.675,\
		  "0.0040":-0.60,\
		  "0.0050":-0.53,\
		  "0.0060":-0.45,\
		  "0.0070":-0.38,\
		  "0.0080":-0.325,\
		  "0.0095":-0.25,\
		  "0.0100":-0.20,\
		  "0.0125":-0.13,\
		  "0.0135":-0.075,\
		  "0.0140":0.00,\
		  "0.0200":0.055,\
		  "0.0216":0.11,\
		  "0.0247":0.17,\
		  "0.0300":0.25,\
		  "0.0344":0.318,\
		  "0.0400":0.39,\
		  "0.0450":0.44,\
		  "0.0500":0.49,\
		  "0.0600":0.57,\
		  "0.0700":0.645,\
		  "0.0800":0.70,\
		  "0.0900":0.765,\
		  "0.1000":0.82,\
		  "0.1100":0.86,\
		  "0.1200":0.905,\
		  "0.1300":0.945,\
		  "0.1400":0.98,\
		  "0.1500":1.00,\
		   }
	return FeH_dict

def get_Z_dict(): ### 37 entries
	Z_dict={-2.2000:"0.0001",\
			-1.5300:"0.0005",\
			-1.2000:"0.0010",\
			-1.1050:"0.0013",\
			-1.0200:"0.0018",\
			-0.9300:"0.0020",\
			-0.8350:"0.0025",\
			-0.7500:"0.0030",\
			-0.6750:"0.0036",\
			-0.6000:"0.0040",\
			-0.5300:"0.0050",\
			-0.4500:"0.0060",\
			-0.3800:"0.0070",\
			-0.3250:"0.0080",\
			-0.2500:"0.0095",\
			-0.2000:"0.0100",\
			-0.1300:"0.0125",\
			-0.0750:"0.0135",\
			0.0000:"0.0140",\
			0.0550:"0.0200",\
			0.1100:"0.0216",\
			0.1700:"0.0247",\
			0.2500:"0.0300",\
			0.3180:"0.0344",\
			0.3900:"0.0400",\
			0.4400:"0.0450",\
			0.4900:"0.0500",\
			0.5700:"0.0600",\
			0.6450:"0.0700",\
			0.7000:"0.0800",\
			0.7650:"0.0900",\
			0.8200:"0.1000",\
			0.8600:"0.1100",\
			0.9050:"0.1200",\
			0.9450:"0.1300",\
			0.9800:"0.1400",\
			1.0000:"0.1500",\
		   }
	return Z_dict


def Y_init(z_in):
	Y0 = 0.2485
	dYdZ = 2.1 ## Luca Casagrande
	yinit = Y0 + dYdZ*float(z_in)
	return yinit