import os
from collections import namedtuple

import pandas as pd
import math
import torch

from numpy import round

dset = namedtuple("dset", ["name", "exposureFileName", "deathFilename"])

SWEDEN = dset("sweden",
                    "../data/expSweHMD.txt",
                    "../data/deathsSweHMD.txt")

USA = dset("usa",
"../data/expUsaHMD.txt",
"../data/deathsUsaHMD.txt")

def load_data(param, train, test):

	dataset = param['country']
	max_age = param['max_age']
	sex = param['sex']

	if train == True and test == True:
		first_year = param['first_year_train']
		last_year = param['last_year_test']
	elif train == True and test == False:
		first_year = param['first_year_train']
		last_year = param['last_year_train']
	elif train == False and test == True:
		first_year = param['first_year_test']
		last_year = param['last_year_test']
	else:
		return


	base_path = os.path.dirname(os.path.realpath(__file__))

	file_loc = os.path.join(base_path, dataset.exposureFileName)

	#Load the exposure dataset
	expdf = pd.read_table(file_loc, delim_whitespace=True)
	if sex == "Male":
		expdf = expdf.drop(columns=["Female","Total"])
		expdf = expdf.rename(columns= {"Male": "Exposure"})
	elif sex == "Female":
		expdf = expdf.drop(columns=["Male","Total"])
		expdf = expdf.rename(columns= {"Female": "Exposure"})
	else:
		expdf = expdf.drop(columns=["Female","Male"])
		expdf = expdf.rename(columns= {"Total": "Exposure"})

	expdf.loc[expdf['Age'] == "110+", 'Age'] = "110"

	expdf["Age"] = pd.to_numeric(expdf["Age"])
	
	expdf = expdf[expdf['Year']>= first_year]
	expdf = expdf[expdf['Year']<= last_year]

	expdf = expdf[expdf['Age']<= max_age]
	
	expArray = expdf["Exposure"].values.reshape((-1,max_age+1))

	#Load the deaths dataset
	file_loc = os.path.join(base_path, dataset.deathFilename)

	deathdf = pd.read_table(file_loc, delim_whitespace=True)
	if sex == "Male":
		deathdf = deathdf.drop(columns=["Female","Total"])
		deathdf = deathdf.rename(columns= {"Male": "Deaths"})
	elif sex == "Female":
		deathdf = deathdf.drop(columns=["Male","Total"])
		deathdf = deathdf.rename(columns= {"Female": "Deaths"})
	else:
		deathdf = deathdf.drop(columns=["Male","Female"])
		deathdf = deathdf.rename(columns= {"Total": "Deaths"})

	deathdf.loc[deathdf['Age'] == "110+", 'Age'] = "110"
	deathdf["Age"] = pd.to_numeric(expdf["Age"])
	
	deathdf = deathdf[deathdf['Year']>= first_year]
	deathdf = deathdf[deathdf['Year']<= last_year]

	deathdf = deathdf[deathdf['Age']<= max_age]
	
	deathArray = deathdf["Deaths"].values.reshape((-1,max_age + 1)).astype(float)

	return expArray, round(deathArray)




