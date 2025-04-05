# from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
import importlib

# import tensorflow as tf
# from tensorflow.keras.models import Sequential, load_model
# import Create_Pseudodata as CPDATA
import CR_ML_Class as CR


# importlib.reload(CR)
para_0_0 = np.load("./Numpy_mock_data/parameter_0.npy")
para_0_1 = np.load("./Numpy_mock_data/new_parameter_0.npy")
origin_data_0 = np.load("./Numpy_mock_data/data_0.npy")
mock_0 = CR.Mock_Data_PreProcessing(para_0_0, para_0_1, origin_data_0)
print("There are {} data ".format(len(mock_0.new_parameter)))
chi_0 = np.load("./Numpy_mock_data/new_chi_0.npy")

para_1_0 = np.load("./Numpy_mock_data/parameter_1.npy")
para_1_1 = np.load("./Numpy_mock_data/new_parameter_1.npy")
origin_data_1 = np.load("./Numpy_mock_data/data_1.npy")
mock_1 = CR.Mock_Data_PreProcessing(para_1_0, para_1_1, origin_data_1)
chi_1 = np.load("./Numpy_mock_data/new_chi_1.npy")

parameter_2_0 =  np.load("./Numpy_mock_data/parameter_2.npy")
parameter_2_1 = np.load("./Numpy_mock_data/new_parameter_2.npy")
origin_data_2 = np.load("./Numpy_mock_data/data_2.npy")
mock_2 = CR.Mock_Data_PreProcessing(parameter_2_0, parameter_2_1, origin_data_2)
chi_2 = np.load("./Numpy_mock_data/new_chi_2.npy")


parameter_3_0 =  np.load("./Numpy_mock_data/parameter_3.npy")
parameter_3_1 = np.load("./Numpy_mock_data/new_parameter_3.npy")
origin_data_3 = np.load("./Numpy_mock_data/data_3.npy")
mock_3 = CR.Mock_Data_PreProcessing(parameter_3_0, parameter_3_1, origin_data_3)
chi_3 = np.load("./Numpy_mock_data/new_chi_3.npy")

parameter_4_0 =  np.load("./Numpy_mock_data/parameter_4.npy")
parameter_4_1 = np.load("./Numpy_mock_data/new_parameter_4.npy")
origin_data_4 = np.load("./Numpy_mock_data/data_4.npy")
mock_4 = CR.Mock_Data_PreProcessing(parameter_4_0, parameter_4_1, origin_data_4)
chi_4 = np.load("./Numpy_mock_data/new_chi_4.npy")

parameter = np.concatenate((mock_0.new_parameter,mock_1.new_parameter))
data = np.concatenate((mock_0.data,mock_1.data))
chi = np.concatenate((chi_0,chi_1))

parameter = np.concatenate((parameter,mock_2.new_parameter))
data = np.concatenate((data,mock_2.data))
chi = np.concatenate((chi,chi_2))

parameter = np.concatenate((parameter,mock_3.new_parameter))
data = np.concatenate((data,mock_3.data))
chi = np.concatenate((chi,chi_3))

parameter = np.concatenate((parameter,mock_4.new_parameter))
data = np.concatenate((data,mock_4.data))
chi = np.concatenate((chi,chi_4))


# parameter = mock_4.new_parameter[:15000]
# data = mock_4.data[:15000]
# chi =chi_4[:15000]

chi_sort = np.argsort(chi)

print("minium chi:", min(chi))
print("There are {} data ".format(parameter.shape[0]))


importlib.reload(CR)
chi_para, chi_data, chi_sele = parameter,data,chi #mock_0.new_parameter[:], mock_0.data[:], chi_0[:]

para_1_sigma, data_1_sigma, _ =  CR.Select_Sample(chi_para, chi_data, chi_sele,1).Sample()
para_2_sigma, data_2_sigma, _ =  CR.Select_Sample(chi_para, chi_data, chi_sele,2).Sample()
para_3_sigma, data_3_sigma, _ =  CR.Select_Sample(chi_para, chi_data, chi_sele,3).Sample()
para_4_sigma, data_4_sigma, _ =  CR.Select_Sample(chi_para, chi_data, chi_sele,4).Sample()
para_5_sigma, data_5_sigma, _ =  CR.Select_Sample(chi_para, chi_data, chi_sele,5).Sample()
para_6_sigma, data_6_sigma, _ =  CR.Select_Sample(chi_para, chi_data, chi_sele,6).Sample()


# pseudoexp = CR.Create_Pseudodata(para_6_sigma,data_6_sigma,chi,len(para_6_sigma)).Create_Pseudodata()
pseudoexp = CR.Create_Pseudodata(para_6_sigma,data_6_sigma,chi, number = len(para_6_sigma)).Create_Pseudodata()
normalfactor, pseudodata = pseudoexp[0],  pseudoexp[1]

print(len(normalfactor),len(pseudodata))

np.save("./pseudo_normalfactor_6_sigma.npy",normalfactor)
np.save("./pseudo_data_6_sigma.npy",pseudodata)


