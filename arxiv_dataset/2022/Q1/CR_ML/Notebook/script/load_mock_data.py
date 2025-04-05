#!/usr/bin/python3

# basic python package
import numpy as np
import random
import time
from  tqdm import tqdm
import logging

logging.basicConfig(level = logging.INFO)

# self-define classes
from script import CR_ML_Class as CR




class load_mock_data:
    def __init__(self, origin_para = [], new_para = [], mock_data = [], chi_square = [], mock_data_path="PATH" ):
        
        logging.info("Now loading...")
        time.sleep(0.5)
        
        para_0 = np.load(mock_data_path + origin_para[0])
        para_1 = np.load(mock_data_path + new_para[0])
        origin_data = np.load(mock_data_path + mock_data[0])
        mock_new = CR.Mock_Data_Rescale(para_0, para_1, origin_data)
        logging.info("{}/{}".format("1",len(origin_para)))
        logging.info("There are {} data ".format(len(mock_new.new_parameter)))
        chisq = np.load(mock_data_path + chi_square[0])
        parameter = mock_new.new_parameter
        data = mock_new.data
        
        for i, (para_ori, para_new, mock, chi) in enumerate(zip(origin_para,new_para,mock_data,chi_square)):
            if i == 0:
                continue
            
            para_0_tmp = np.load(mock_data_path + origin_para[i])
            para_1_tmp = np.load(mock_data_path + new_para[i])
            origin_data_tmp = np.load(mock_data_path + mock_data[i])
            mock_new_tmp = CR.Mock_Data_Rescale(para_0_tmp, para_1_tmp, origin_data_tmp)
            logging.info("{}/{}".format(i+1,len(origin_para)))
            logging.info("There are {} data ".format(len(mock_new_tmp.new_parameter)))
            chi_tmp = np.load(mock_data_path + chi_square[i])
            
            
            parameter = np.concatenate((parameter, mock_new_tmp.new_parameter))
            data = np.concatenate((data, mock_new_tmp.data))
            chisq = np.concatenate((chisq, chi_tmp))
        
        logging.info("Finish")
        
        self.parameter, self.data, self.chisq = parameter, data, chisq

