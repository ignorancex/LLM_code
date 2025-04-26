"""
__author__ = "Babu Kumaran Nalini"
__copyright__ = "2021 TUM-EWK"
__credits__ = []
__license__ = "GPL v3.0"
__version__ = "1.0"
__maintainer__ = "Babu Kumaran Nalini"
__email__ = "babu.kumaran-nalini@tum.de"
__status__ = "Development"
"""

from functions import *
import pickle
import pandas as pd
import numpy as np

eol = 20                          
roi = 0.035
af = annuity_factor(roi, eol)

siz_pv = [0, 1.62, 3.24, 4.86, 6.48, 8.1]
siz_bat = [0, 4.5, 9, 13.5, 18, 22.5]
scenario = [10]

for yr in scenario:
    num_of_offers = [50, 100, 200, 365]
    if yr == 0:
        rev_flex = [0.04, 0.08, 0.12, 0.16] 
    elif yr == 10:
        rev_flex = [0.02, 0.06, 0.12, 0.16]

    dict_all = {}
    matx_amep = pd.DataFrame(np.zeros((len(num_of_offers), len(rev_flex)))) 
    matx_size = pd.DataFrame(np.zeros((len(num_of_offers), len(rev_flex))))
    
    for nof in range(len(num_of_offers)):
        for rf in range(len(rev_flex)):
            least_cost = price_predict('import', yr)
            c_net = pd.DataFrame(np.zeros((len(siz_pv), len(siz_bat))))
            min_i = min_j = 0
            for i in range(len(siz_pv)):
                for j in range(len(siz_bat)):
                    file_to_read = open("TUMflexpy/var/ems"+str(i)+str(j)+".pickle", "rb")
                    dict_load = pickle.load(file_to_read)
                    
                    # investment costs
                    c_i = invest_cost('pv', siz_pv[i], yr)
                    c_pv = c_i*af*1.04
                    
                    c_i = invest_cost('bat', siz_bat[j], yr)
                    c_bat = c_i*af*1.04
                    
                    # cost of procured electricity
                    fd_1 = [x1-x2-x3 for (x1,x2,x3) in zip(dict_load['optplan']['Last_elec'],
                                                           dict_load['optplan']['pv_pv2demand'],
                                                           dict_load['optplan']['bat_output_power'])]
                    avg = sum(dict_load['fcst']['ele_price_in'])/len(dict_load['fcst']['ele_price_in'])
                    act = price_predict('import', yr)
                    new = [x*act/avg for x in dict_load['fcst']['ele_price_in']]
                    fd = [x1*x2 for (x1,x2) in zip(fd_1, new)]
                    c_el = sum(fd)*0.25
                    # c_el = sum(fd_1)*0.25*price_predict('import', yr)
                    
                    # revenue from PV
                    r_pv = dict_load['optplan']['pv_pv2grid']
                    r_pv = sum(r_pv)*0.25*price_predict('export', yr)
                    
                    # revenue from flexibility
                    bat_neg = dict_load['batflex']['Neg_E']
                    avg_negflex = get_flex(bat_neg)
                    bat_pos = dict_load['batflex']['Pos_E']
                    avg_posflex = get_flex(bat_pos)
                    r_flex = ((avg_negflex+avg_posflex)/2)*num_of_offers[nof]*rev_flex[rf]
                    # r_flex = 0
                    
                    # cost of prosumer purchase
                    c_net.iloc[i,j] = (c_pv + c_bat + c_el - r_pv - r_flex)/(sum(dict_load['optplan']['Last_elec'])*0.25)
                    
                    # get optimal size
                    if c_net.iloc[i,j] <= least_cost:
                        least_cost = c_net.iloc[i,j]
                        min_i = i
                        min_j = j
                                              
            dict_all[str(nof*1000+rf*100+i*10+j)] = c_net 
            matx_amep.iloc[nof,rf] = least_cost
            matx_size.iloc[nof,rf] = 'PV:'+str(min_i*0.5)+'\nBES:'+str(min_j*0.5)
            
    # matrix re-order
    matx_size = nearest_val(matx_size, yr)
    plt_imshow_anal(matx_amep, matx_size, yr, save='flexmat_constant')
                       
    # save analysis
    # file_to_write = open("present_dict"+".pickle", "wb")
    # pickle.dump(dict_anal, file_to_write)