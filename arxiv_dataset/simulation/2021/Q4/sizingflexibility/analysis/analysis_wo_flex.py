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
    c_net = pd.DataFrame(np.zeros((len(siz_pv), len(siz_bat))))
    t_flex = pd.DataFrame(np.zeros((len(siz_pv), len(siz_bat))))
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
            pv_neg = dict_load['pvflex']['Neg_E']
            avg_negflexpv = get_flex(bat_neg)
            t_flex.iloc[i,j] = (avg_negflex + avg_posflex)/2
            r_flex = t_flex.iloc[i,j]*0
            t_flex.iloc[i,j] += avg_negflexpv
            
            # cost of prosumer purchase
            c_net.iloc[i,j] = (c_pv + c_bat + c_el - r_pv - r_flex)/(sum(dict_load['optplan']['Last_elec'])*0.25)
    
    # plot average electricity price
    plt_contour(c_net, yr, save='woflex_const')
    plt_tflex(t_flex.round(2), yr, typ='imshow', save='flex_const')
