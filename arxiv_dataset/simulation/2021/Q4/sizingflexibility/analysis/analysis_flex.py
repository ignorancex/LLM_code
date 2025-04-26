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

from functions import annuity_factor, invest_cost, plt_contour, price_predict, get_flex
import pickle
import pandas as pd
import numpy as np

eol = 20                          
roi = 0.035
af = annuity_factor(roi, eol)

siz_pv = [0, 1.62, 3.24, 4.86, 6.48, 8.1]
siz_bat = [0, 4.5, 9, 13.5, 18, 22.5]
scenario = [0]

for yr in scenario:
    c_net = pd.DataFrame(np.zeros((len(siz_pv), len(siz_bat))))
    for i in range(len(siz_pv)):
        for j in range(len(siz_bat)):
            file_to_read = open("TUMflexpy/var/variable/ems"+str(i)+str(j)+".pickle", "rb")
            dict_load = pickle.load(file_to_read)
            
            # investment costs
            c_i = invest_cost('pv', siz_pv[i], yr)
            c_pv = c_i*af*1.02
            
            c_i = invest_cost('bat', siz_bat[j], yr)
            c_bat = c_i*af*1.02
            
            # cost of procured electricity
            fd_1 = [x1-x2-x3 for (x1,x2,x3) in zip(dict_load['optplan']['Last_elec'],
                                                   dict_load['optplan']['pv_pv2demand'],
                                                   dict_load['optplan']['bat_output_power'])]
            avg = sum(dict_load['fcst']['ele_price_in'])/len(dict_load['fcst']['ele_price_in'])
            act = price_predict('import', yr)
            new = [x*act/avg for x in dict_load['fcst']['ele_price_in']]
            fd = [x1*x2 for (x1,x2) in zip(fd_1, new)]
            c_el = sum(fd)*0.25
            
            # revenue from PV
            r_pv = dict_load['optplan']['pv_pv2grid']
            r_pv = sum(r_pv)*0.25*price_predict('export', yr)
            
            # revenue from flexibility
            bat_neg = dict_load['batflex']['Neg_E']
            avg_negflex = get_flex(bat_neg)
            bat_pos = dict_load['batflex']['Pos_E']
            avg_posflex = get_flex(bat_pos)
            r_flex = (avg_negflex+avg_posflex)*182*0.2
            
            # cost of prosumer purchase
            c_net.iloc[i,j] = (c_pv + c_bat + c_el - r_pv - r_flex)/(sum(dict_load['optplan']['Last_elec'])*0.25)
    
    # plot average electricity price
    plt_contour(c_net, yr, save='flex_var_182_20')