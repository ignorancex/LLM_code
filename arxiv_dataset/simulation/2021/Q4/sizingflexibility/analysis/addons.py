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

def invest_cost(dev, size, yr):
    if dev == 'pv':
        if size < 3:
            c_inv = 1900
        elif size > 3 and size < 4:
            c_inv = 1755
        elif size > 4 and size < 5:
            c_inv = 1530
        elif size > 5 and size < 6:
            c_inv = 1545
        elif size > 6 and size < 7:
            c_inv = 1485
        elif size > 7 and size < 8:
            c_inv = 1510
        elif size > 8 and size < 9:
            c_inv = 1510
        elif size > 9 and size < 10:
            c_inv = 1450
        elif size > 10 and size < 15:
            c_inv = 1355
        elif size > 15 and size < 20:
            c_inv = 1350
        else:
            print('investment cost unknown')
        c_inv = (c_inv-yr*50)*size
    elif dev == 'bat':
        if size < 5:
            c_inv = 1747
        elif size > 5 and size < 10:
            c_inv = 1348
        elif size > 10:
            c_inv = 1212          
        c_inv = c_inv*size*pow(0.89, yr)
    return c_inv

def price_predict(typ, yr):
    if typ == 'import':
        prc = 0.3189*pow(1.015, yr)
    elif typ == 'export':
        if yr == 0:
            prc = 0.0683
        elif yr == 2:
            prc = 0.44
        elif yr == 5:
            prc = 0.03
        elif yr == 10:
            prc = 0.02
    return prc


# overall data
overall_data.iloc[180*nof+36*rf+6*i+j,0] = num_of_offers[nof]
overall_data.iloc[180*nof+36*rf+6*i+j,1] = rev_flex[rf]
overall_data.iloc[180*nof+36*rf+6*i+j,2] = siz_pv[i]
overall_data.iloc[180*nof+36*rf+6*i+j,3] = siz_bat[j]
overall_data.iloc[180*nof+36*rf+6*i+j,4] = c_net.iloc[i,j]              
                    
# get least cost config
if c_net.iloc[i,j] < least_cost:
    count.iloc[nof,rf] += 1
    best_config.iloc[nof,rf] += 1
    if i < 2 and j < 2:
        under_sizing.iloc[nof,rf] += 1
    elif i > 2 and j > 2:
        over_sizing.iloc[nof,rf] += 1

best_config = pd.DataFrame(np.zeros((len(num_of_offers), len(rev_flex))))
under_sizing = over_sizing = count = best_config.copy()
    