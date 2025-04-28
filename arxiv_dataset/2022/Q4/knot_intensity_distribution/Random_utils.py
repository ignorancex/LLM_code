import numpy as np
import pickle
from Functions_utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

LENGHTS = [100,150,200,250]


KNOTS = ['+3_1','-3_1', '4_1', '+5_1', '-5_1', '+5_2', '-5_2', '+6_1', '-6_1','+6_2', '-6_2','6_3', 
         '+3_1#+3_1','+3_1#-3_1', '-3_1#-3_1', '+3_1#4_1','-3_1#4_1' ]
ACHIRAL_KNOTS = ['3_1','4_1', '5_1', '5_2', '6_1', '6_2', '6_3','3_1#3_1','3_1#m3_1','3_1#4_1']

GREY_PALETTE = ['#0a0a0a','#333333', '#595959','#808080','#a6a6a6']

COLOR_PALETTE = ['#d0e3f5',
                 '#267592',
                 '#47b58e',
                 '#5fb12a',
                 '#92bd11',
                 '#fac800',
                 '#ff7917',
                 '#e23a34',
                 '#712e67',
                 '#870837',
                 '#480b66',
                 "#ffd700",
                 "#ffb14e",
                 "#fa8775",
                 "#ea5f94",
                 "#cd34b5",
                 "#9d02d7",
                 "#ab7ca3",
                 "#0064ab",
                 "#0000ff"]


GRAD_PALETTE = [ '#013c66',  '#8ac3eb', '#47b58e',
                '#0c6624', '#3a8a5c','#68e39c', '#07f047',
                '#f2bd49', '#f76125', '#bd4a4d','#d47f81', '#f7a711', 
                '#4f080b', '#8f1d21', 
                
               ]


def cores():
    dic = {}
    for k in KNOTS:
        dic[k] = {}
        for l in LENGHTS:
            dic[k][l] = load_cores(k,l)
    return dic



def load_cores(knot_type = '+3_1', length = 100, dataset = 'random'):
    with open('knot_cores_dics/{}_{}_cores.pickle'.format(knot_type, length),'rb') as handle:
        dic = pickle.load(handle)
    return(dic) 


KNOT_CORES = cores()






def plot_knot_intensities(k = '+3_1', l = 100, start_index = 0):
    s = []
    f = []
    for ind in range(100):
    
        core = KNOT_CORES[k][l][ind+ start_index]
        g = knot_intensity(core, l)

        f.append(g)
        s.append(np.sum(g)/l)
    s = np.round(s,3)    
    fig = make_subplots(rows=25, cols=4,
                   shared_xaxes=True,
                   shared_yaxes=True,
                   subplot_titles = ['{} N {} D {}'.format(k,i + start_index,s[i]) 
                                     for i in range(100)]
                       )
        
    for ind in range(100):
        fig.add_trace(
                go.Scatter(x=[i for i in range(l)], y=f[ind], showlegend= False,
                          line=dict(color = GREY_PALETTE[1]),
                          ),
                row=int(ind/4) +1, col=ind%4 +1
                )
 
    fig.update_yaxes(showticklabels=False)    
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=10)    
    fig.update_layout(height=3000, width=1200)    
    return fig


