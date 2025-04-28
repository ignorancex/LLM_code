import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd




def plot_curve(fig, curve, color, label):
    """
    Add a 3d plot of a curve to fig.
    
    Params:
        fig   --- go.Figure()
        curve --- (3,n) array
        label --- string  
        color --- string
        
    Returns:
        go.Figure()  

    """    
    df = pd.DataFrame.from_records(curve, columns=['X', 'Y','Z'])

    fig.add_trace(go.Scatter3d(
    x=df['X'], y=df['Y'], z=df['Z'], 
    name = label,    
        marker=dict(
        size=5,
        color= color,
        line=dict(width=0.3,
                color='DarkSlateGrey'),
    ),
    line=dict(
        width=5,
        color = color
    )),
    )
    
    fig.update_layout(scene=dict(xaxis = dict(
                         backgroundcolor="rgb(200, 200, 230)",
                         gridcolor='rgba(0,0,0,0)',
                         showbackground=False,
                         zerolinecolor='rgba(0,0,0,0)',showticklabels=False,),
     yaxis = dict(
                         backgroundcolor="rgb(200, 200, 230)",
                         gridcolor='rgba(0,0,0,0)',
                         showbackground=False,
                         zerolinecolor='rgba(0,0,0,0)',showticklabels=False),
               zaxis = dict(
                         backgroundcolor="rgb(200, 200, 230)",
                         gridcolor='rgba(0,0,0,0)',
                         showbackground=False,
                         zerolinecolor='rgba(0,0,0,0)',showticklabels=False),
    ))
                  
    fig.update_layout(scene = dict(
                    xaxis_title=' ',
                    yaxis_title=' ',
                    zaxis_title=' '))

    
    return(fig)


def add_core(fig,curve,index,list_of_cores,color):
    
    if list_of_cores[index][0] != '0_1':
        s,e = list_of_cores[index][1] ,list_of_cores[index][2]
        
        if s<e:
            core = curve[s:e]
        else:
            core = [el for el in curve[s:]]
            for i in range(e):
                core.append(curve[i])
        
        df = pd.DataFrame.from_records(core, columns=['X', 'Y','Z'])
        fig.add_trace(go.Scatter3d(
        x=df['X'], y=df['Y'], z=df['Z'], 
        name = 'Knot core',    
        marker=dict(
            size=8,
        line=dict(width=0.3,
                color='DarkSlateGrey'),
            color= color,
        ),
        line=dict(
            width=5,
            color = color
        )),
        )
        
        df = pd.DataFrame.from_records(curve[index:index+1], columns=['X', 'Y','Z'])
        fig.add_trace(go.Scatter3d(
        x=df['X'], y=df['Y'], z=df['Z'], 
        name = 'Opening point',    
        showlegend=True,
        mode='markers' ,   
        marker=dict(
            size=8,
        line=dict(width=0.3,
                color='DarkSlateGrey'),
            color= "#ff9b71",
        ),
        ),
        )        
    return fig    



def plot_radar_intensity(g):

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=g, 
        fill='toself',
        line = dict(width = 5),
        marker=dict(color = '#267592',
                   size = 10),
                    connectgaps = True,
        showlegend=False
                             ),
            
              
             )
    
    fig.update_layout(legend=dict(
    yanchor="top",
    y=0.7,
    xanchor="left",
    x=1
    ))
    
    fig.update_layout(title = 'Density {}'.format(np.round(np.sum(g)/100,3)))
    fig.update_polars(radialaxis_dtick=0.1)
    fig.update_polars(radialaxis_ticks='outside',
                     radialaxis_tickwidth=2)
    fig.update_polars(radialaxis_showticklabels=False)

    fig.show()


def plot_fingerprint(fingerprint):

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=[i/100 for i in range(100)], 
                         y=fingerprint, 
                         line=dict(width = 3,
                                    color = '#267592'),
                         showlegend= False),
              
                
                )
    fig.update_layout(height = 600, width = 600,
                  title = 'Fingerprint',
                     title_x = 0.5,
                     title_y = 0.05,
                 )
    fig.update_yaxes(range=[0, 1])

    fig.show()



def size_dic(color):
    
    new = [5+(elem - min(color))*50 + 1  for elem in color]
    return np.array(new)

def plot_knot_with_intensity(curve, knot_intensity):
    """
    """
    
    fig = go.Figure()
    color = knot_intensity
    if curve[0][0] != curve[-1][0]:
        curve = [el for el in curve]
        curve.append(curve[0])
        color = np.append(color, color[0])
    
    df = pd.DataFrame.from_records(curve, columns=['X', 'Y','Z'])

    fig.add_trace(go.Scatter3d(
    x=df['X'], y=df['Y'], z=df['Z'],
        
        marker=dict(
        size=size_dic(color),
        color= color,
        colorscale= 'viridis',
    
        #line=dict(width=0.3,
         #       color='white'),
        cmax=1,
        cmin=0,
        

    ),
    line=dict(
        width=5,
        color = color,
        colorscale = 'viridis',
        cmax=1,
        cmin=0
    )),
    )
    
    fig.update_layout(scene=dict(xaxis = dict(
                         backgroundcolor="rgb(200, 200, 230)",
                         gridcolor='rgba(0,0,0,0)',
                         showbackground=False,
                         zerolinecolor='rgba(0,0,0,0)',showticklabels=False,),
     yaxis = dict(
                         backgroundcolor="rgb(200, 200, 230)",
                         gridcolor='rgba(0,0,0,0)',
                         showbackground=False,
                         zerolinecolor='rgba(0,0,0,0)',showticklabels=False),
               zaxis = dict(
                         backgroundcolor="rgb(200, 200, 230)",
                         gridcolor='rgba(0,0,0,0)',
                         showbackground=False,
                         zerolinecolor='rgba(0,0,0,0)',showticklabels=False),
    ))
                  
    fig.update_layout(scene = dict(
                    xaxis_title=' ',
                    yaxis_title=' ',
                    zaxis_title=' '))

    fig.update_traces(hoverinfo="text",hovertemplate=[[i,color[i]] for i in range(len(curve)-1)])
    return(fig)


