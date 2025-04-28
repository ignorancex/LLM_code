import importlib
import setup_nx # your own module, setup.nx.py

importlib.reload(setup_nx)
from setup_nx import *
from graphviz import render
import datetime
import my_configVis_funcs as vis

def feeder_init(modelpath, loadfolder, loadpath, timesteps, Vbase_ll, Sbase, depths, leaves):
    #initialize the feeder object
    modeldata = pd.ExcelFile(modelpath)
    actpath = loadpath
    
    # set dummy values for undefined variables
    date = datetime.datetime.now()
    month = date.month
    day = date.day
    hour = date.hour
    minute = date.minute
    #timestepcur = hour*60+minute
    timestepcur = 0
    
    Psat_nodes = []
    Qsat_nodes = []
    
    refphasor = np.ones((3,2))
    refphasor[:,0]=1
    refphasor[:,1]=[0,4*np.pi/3,2*np.pi/3]
    
    PVforecast = 0
    
    #get feeder
    fin_feeder = feeder(modelpath,loadfolder,loadpath,actpath,timesteps,timestepcur,
                         Vbase_ll,Sbase,refphasor,Psat_nodes,Qsat_nodes,PVforecast, depths, leaves)    
    return fin_feeder


def make_graph(feeder, file_name):
    #generates graph of network as png
    #file_name = string specifying name of dot file created when make_graph() is called
    #feeder = initialized feeder object from feeder_init()
    #for node in feeder.network.nodes:
    graph = feeder.network
    nodes = graph.nodes
    edges = graph.edges
    
    for n in nodes:
        graph.nodes[n]['fontsize'] = 15
        graph.nodes[n]['label'] = ''
        graph.nodes[n]['shape'] = 'circle'
        graph.nodes[n]['width'] = .25
        graph.nodes[n]['xlabel'] = n[4:]
        
    for e in edges:
        graph.edges[e]['arrowhead'] = 'none'
        graph.edges[e]['color'] = 'gray'

    vis.write_formatted_dot(graph, file_name+'_blank')
    # nx.nx_pydot.write_dot(graph, 'generated_figs/'+file_name+'_blank')
    # render('dot', 'png', 'generated_figs/'+file_name+'_blank')
    return

def clear_graph(feeder):
    #returns feeder graph to original state (removes any node colors/changes to shape or size)
    graph = feeder.network
    for n in graph.nodes:
        graph.nodes[n]['style'] = 'filled'
        graph.nodes[n]['fillcolor'] = 'white'
        graph.nodes[n]['color'] = 'black'
        graph.nodes[n]['fontcolor'] = 'black'
        graph.nodes[n]['fontsize'] = 15
        graph.nodes[n]['label'] = ''
        graph.nodes[n]['shape'] = 'circle'
        graph.nodes[n]['width'] = .25
    return

# see img: https://i.stack.imgur.com/3ZRVT.png