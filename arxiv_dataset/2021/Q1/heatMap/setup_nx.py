# In[22]:


## THIS SCRIPT STORES THE FUNCTIONS AND CLASSES NECESSARY TO CREATE A 'FEEDER' OBJECT FROM AN EPHASORSIM MODEL
#SPBC-HIL files moved 7/15/19

# In[23]:


import pandas as pd
import numpy as np
import string
import networkx as nx
from copy import deepcopy
import pdb
import matplotlib.pyplot as plt
import time

# In[24]:


# Adds traceback to warning messages, which is sometimes useful
import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback


# In[25]:


## CLASS DEFINITIONS/HELPER FUNCTIONS


# In[26]:



class feeder:
# The main class. Creating this from an ePHASORsim model and set of loads stores all the information about your network
# Buses, loads, lines, actuators, etc. are all stored as dictionaries within this object
    #[HIL] - added ICDI vars and phasor reference vars
    def __init__(self,modelpath,loadfolder,loadpath,actpath,timesteps,timestepcur,subkVbase_phg,subkVAbase,refphasor,Psat_nodes,Qsat_nodes,PVforecast, depth_dict, leaf_list):
    #def __init__(self,modelpath,loadfolder,loadpath,timesteps,subkVbase_phg,subkVAbase,depth_dict,leaf_list):
        # import data
        self.modeldata = pd.ExcelFile(modelpath)
 #       self.loadfolder = loadfolder
 #       self.loadpath = loadpath
        self.actpath = actpath
        self.timesteps = timesteps
        self.timestepcur = timestepcur
        self.subkVbase_phg = subkVbase_phg
        self.subkVAbase = subkVAbase
        self.refphasor = refphasor
        self.Psat_nodes = Psat_nodes
        self.Qsat_nodes = Qsat_nodes
        self.PVforecast = PVforecast
        # groom data (note that the functions below are not separable... 
        # ...e.g. buses are not completely defined until all these functions have been run)
        self.busdict = busbuilder(self.modeldata, subkVbase_phg, subkVAbase, timesteps)
        #self.Vsrcdict = Vsrcbuilder(self.modeldata, self.busdict)
        #****** NOTE: Uncomment this after formatting has been changed
        #self.shuntdict = shuntbuilder(self.modeldata, self.busdict,timesteps)
        self.loaddict = loadbuilderPQ(self.modeldata, self.busdict, loadpath, timesteps, timestepcur)
  #      self.actdict = actbuilder(self.modeldata, self.busdict, loadpath, timesteps, timestepcur)
        self.linedict = linebuilder(self.modeldata, self.busdict, timesteps)
        self.transdict = transbuilder(self.modeldata, self.busdict, self.subkVAbase, timesteps)
        self.switchdict = switchbuilder(self.modeldata, self.busdict, timesteps)
        self.network = network_mapper(self.modeldata,self.busdict,self.linedict,self.transdict,self.switchdict, depth_dict, leaf_list)
        set_per_unit(self.linedict)
        
class bus:
    def __init__(self,desig,subkVbase_phg,subkVAbase,timesteps):
        self.name = 'bus_' + str(desig)
        self.phases = list()
        self.phasevec = np.zeros((3,1))
        self.type = ''
        self.kVbase_phg = subkVbase_phg
        self.kVAbase = subkVAbase
        self.Zbase = subkVbase_phg*subkVbase_phg*1000/subkVAbase
        self.loads = list()
        self.cap = list()
        self.actuators = list()
        self.edges_out = list()
        self.edges_in = list()
        self.Vsrcs = list()
        
        #self.Vmagsq_linopt = cp.Variable((3,timesteps), name=self.name + '_Vmagsq')
        #self.Vang_linopt = cp.Variable((3,timesteps), self.name + '_Vang')
        
        self.Vmag_NL = np.ones((3,timesteps))
        self.Vang_NL = np.array([[0.],[240.],[120.]])
        for idx in range(timesteps-1): #[HIL] - delete -1? or set timesteps=2 and just use first value
            self.Vang_NL = np.concatenate((self.Vang_NL,np.array([[0.],[240.],[120.]])), axis=1)
                                
class connector:
# Includes lines, transformers, and switches as subclasses
    def __init__(self,timesteps):
        self.phasevec = np.zeros((3,1))
        self.Imag_NL = np.zeros((3,timesteps))
        self.Iang_NL = np.array([[0.],[0.],[0.]])
        for idx in range(timesteps-1):
            self.Iang_NL = np.concatenate((self.Iang_NL,np.array([[0.],[0.],[0.]])), axis=1)
        
class Vsrc:
# These are defined and populated, but as of 7/31/2018 they are not used.
    def __init__(self,desig):
        self.name = 'Vsrc_' + str(desig)
        self.id = ''
        self.phases = list()
        self.kV = complex(0)
        self.angle_phA = complex(0)
        
        self.phasevec = np.zeros((3,1))
        self.Vvec_phg = np.zeros((3,1))
        self.Vangvec = np.zeros((3,1))
        
class shunt:
# As of 7/31/2018, this class only supports cap banks.
    def __init__(self,desig,timesteps):
        self.name = 'cap_' + str(desig)
        self.id = ''
        self.node = None
        self.phases = list()
        self.phasevec = np.zeros((3,1))
        self.kV_phg = complex(0)
        
        self.phasevec = np.zeros((3,1))
        self.Pvec = np.zeros((3,1))
        self.Qvec = np.zeros((3,1))
        self.statusvec = np.zeros((3,1))
        
        self.P_NL = np.zeros((3,timesteps))
        self.Q_NL = np.zeros((3,timesteps))
        
        
class load:
    def __init__(self, desig, timesteps):
        self.name = 'load_' + str(desig)
        self.id = ''
        self.type = ''
        self.node = None
        self.phases = list()
        self.phasevec = np.zeros((3,1))
        self.kV_phph = complex(0)
        self.bandwidth = complex(0)
        self.conn = ''
        self.constZ = complex(0)
        self.constI = complex(0)
        self.constP = complex(0)
        self.status = list()
        
        self.Rsched = np.zeros((3,timesteps))
        self.Xsched = np.zeros((3,timesteps))
        self.Psched = np.zeros((3,timesteps))
        self.Qsched = np.zeros((3,timesteps))
        
        self.P_NL = np.zeros((3,timesteps))
        self.Q_NL = np.zeros((3,timesteps))
        
class actuator:
# As of 7/31/2018, only set up to handle one actuator per node
    def __init__(self, desig, timesteps):
        self.name = 'act_' + str(desig)
        self.node = None
        self.phases = list()
        self.phasevec = np.zeros((3,1))
        
        self.Psched = np.zeros((3,timesteps))
        self.Ssched = np.zeros((3,timesteps))
        
        #self.Pgen = cp.Variable((3,timesteps), name=self.name + '_actP')
        #self.Qgen = cp.Variable((3,timesteps), name=self.name + '_actQ')
        
class line(connector):
    def __init__(self, desig, timesteps):
        connector.__init__(self,timesteps)
        self.name = 'line_' + str(desig)
        self.id = ''
        
        self.from_node = None
        self.from_phases = list()
        self.to_node = None
        self.to_phases = list()
        self.length = complex(0)
        self.kVbase_phg = complex(0)
        self.kVAbase = complex(0)
        self.Zbase = complex(0)
        
        self.R = np.zeros((3, 3))
        self.X = np.zeros((3, 3))
        self.shuntB = np.zeros((3, 3), dtype=np.complex_)
        
        self.Z = np.zeros((3, 3), dtype=np.complex_)
        self.Y = np.zeros((3, 3), dtype=np.complex_)
        
        self.Zpu = np.zeros((3, 3), dtype=np.complex_)
        self.Ypu = np.zeros((3, 3), dtype=np.complex_)
        
        #self.P_linopt = cp.Variable((3,timesteps), self.name + '_P')
        #self.Q_linopt = cp.Variable((3,timesteps), self.name + '_Q')
        
class switch(connector):
    def __init__(self,desig,timesteps):
        connector.__init__(self,timesteps)
        self.name = 'switch_' + str(desig)
        self.from_phases = list()
        self.to_phases = list()
        self.from_node = None
        self.to_node = None
        self.status = list()
        
        self.Z = np.zeros((3, 3), dtype=np.complex_)
        self.Y = np.zeros((3, 3), dtype=np.complex_)
        self.Zpu = np.zeros((3, 3), dtype=np.complex_)
        self.Ypu = np.zeros((3, 3), dtype=np.complex_)

        #self.P_linopt = cp.Variable((3,timesteps), self.name + '_P')
        #self.Q_linopt = cp.Variable((3,timesteps), self.name + '_Q')
        
class transformer(connector):
    def __init__(self,desig,timesteps):
        connector.__init__(self,timesteps)
        self.name = 'transformer_' + str(desig)
        self.id = ''
        
        self.w0_node = None
        self.w0_phases = list()
        self.w0_kVbase_phg = complex(0)
        self.w0_kVAbase = complex(0)
        self.w0_rpu = complex(0)
        self.w0_conn = ''
        self.w1_node = None
        self.w1_phases = list()
        self.w1_kVbase_phg = complex(0)
        self.w1_kVAbase = complex(0)
        self.w1_rpu = complex(0)
        self.w1_conn = ''
        
        self.xpu = complex(0)
        self.tappos = list()
        self.taprange_high = complex(0)
        self.taprange_low = complex(0)
        self.taprangepct_high = complex(0)
        self.taprangepct_low = complex(0)
        
        self.Zpu = np.zeros((3, 3), dtype=np.complex_)
        self.Ypu = np.zeros((3, 3), dtype=np.complex_)
        
        #self.P_linopt = cp.Variable((3,timesteps), self.name + '_P')
        #self.Q_linopt = cp.Variable((3,timesteps), self.name + '_Q')


# In[27]:


def phase2vec(letter):
    # Converts 'a','b','c' to elementary vectors e1, e2, or e3
    possible_letters = ['a', 'b', 'c', 'A', 'B', 'C']
    assert (letter in possible_letters), "Bad call to phase2vec"
    vecout = np.zeros((3,1))
    if letter == 'a' or letter == 'A':
        vecout = vecout + np.array([[1],[0],[0]])
    elif letter == 'b' or letter == 'B':
        vecout = vecout + np.array([[0],[1],[0]])
    elif letter == 'c' or letter == 'C':
        vecout = vecout + np.array([[0],[0],[1]])
    return vecout

def ZtoY(Zmat):
    # Converts an impedance matrix to an admittance matrix
    # Treats '0' impedances on the diagonal as 'line not present' (i.e. temporarily assigns them Z=infinite before inverting)
    tempZ = deepcopy(Zmat)
    for idx in range(0,3):
        if Zmat[idx,idx] == 0:
            tempZ[idx,idx] = np.inf
    return np.linalg.inv(tempZ)
            
def fixconnections(tree,headnode):
    #Iterative approach to prevent maximum recursion depth exceeded error from occuring
    current_level = [headnode]
    while len(current_level) != 0:
        next_level = []
        for current_level_node in current_level:
            for successor in tree.successors(current_level_node):
                if (successor, current_level_node) in tree.edges():
                    tree.remove_edge(successor, current_level_node)
                next_level.append(successor)
        current_level = next_level
    '''
    # Removes all upstream-facing connections from a networkx digraph
    # This will break when we start moving to mesh networks
    succlist = list()
    for inode in tree.successors(headnode):
        succlist.append(inode)
    for inode in succlist:
        if (inode, headnode) in tree.edges():
            tree.remove_edge(inode,headnode)
        #old line causing recursive error
        #for inode2 in tree.successors(inode):
        #this next line used to be indented for for loop
        fixconnections(tree,inode)
        '''

def propogatebasesup(tree,currnode,kVbase_phg):
    # Sets the voltage bases of every node on a path upstream from a given node. Stops if it hits a transformer.
    # 'tree' should be a networkx graph object
    predlist = list()
    currnode.kVbase_phg = kVbase_phg
    currnode.Zbase = kVbase_phg*kVbase_phg*1000/currnode.kVAbase
    for inode in tree.predecessors(currnode):
        predlist.append(inode)      
    for inode in predlist:
        if isinstance(tree[inode][currnode]['connector'],transformer):
            return
        else:
            propogatebasesup(tree,inode,kVbase_phg)
        
def propogatebasesdown(tree,currnode,kVbase_phg):
    # Sets the voltage bases of every node on a path downtream from a given node. Stops if it hits a transformer.
    # 'tree' should be a networkx graph object
    succlist = list()
    currnode.kVbase_phg = kVbase_phg
    currnode.Zbase = kVbase_phg*kVbase_phg*1000/currnode.kVAbase
    for inode in tree.successors(currnode):
        succlist.append(inode)      
    for inode in succlist:
        propogatebasesdown(tree,inode,kVbase_phg)
        
def propogatetrans(tree,currnode):
    # Uses propogatebasesup and down to set appropriate voltage bases throughout a networkx graph based on transformer ratings.
    edgelist = list()
    for iedge in tree.out_edges(currnode): #2to3 out_edges_iter to out_edges
        edgelist.append(iedge)
    for iedge in edgelist:
        conn = tree[iedge[0]][iedge[1]]['connector']
        if isinstance(conn,transformer):
            if currnode == conn.w0_node:
                propogatebasesup(tree,currnode,conn.w0_kVbase_phg)
                propogatebasesdown(tree,iedge[1],conn.w1_kVbase_phg)
            elif currnode == conn.w1_node:
                propogatebasesup(tree,currnode,conn.w1_kVbase_phg)
                propogatebasesdown(tree,iedge[1],conn.w0_kVbase_phg)
    succlist = list()
    for inode in tree.successors(currnode):
        succlist.append(inode)
    for inode in succlist:
        propogatetrans(tree,inode)


# In[28]:


## DATA IMPORT FUNCTIONS


# In[29]:


# CREATING NODES AND LOAD OBJECTS


# In[30]:


# Create bus dictionary from dataframe
def busbuilder(modeldata,subkVbase_phg,subkVAbase,timesteps):
    bussheet = modeldata.parse('Bus') # parse Bus tab of impedance excel
    busdict=dict()
    
    for idx, row in bussheet.iterrows():
        # row['Bus'] is for example 'N_300216892_B'
        indkey = row['Bus'][:len(row['Bus'])-2] # all except the last 2
        indphase = row['Bus'][len(row['Bus'])-1] # pulling out A/B/C substring from each cell in "Bus" column
        if indkey in busdict.keys(): # if bus already in dict, add on the phase
            busdict[indkey].phases.append(indphase)
        else: # if bus not already in dict, add it
            busdict[indkey] = bus(indkey,subkVbase_phg,subkVAbase, timesteps) # create a bus object
            busdict[indkey].type = row['Type'] # add whether it's PQ,slack, or PV bus
            busdict[indkey].phases.append(indphase)

    for key, obj in busdict.items(): # for each multiphase bus
        for idx2 in range(len(obj.phases)): # for each phase
            if obj.phases[idx2] == 'a':
                obj.phasevec = obj.phasevec + np.array([[1],[0],[0]])
            elif obj.phases[idx2] == 'b':
                obj.phasevec = obj.phasevec + np.array([[0],[1],[0]])
            elif obj.phases[idx2] == 'c':
                obj.phasevec = obj.phasevec + np.array([[0],[0],[1]])

    return busdict


# In[31]:


# Create Vsource dictionary from dataframe
# This is a superfluous object since the present (July 2018) code assumes that there is only one Vsrc (the substation)
# So, the current code doesn't use it for anything, but if later models have multiple Vsrcs, we might find it useful
def Vsrcbuilder(modeldata, busdict):
    Vsrcsheet = modeldata.parse('Vsource 3-phase')
    Vsrcdict=dict()
    
    # Initiation
    for idx, row in Vsrcsheet.iterrows():
        if row['bus A']:
            indkey = row['bus A'][:len(row['bus A'])-2]
        elif row['bus B']:
            indkey = row['bus B'][:len(row['bus B'])-2]
        elif row[' bus C']:
            indkey = row['bus C'][:len(row[' bus C'])-2] # Note that there is a leading space in the Excel sheet
        Vsrcdict[indkey] = Vsrc(indkey)
        
        # Check phases
        if row['bus A'] and isinstance(row['bus A'],str):
            Vsrcdict[indkey].phases.append('a')
        if row['bus B'] and isinstance(row['bus B'],str):
            Vsrcdict[indkey].phases.append('b')
        if row[' bus C'] and isinstance(row[' bus C'],str): # Note that there is a leading space in the Excel sheet
            Vsrcdict[indkey].phases.append('c')
            
        # Set angles. This assumes that "Angle A" is the only one populated, (the only form I've seen so far)
        angle_phA = row[' Angle_a (Degree)']
        Vsrcdict[indkey].Vangvec[0,0] = angle_phA
        Vsrcdict[indkey].Vangvec[1,0] = angle_phA + 240.
        Vsrcdict[indkey].Vangvec[2,0] = angle_phA + 120.     
        
        # Misc field, likely not useful
        Vsrcdict[indkey].id = row['ID'] 
        
        # Set voltages
        Vsrcdict[indkey].kV_phph = row['kV (ph-ph RMS)']
        Vsrcdict[indkey].Vvec_phg[0,0] = Vsrcdict[indkey].kV_phph/np.sqrt(3)
        Vsrcdict[indkey].Vvec_phg[1,0] = Vsrcdict[indkey].kV_phph/np.sqrt(3)
        Vsrcdict[indkey].Vvec_phg[2,0] = Vsrcdict[indkey].kV_phph/np.sqrt(3)
        
        # Append object to buslist 
        busdict[indkey].Vsrcs.append(Vsrcdict[indkey])
    
    # Populate phase vectors
    for key, obj in Vsrcdict.items():
        for idx2 in range(len(obj.phases)):
            obj.phasevec = obj.phasevec + phase2vec(obj.phases[idx2])

    return Vsrcdict


# In[32]:


# Create shunt elements dictionary from dataframe
# Currently (July 2018) this object handles constant power capacitors only
def shuntbuilder(modeldata, busdict,timesteps):
    shuntsheet = modeldata.parse('Multiphase Shunt')
    shuntdict = dict()
    
    for idx, row in shuntsheet.iterrows():
        if row['Bus1'] and isinstance(row['Bus1'],str):
            indkey = row['Bus1'][:len(row['Bus1'])-2]
        elif row['Bus2'] and isinstance(row['Bus2'],str):
            indkey = row['Bus2'][:len(row['Bus2'])-2]
        elif row['Bus3'] and isinstance(row['Bus3'],str):
            indkey = row['Bus3'][:len(row['Bus3'])-2]

        shuntdict[indkey] = shunt(indkey,timesteps)

        if row['Bus1'] and isinstance(row['Bus1'],str):
            indphase = row['Bus1'][len(row['Bus1'])-1]
            shuntdict[indkey].phases.append(indphase)

        if row['Bus2'] and isinstance(row['Bus2'],str):
            indphase = row['Bus2'][len(row['Bus2'])-1]
            shuntdict[indkey].phases.append(indphase)

        if row['Bus3'] and isinstance(row['Bus3'],str):
            indphase = row['Bus3'][len(row['Bus3'])-1]
            shuntdict[indkey].phases.append(indphase)

        shuntdict[indkey].id = row['ID']
        shuntdict[indkey].node = busdict[indkey]
        shuntdict[indkey].kV_phg = row['kV (ph-gr RMS)']
# 2to3 long to int in (isinstance(row['P1(kW)'],long)
        if (row['P1(kW)']) and (not np.isnan(row['P1(kW)'])) and (isinstance(row['P1(kW)'],int) or isinstance(row['P1(kW)'],float)): 
            # Note that spacings in P,Q headings are very inconsistent
            shuntdict[indkey].Pvec = shuntdict[indkey].Pvec + row['P1(kW)'] * phase2vec(shuntdict[indkey].phases[0])

        if row['P2 (kW)'] and (not np.isnan(row['P2 (kW)'])) and (isinstance(row['P2 (kW)'],int) or isinstance(row['P2 (kW)'],float)): 
            shuntdict[indkey].Pvec = shuntdict[indkey].Pvec + row['P2 (kW)'] * phase2vec(shuntdict[indkey].phases[1]) 
            
        if row['P3(kW)'] and (not np.isnan(row['P3(kW)'])) and (isinstance(row['P3(kW)'],int) or isinstance(row['P3(kW)'],float)): 
            shuntdict[indkey].Pvec = shuntdict[indkey].Pvec + row['P3(kW)'] * phase2vec(shuntdict[indkey].phases[2])  
            
        if row['Q1(kVAr)'] and (not np.isnan(row['Q1(kVAr)'])) and (isinstance(row['Q1(kVAr)'],int) or isinstance(row['Q1(kVAr)'],float)): 
            shuntdict[indkey].Qvec = shuntdict[indkey].Qvec + row['Q1(kVAr)'] * phase2vec(shuntdict[indkey].phases[0]) 
            
        if row['Q2 (kVAr)'] and (not np.isnan(row['Q2 (kVAr)'])) and (isinstance(row['Q2 (kVAr)'],int) or isinstance(row['Q2 (kVAr)'],float)): 
            shuntdict[indkey].Qvec = shuntdict[indkey].Qvec + row['Q2 (kVAr)'] * phase2vec(shuntdict[indkey].phases[1])
            
        if row['Q3 (kVAr)'] and (not np.isnan(row['Q3 (kVAr)'])) and (isinstance(row['Q3 (kVAr)'],int) or isinstance(row['Q3 (kVAr)'],float)): 
            shuntdict[indkey].Qvec = shuntdict[indkey].Qvec + row['Q3 (kVAr)'] * phase2vec(shuntdict[indkey].phases[2])

        if row['Status1'] and row['Status1'] == 1:
            shuntdict[indkey].statusvec = shuntdict[indkey].statusvec + phase2vec(shuntdict[indkey].phases[0])                                        
        if row['Status2'] and row['Status2'] == 1:
            assert(len(shuntdict[indkey].phases) >= 2)
            shuntdict[indkey].statusvec = shuntdict[indkey].statusvec + phase2vec(shuntdict[indkey].phases[1])
        if row['Status3'] and row['Status3'] == 1:
            assert(len(shuntdict[indkey].phases) == 3)
            shuntdict[indkey].statusvec = shuntdict[indkey].statusvec + phase2vec(shuntdict[indkey].phases[2])
            
        busdict[indkey].cap.append(shuntdict[indkey])
            
    for key, obj in shuntdict.items():
        for idx2 in range(len(obj.phases)):       
            obj.phasevec = obj.phasevec + phase2vec(obj.phases[idx2])
    
    return shuntdict


# In[33]:


# Create load dictionary from ePHASORsim model and Gridbright load files
def loadbuilderPQ(modeldata, busdict, loadpath, timesteps, timestepcur):
    
    # This creates the load objects from the ePHASORsim model, but ignores the specified values (values are set with a Gridbright load file)
    loadsheet = modeldata.parse('Multiphase Load')
    loaddict = dict()
    
    for idx, row in loadsheet.iterrows():          
        if row['Bus1']:
            indkey = row['Bus1'][:len(row['Bus1'])-2]
            loadname = row['Bus1']
        elif row['Bus2']:
            indkey = row['Bus2'][:len(row['Bus2'])-2]
            loadname = row['Bus2']
        elif row['Bus3']:
            indkey = row['Bus3'][:len(row['Bus3'])-2]
            loadname = row['Bus3']

        loaddict[indkey] = load(indkey, timesteps)
        if row['Bus1'] and isinstance(row['Bus1'],str):
            indphase = row['Bus1'][len(row['Bus1'])-1]
            loaddict[indkey].phases.append(indphase)

        if row['Bus2'] and isinstance(row['Bus2'],str):
            indphase = row['Bus2'][len(row['Bus2'])-1]
            loaddict[indkey].phases.append(indphase)

        if row['Bus3'] and isinstance(row['Bus3'],str):
            indphase = row['Bus3'][len(row['Bus3'])-1]
            loaddict[indkey].phases.append(indphase)
        
        loaddict[indkey].id = row['ID']
        loaddict[indkey].node = busdict[indkey]
        loaddict[indkey].type = row['Type']
        loaddict[indkey].kV_phph = row['V (kV)']
        loaddict[indkey].bandwidth = row['Bandwidth (pu)']
        loaddict[indkey].conn = row['Conn. type']
        loaddict[indkey].constZ = row['K_z']
        loaddict[indkey].constI = row['K_i']
        loaddict[indkey].constP = row['K_p']
        loaddict[indkey].status = row['Status']
        
        
        busdict[indkey].loads.append(loaddict[indkey])
        
    #loadfile = pd.read_csv(loadpath)
    #loaddf = loadfile.parse('Time_Series_data')
    #[HIL] - parse out current timestep

# jaimie removed these lines 10/19/21:
#     loaddf = pd.read_csv(loadpath)
#     loaddf = loaddf[timestepcur:timestepcur+timesteps]
    
#     loaddf.index -= timestepcur
    

    # Populate the load dictionary's P and Q schedules with a Gridbright load file
    
#JPEDIT START
    
# Jaimie removed this 10/19/21
#     multiph = ['1','2','3'] ##generalize to 1,2,3 instead of first second third
#     for key, iload in loaddict.items():
#         Pkey = []
#         Qkey = []
#         phkey = []
#         for header in loaddf.columns:
#             if key in header:
#                 #for mph in multiph:
#                 if '/P' in header:
#                     Pkey.append(header)
#                 if '/Q' in header:
#                     Qkey.append(header)
#         for ph in iload.phases:
#             phkey.append(ph)
#         for idx in range(len(phkey)):
#             for ts in range(0,timesteps):  #write with dict??
#                 if idx+1 > len(Pkey):
#                     kW = 0
#                     kVar = 0
#                 else: 
#                     kW = loaddf[Pkey[idx]][ts]
#                     kVAR = loaddf[Qkey[idx]][ts]
                    
#                 S = kW + 1j*kVAR
                    
#                 if not (S==0):
#                     Z = np.conj(np.square(iload.node.kVbase_phg)*1000/S)
#                 else:
#                     Z = 0
                
#                 if phkey[idx] == 'a':
#                     iload.Psched[0,ts] = kW
#                     iload.Qsched[0,ts] = kVAR
    
#                     iload.Rsched[0,ts] = np.real(Z)
#                     iload.Xsched[0,ts] = np.imag(Z)
    
#                 if phkey[idx] == 'b':
#                     iload.Psched[1,ts] = kW
#                     iload.Qsched[1,ts] = kVAR
    
#                     iload.Rsched[1,ts] = np.real(Z)
#                     iload.Xsched[1,ts] = np.imag(Z)
    
#                 if phkey[idx] == 'c':
#                     iload.Psched[2,ts] = kW
#                     iload.Qsched[2,ts] = kVAR
    
#                     iload.Rsched[2,ts] = np.real(Z)
#                     iload.Xsched[2,ts] = np.imag(Z)
                
#JPEDIT END

#EDITED SECTION START               
    '''         
    for key, iload in loaddict.items():
        for ph in iload.phases:
            for ts in range(0,timesteps):
                
                #loadname_kW = 'LD_KW_' + key + '_' + ph
                #loadname_kVAR = 'LD_KV_' + key + '_' + ph
    


#Jaimie
                loadname_kW = 'LD_' + key + '/P_' + ph
                loadname_kVAR = 'LD_' + key + '/Q_' + ph

                if loadname_kW in list(loaddf.columns.values):
                    kW = loaddf[loadname_kW][ts]
                else:
                    kW = 0

                if loadname_kVAR in list(loaddf.columns.values):
                    kVAR = loaddf[loadname_kVAR][ts]
                else:
                    kVAR = 0

                S = kW + 1j*kVAR
                
                if not (S==0):
                    Z = np.conj(np.square(iload.node.kVbase_phg)*1000/S)
                else:
                    Z = 0

                if ph == 'a':
                    iload.Psched[0,ts] = kW
                    iload.Qsched[0,ts] = kVAR

                    iload.Rsched[0,ts] = np.real(Z)
                    iload.Xsched[0,ts] = np.imag(Z)

                if ph == 'b':
                    iload.Psched[1,ts] = kW
                    iload.Qsched[1,ts] = kVAR

                    iload.Rsched[1,ts] = np.real(Z)
                    iload.Xsched[1,ts] = np.imag(Z)

                if ph == 'c':
                    iload.Psched[2,ts] = kW
                    iload.Qsched[2,ts] = kVAR

                    iload.Rsched[2,ts] = np.real(Z)
                    iload.Xsched[2,ts] = np.imag(Z)
                    
                    '''
#EDITED SECTION END

    for key, iload in loaddict.items():
        for idx in range(len(iload.phases)):
            if iload.phases[idx] == 'a':
                iload.phasevec = iload.phasevec + np.array([[1],[0],[0]])
            elif iload.phases[idx] == 'b':
                iload.phasevec = iload.phasevec + np.array([[0],[1],[0]])
            elif iload.phases[idx] == 'c':
                iload.phasevec = iload.phasevec + np.array([[0],[0],[1]])
            
    return loaddict


# In[34]:


# Create actuator dictionary from Gridbright load file
# This interprets all 'PV_KW_' entries in the load file as controllable PV. 
# If the PV is uncontrollable, it should be included as negative load in the load P and Q schedules
def actbuilder(modeldata, busdict, actpath, timesteps, timestepcur):
    
    actdict = dict()
    
# Jaimie removing actuators on 10/19/21
#     #actfile = pd.read_csv(actpath)
#     actdf = pd.read_csv(actpath)
#     #[HIL] - parse current timestep
#     actdf = actdf[timestepcur:timestepcur+timesteps]
#     actdf.index -= timestepcur
    
#     for key,ibus in busdict.items():
#         for ph in ibus.phases:
#             if 'act_kVA_' + key + '_' + ph in list(actdf.columns.values):
#                 if not (key in actdict):
#                     actdict[key] = actuator(key, timesteps)
#                     actdict[key].node = busdict[key]
#                     ibus.actuators.append(actdict[key])
#                 actdict[key].phases.append(ph)
    
#     for key, iact in actdict.items():
#         for ph in iact.phases:
#             for ts in range(0,timesteps):
#                 actname_kVA = 'act_kVA_' + key + '_' + ph
#                 act = actdf[actname_kVA][ts]

#                 if ph == 'a':
#                     iact.Psched[0,ts] = act
#                     iact.Ssched[0,ts] = act
#                 if ph == 'b':
#                     iact.Psched[1,ts] = act
#                     iact.Ssched[1,ts] = act
#                 if ph == 'c':
#                     iact.Psched[2,ts] = act
#                     iact.Ssched[2,ts] = act
        
#     for key, iact in actdict.items():
#         for idx in range(len(iact.phases)):
#             if iact.phases[idx] == 'a':
#                 iact.phasevec = iact.phasevec + np.array([[1],[0],[0]])
#             elif iact.phases[idx] == 'b':
#                 iact.phasevec = iact.phasevec + np.array([[0],[1],[0]])
#             elif iact.phases[idx] == 'c':
#                 iact.phasevec = iact.phasevec + np.array([[0],[0],[1]])
    
    return actdict


# In[35]:


# LINES AND CONNECTOR OBJECTS


# In[36]:


# Create line dictionary
# Note that this isn't set up to handle cross-phase connections: i.e. please don't connect phase A at node 1 to phase B at node 2
def linebuilder(modeldata, busdict, timesteps):
    linesheet = modeldata.parse('Multiphase Line')
    # Create line dictionary from dataframe
    linedict = dict()
    for idx, row in linesheet.iterrows():
        if row['From1']:
            indkeyfrom = row['From1'][:len(row['From1'])-2]       
        elif row['From2']:
            indkeyfrom = row['From2'][:len(row['From2'])-2]
        elif row['From3']:
            indkeyfrom = row['From3'][:len(row['From3'])-2] 

        if row['To1']:
            indkeyto = row['To1'][:len(row['To1'])-2]       
        elif row['To2']:
            indkeyto = row['To2'][:len(row['To2'])-2]
        elif row['To3']:
            indkeyto = row['To3'][:len(row['To3'])-2]     

        indkey = indkeyfrom + 'to' + indkeyto    
        linedict[indkey] = line(indkey, timesteps)
        
        busdict[indkeyfrom].edges_out.append(linedict[indkey])
        busdict[indkeyto].edges_in.append(linedict[indkey])
        
        linedict[indkey].id = row['ID']
        
        length_unit = ' (length_unit)'
        ohm_per_lu = ' (ohm/length_unit)'
        us_per_lu = ' (uS/length_unit)'
        
        if 'Length (length_unit)' not in row:
            length_unit = ' (Mile)'
            ohm_per_lu = ' (ohm/Mile)'
            us_per_lu = ' (uS/Mile)'
        
        
        linedict[indkey].length = row['Length' + length_unit]
        

        linedict[indkey].from_node = busdict[indkeyfrom]
        linedict[indkey].to_node = busdict[indkeyto]
        
        if row['From1'] and isinstance(row['From1'],str):
            linedict[indkey].from_phases.append(row['From1'][len(row['From1'])-1].lower())    
        if row['From2'] and isinstance(row['From2'],str):
            linedict[indkey].from_phases.append(row['From2'][len(row['From2'])-1].lower())   
        if row['From3'] and isinstance(row['From3'],str):
            linedict[indkey].from_phases.append(row['From3'][len(row['From3'])-1].lower()) 

        if row['To1'] and isinstance(row['To1'],str):
            linedict[indkey].to_phases.append(row['To1'][len(row['To1'])-1].lower())    
        if row['To2'] and isinstance(row['To2'],str):
            linedict[indkey].to_phases.append(row['To2'][len(row['To2'])-1].lower())   
        if row['To3'] and isinstance(row['To3'],str):
            linedict[indkey].to_phases.append(row['To3'][len(row['To3'])-1].lower()) 
            
        for idx in range(0,len(linedict[indkey].to_phases)):
            linedict[indkey].phasevec = linedict[indkey].phasevec + phase2vec(linedict[indkey].to_phases[idx])
            
        # Build R, X, shunt Y matrices (shunt Y are not included in LUPFM)
        rdict = dict()
        rdict['11'] = row['r11' + ohm_per_lu]
        rdict['21'] = row['r21' + ohm_per_lu]
        rdict['22'] = row['r22' + ohm_per_lu]
        rdict['31'] = row['r31' + ohm_per_lu]
        rdict['32'] = row['r32' + ohm_per_lu]
        rdict['33'] = row['r33' + ohm_per_lu]
        rdict['12'] = rdict['21']
        rdict['13'] = rdict['31']
        rdict['23'] = rdict['32']

        xdict = dict()
        xdict['11'] = row['x11' + ohm_per_lu]
        xdict['21'] = row['x21' + ohm_per_lu]
        xdict['22'] = row['x22' + ohm_per_lu]
        xdict['31'] = row['x31' + ohm_per_lu]
        xdict['32'] = row['x32' + ohm_per_lu]
        xdict['33'] = row['x33' + ohm_per_lu]
        xdict['12'] = xdict['21']
        xdict['13'] = xdict['31']
        xdict['23'] = xdict['32']

        bdict = dict()
        bdict['11'] = row['b11' + us_per_lu]
        bdict['21'] = row['b21' + us_per_lu]
        bdict['22'] = row['b22' + us_per_lu]
        bdict['31'] = row['b31' + us_per_lu]
        bdict['32'] = row['b32' + us_per_lu]
        bdict['33'] = row['b33' + us_per_lu]
        bdict['12'] = bdict['21']
        bdict['13'] = bdict['31']
        bdict['23'] = bdict['32']

        frm_ph_lst = list(map(string.ascii_lowercase.index,linedict[indkey].from_phases))
        to_ph_lst = list(map(string.ascii_lowercase.index,linedict[indkey].to_phases))
        for fromidx in range(0, len(frm_ph_lst)):
            for toidx in range(0, len(to_ph_lst)):
                linedict[indkey].R[frm_ph_lst[fromidx], to_ph_lst[toidx]] = rdict[str(fromidx + 1) + str(toidx + 1)]
                linedict[indkey].R[to_ph_lst[toidx], frm_ph_lst[fromidx]] = rdict[str(fromidx + 1) + str(toidx + 1)]
                linedict[indkey].X[to_ph_lst[toidx], frm_ph_lst[fromidx]] = xdict[str(fromidx + 1) + str(toidx + 1)]
                linedict[indkey].X[frm_ph_lst[fromidx], to_ph_lst[toidx]] = xdict[str(fromidx + 1) + str(toidx + 1)]
        linedict[indkey].R = np.nan_to_num(np.multiply(linedict[indkey].R,linedict[indkey].length))
        linedict[indkey].X = np.nan_to_num(np.multiply(linedict[indkey].X,linedict[indkey].length))
        
        linedict[indkey].Z = linedict[indkey].R + 1j*linedict[indkey].X
        linedict[indkey].Y = ZtoY(linedict[indkey].Z)
        # Line per-unit quantities are set later in this script, once voltage bases have been established.
    
    return linedict


# In[37]:


# Create switch dictionary
def switchbuilder(modeldata, busdict, timesteps):
    switchsheet = modeldata.parse('Switch')
    
    switchdict = dict()
    for idx, row in switchsheet.iterrows():
        if row['Status'] != 0: #only create node/edge if switch is closed
            #uscoreposfrom = row['From Bus'].find('_')            
            #endfrom = len(row['From Bus'])-2 if uscoreposfrom == len(row['From Bus'])-2 else len(row['From Bus'])-1            
            indkeyfrom = row['From Bus'][:len(row['From Bus'])-2]
            
            #uscoreposto = row['To Bus'].find('_')            
            #ndto = len(row['To Bus'])-2 if uscoreposto == len(row['To Bus'])-2 else len(row['To Bus'])-1
            indkeyto = row['To Bus'][:len(row['To Bus'])-2]

            indphasefrom = row['From Bus'][len(row['From Bus'])-1].lower()
            indphaseto = row['To Bus'][len(row['To Bus'])-1].lower()

            indkey = indkeyfrom + 'to' + indkeyto

            if indkey in switchdict.keys(): # if already in switch list
                switchdict[indkey].from_phases.append(indphasefrom)
                switchdict[indkey].to_phases.append(indphaseto)

            else:
                switchdict[indkey] = switch(indkey, timesteps)
                switchdict[indkey].from_phases.append(indphasefrom)
                switchdict[indkey].to_phases.append(indphaseto)

                switchdict[indkey].from_node = busdict[indkeyfrom]
                switchdict[indkey].to_node = busdict[indkeyto]
                switchdict[indkey].status.append(row['Status'])

                busdict[indkeyfrom].edges_out.append(switchdict[indkey])
                busdict[indkeyto].edges_in.append(switchdict[indkey])
        
    # This next loop is ugly, but it works (it's the result of fixing a bug)
    for indkey, entry in switchdict.items():
        for idx in range(0,len(switchdict[indkey].to_phases)):
            switchdict[indkey].phasevec = switchdict[indkey].phasevec + phase2vec(switchdict[indkey].to_phases[idx])
            
        # Build Z, Y matrices (per unit)
        assert(switchdict[indkey].from_node.Zbase == switchdict[indkey].to_node.Zbase)
        Zbase = switchdict[indkey].from_node.Zbase
 
        tempz = (1e-4)/Zbase # Based off of the DSS switch approximation: r1=r0=1e-4
        tempy = 1/tempz

        from_ph_lst = list(map(string.ascii_lowercase.index,switchdict[indkey].from_phases))
        to_ph_lst = list(map(string.ascii_lowercase.index,switchdict[indkey].to_phases))

        for idx in range(0, len(from_ph_lst)): # set impedance over switch to be tiny value, not zero
            switchdict[indkey].Z[from_ph_lst[idx], to_ph_lst[idx]] = 1e-4
            switchdict[indkey].Y[from_ph_lst[idx], to_ph_lst[idx]] = 1e4
            switchdict[indkey].Zpu[from_ph_lst[idx], to_ph_lst[idx]] = tempz
            switchdict[indkey].Ypu[from_ph_lst[idx], to_ph_lst[idx]] = tempy
            
    return switchdict


# In[38]:

#Phase is a character among {'a', 'b', 'c'}
#Accomodates cross-phsae connections
def trans_helper(indkeyw0, indkeyw1, transdict, busdict, timesteps, subkVAbase, row):
    '''
    These are all of the following cases for a transformer:
    '''
    '''
    bus_a_w0_side = None
    bus_a_w0_side_phase = None
    
    bus_b_w0_side = None
    bus_b_w0_side_phase = None
    
    bus_c_w0_side = None
    bus_c_w0_side_phase = None
    
    
    w0_phase_dict = {}
    
    if row['w0_bus a']:
        bus_a_w0_side = row['w0_bus a'][:len(row['w0_bus a'])-2]
        bus_a_w0_side_phase = row['w0_bus a'][len(row['w0_bus a'])-1].lower()
        
        w0_phase_dict[bus_a_w0_side] = [bus_a_w0_side_phase]
    if row['w0_bus b']:
        bus_b_w0_side = row['w0_bus b'][:len(row['w0_bus b'])-2]
        bus_b_w0_side_phase = row['w0_bus b'][len(row['w0_bus b'])-1].lower()
        
        if bus_b_w0_side in w0_phase_dict.keys():
            w0_phase_dict[bus_b_w0_side].append(bus_b_w0_side_phase)
        else:
            w0_phase_dict[bus_b_w0_side] = [bus_b_w0_side_phase]
        
    if row['w0_bus c']:
        bus_c_w0_side = row['w0_bus c'][:len(row['w0_bus c'])-2]
        bus_c_w0_side_phase = row['w0_bus c'][len(row['w0_bus c'])-1].lower()
        
        if bus_c_w0_side in w0_phase_dict.keys():
            w0_phase_dict[bus_c_w0_side].append(bus_c_w0_side_phase)
        else:
            w0_phase_dict[bus_c_w0_side] = [bus_c_w0_side_phase]
        
    
    bus_a_w1_side = None
    bus_a_w1_side_phase = None
    
    bus_b_w1_side = None
    bus_b_w1_side_phase = None
    
    bus_c_w1_side = None
    bus_c_w1_side_phase = None
    
    w1_phase_dict = {}
    
    if row['w1_bus_a']:
        bus_a_w1_side = row['w1_bus_a'][:len(row['w1_bus_a'])-2]
        bus_a_w1_side_phase = row['w1_bus_a'][len(row['w1_bus_a'])-1].lower()
        
        w1_phase_dict[bus_a_w1_side] = [bus_a_w1_side_phase]
    if row['w1_bus_b']:
        bus_b_w1_side = row['w1_bus_b'][:len(row['w1_bus_b'])-2]
        bus_b_w1_side_phase = row['w1_bus_b'][len(row['w1_bus_b'])-1].lower()
        
        if bus_b_w1_side in w1_phase_dict.keys():
            w1_phase_dict[bus_b_w1_side].append(bus_b_w1_side_phase)
        else:
            w1_phase_dict[bus_b_w1_side] = [bus_b_w1_side_phase]
        
    if row['w1_bus_c']:
        bus_c_w1_side = row['w1_bus_c'][:len(row['w1_bus_c'])-2]
        bus_c_w1_side_phase = row['w1_bus_c'][len(row['w1_bus_c'])-1].lower()
        
        if bus_c_w1_side in w1_phase_dict.keys():
            w1_phase_dict[bus_c_w1_side].append(bus_c_w1_side_phase)
        else:
            w1_phase_dict[bus_c_w1_side] = [bus_c_w1_side_phase]
            '''
    pass


# Create transformer dictionary from dataframe
def transbuilder(modeldata,busdict,subkVAbase,timesteps):
    transsheet = modeldata.parse('Multiphase Transformer')
    #[HIL] - NEW CODE index misalignment during parse
    if transsheet.iloc[0][0] == 'ID':
        transsheet = modeldata.parse('Multiphase Transformer', index_col=0)
    
    # Prep transformer column headers (This is built to handle a 2-winding transformer)
    w0_start = transsheet.columns.get_loc('winding 0')
    w1_start = transsheet.columns.get_loc('winding 1')
    
    #print("w0 start is " + str(w0_start))
    
    windcols = w1_start - w0_start
    othercols = len(transsheet.columns) - transsheet.columns.get_loc('winding 1') - windcols
    
    for idx in range(w0_start, w0_start + windcols):
        #print("Transforming into w0_" + transsheet.iloc[0][idx])
        transsheet.iloc[0][idx] = 'w0_' + transsheet.iloc[0][idx]
    for idx in range(w0_start + windcols, w0_start + 2 * windcols):
        #print("Transforming into w1_" + transsheet.iloc[0][idx])
        transsheet.iloc[0][idx] = 'w1_' + transsheet.iloc[0][idx]
    transsheet.columns = transsheet.iloc[0]
    transsheet = transsheet.drop(transsheet.index[0]);
    
    # Create dictionary
    transdict = dict()
    
    for idx, row in transsheet.iterrows():
        if row['w0_bus a']:
            indkeyw0 = row['w0_bus a'][:len(row['w0_bus a'])-2]       
        elif row['w0_bus b']:
            indkeyw0 = row['w0_bus b'][:len(row['w0_bus b'])-2]
        elif row['w0_bus c']:
            indkeyw0 = row['w0_bus c'][:len(row['w0_bus c'])-2] 

        if row['w1_bus_a']:
            indkeyw1 = row['w1_bus_a'][:len(row['w1_bus_a'])-2]       
        elif row['w1_bus_b']:
            indkeyw1 = row['w1_bus_b'][:len(row['w1_bus_b'])-2]
        elif row['w1_bus_c']:
            indkeyw1 = row['w1_bus_c'][:len(row['w1_bus_c'])-2]     
        
        
        indkey = indkeyw0 + 'to' + indkeyw1
        #print(indkey)
    
        transdict[indkey] = transformer(indkey, timesteps)
        transdict[indkey].id = idx
        
        busdict[indkeyw0].edges_out.append(transdict[indkey])
        busdict[indkeyw1].edges_in.append(transdict[indkey])
        
        if row['w0_bus a']:
            transdict[indkey].w0_phases.append('a')    
        if row['w0_bus b']:
            transdict[indkey].w0_phases.append('b')   
        if row['w0_bus c']:
            transdict[indkey].w0_phases.append('c')
        
        transdict[indkey].w0_name = indkeyw0
        transdict[indkey].w0_node = busdict[indkeyw0]
    
        transdict[indkey].w0_kVbase_phg = row['w0_kV (ph-ph RMS)']/np.sqrt(3) 
        transdict[indkey].w0_kVAbase = row['w0_kVA_base']
        transdict[indkey].w0_rpu = row['w0_R_w0 (pu)'] 
        #transdict[indkey].w0_conn = row[headmap['w0_conn']]
        
        transdict[indkey].w0_conn = row['w0_conn'] #[HIL] - edit, error

        if row['w1_bus_a']:
            transdict[indkey].w1_phases.append('a')    
        if row['w1_bus_b']:
            transdict[indkey].w1_phases.append('b')   
        if row['w1_bus_c']:
            transdict[indkey].w1_phases.append('c')
            
        for idx in range(0,len(transdict[indkey].w1_phases)):
            transdict[indkey].phasevec = transdict[indkey].phasevec + phase2vec(transdict[indkey].w1_phases[idx])

        transdict[indkey].w1_name = indkeyw1
        transdict[indkey].w1_node = busdict[indkeyw1]
    
        transdict[indkey].w1_kVbase_phg = row['w1_kV (ph-ph RMS)']/np.sqrt(3) 
        transdict[indkey].w1_kVAbase = row['w1_kVA_base'] 
        transdict[indkey].w1_rpu = row['w1_R_w1 (pu)'] 
        transdict[indkey].w1_conn = row['w1_conn'] 
        
        transdict[indkey].xpu = row['X (pu)'] 

        if row['Tap A']:
            transdict[indkey].tappos.append(row['Tap A'])
        else:
            transdict[indkey].tappos.append(0)
        if row['Tap B']:
            transdict[indkey].tappos.append(row['Tap B'])
        else:
            transdict[indkey].tappos.append(0)
        if row['Tap C']:
            transdict[indkey].tappos.append(row['Tap C'])
        else:
            transdict[indkey].tappos.append(0)
    
        transdict[indkey].taprange_high = row['Highest Tap'] 
        transdict[indkey].taprange_low = row['Lowest Tap']
        transdict[indkey].taprangepct_high = row['Max Range (%)']
        transdict[indkey].taprangepct_low = row['Min Range (%)']
        
        # Build Z, Y matrices
        #assert(transdict[indkey].w1_kVAbase == transdict[indkey].w0_kVAbase)  # dont need this true
        tempz = (transdict[indkey].w0_rpu + transdict[indkey].w1_rpu + 1j*transdict[indkey].xpu)*subkVAbase/transdict[indkey].w0_kVAbase
        tempy = 1/tempz

    
        w0_ph_lst = list(map(string.ascii_lowercase.index,transdict[indkey].w0_phases))
        w1_ph_lst = list(map(string.ascii_lowercase.index,transdict[indkey].w1_phases))
    
        for idx in range(0, len(w0_ph_lst)):
            transdict[indkey].Zpu[w0_ph_lst[idx], w1_ph_lst[idx]] = tempz
            transdict[indkey].Ypu[w0_ph_lst[idx], w1_ph_lst[idx]] = tempy
            
    return transdict


# In[39]:


## NETWORK MAP (USED TO DEFINE VOLTAGE BASES FROM TRANSFORMER CONNECTIVITY)


# In[40]:


def network_mapper(modeldata,busdict,linedict,transdict,switchdict, depth_dict, leaf_list):
    # Create networkx digraph with all connections pointing both ways
    network = nx.DiGraph()
    
    
    slack = None
    
    all_edges = {}
    for node in busdict.values():
        #Assign slack node one time
        if slack == None and (node.type == 'SLACK' or node.type == 'Slack' or node.type == 'slack'):
            slack = node
        #Create an empty list of connected vertices for each node
        all_edges[node.name] = []
        #Add node to graph
        network.add_node(node.name)
    
    #Make sure slack bus was found properly
    assert(slack != None)
        
    for value in linedict.values(): # depths dictionary includes all nodes mentioned in line tab, transformer tab, and switch tab
        #print("Value of line is " + str(value))
        all_edges[value.from_node.name] += [[value.to_node.name, value]]
        all_edges[value.to_node.name] += [[value.from_node.name, value]]
    for value in transdict.values():
        all_edges[value.w0_node.name] += [[value.w1_node.name, value]]
        all_edges[value.w1_node.name] += [[value.w0_node.name, value]]
    for value in switchdict.values():
        all_edges[value.from_node.name] += [[value.to_node.name, value]]
        all_edges[value.to_node.name] += [[value.from_node.name, value]]
    
    
    #Iteratively delete extra edges, starting from the slack node as the root; ASSUMES NO LOOPS ARE PRESENT, i.e. that
    #the format of the feeder is that of a tree
    current_level_pairs = [[slack.name, None]]
    
    depth = 0 # start at substation and move downward
       
    while len(current_level_pairs) != 0:
        next_level_pairs = []
        for current_level_pair in current_level_pairs:
            depth_dict[current_level_pair[0]] = depth
            if len(all_edges[current_level_pair[0]]) == 0:
                leaf_list.append(current_level_pair[0])
            for next_level_pair in all_edges[current_level_pair[0]]:
                if [current_level_pair[0], next_level_pair[1]] in all_edges[next_level_pair[0]]:                
                    all_edges[next_level_pair[0]].remove([current_level_pair[0], next_level_pair[1]])
                next_level_pairs += [next_level_pair]
        depth += 1 # move down the tree by one level
        current_level_pairs = next_level_pairs
    
    
    for node, destinations in all_edges.items():
        for destination_pair in destinations:
            #print("Destination connector is " + str(destination_pair[1]))
            network.add_edge(node, destination_pair[0], connector = destination_pair[1])
    
    #Connections are already "fixed" prior to adding to the network digraph; this implementation is faster than removing
    #extra edges after they have already been added to the graph; thus, fixconnections is no longer needed
    
    #propogatetrans(network,slack)
    
    
    return network



# In[41]:


def set_per_unit(linedict):
    # Set the values of Zpu and Ypu matrices for lines
    for key, iconn in linedict.items():
        assert (iconn.from_node.kVbase_phg == iconn.to_node.kVbase_phg and iconn.from_node.kVAbase == iconn.to_node.kVAbase), "Base mismatch " + iconn.name
        iconn.kVbase_phg = iconn.from_node.kVbase_phg
        iconn.kVAbase = iconn.from_node.kVAbase
        iconn.Zbase = iconn.kVbase_phg*iconn.kVbase_phg*1000/iconn.kVAbase
        iconn.Zpu = (iconn.R + 1j*iconn.X)/iconn.Zbase
        iconn.Ypu = ZtoY(iconn.Zpu)
        
        
# In[42]:
        
# add costs for actuators
'''
def add_act_costs


'''
