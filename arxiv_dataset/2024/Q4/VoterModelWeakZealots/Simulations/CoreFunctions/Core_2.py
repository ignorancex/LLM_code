import networkx as nx

from pylab import *

import random

import copy

import matplotlib.pyplot as plt

import time

import scipy.integrate as integrate

import sys



def Initialise(n,p,P):
    """Generates Network and key parameters of that network

    Arguments:
        n:      the number of nodes
        p:      the probability two nodes are connected
        P:      the proportion of zealots
    
    Returns:
        positions:          List positions of nodes useful for plotting
        CompleteGraph:      CompleteGraph networkx object
        RandomGraph:        RandomGraph Networkx object
        InactivePatchIDs:   List ID's of inactive Zealots
    """

    #Create random graph:
    RandomGraph = nx.gnp_random_graph(n, p, seed=None, directed=False)

    #Isolate the largest component
    #ConnectedComponents = sorted(nx.connected_components(RandomGraph),key=len,reverse=True)
    #RandomGraph = G.subgraph(Gcc[0])
    LargestComponent = max(nx.connected_components(RandomGraph), key=len)

    Nodes = set(RandomGraph.nodes())

    Difference = Nodes - LargestComponent

    #print("Largest comp",LargestComponent)
    #print("Nodes",Nodes)

    for i in Difference:
        RandomGraph.remove_node(i)

    Nodes = RandomGraph.nodes()

    NodeNum = RandomGraph.number_of_nodes()

    positions = nx.spring_layout(RandomGraph)#[]
    CompleteGraph = 0
    MList = []
    SepList = []

    #Datalist: we record:
    #Distance from origin node
    #Degree of node
    #List of the number of times it is invaded by M.
    DataList = []


    #Set patches and infections
    NodesPlaced = 0
    """
    MAKE A LIST THAT KEEPS TRACK OF INACTIVE PATCHES - DELETE AS INVASIONS HAPPEN, SIM ENDS WHEN THIS LIST EMPTY
    CREATE NEW ASPECT FOR PATCHES THAT INDICATE IF ACTIVE OR NOT
    CHANGE UPDATE DYNAMICS TO ACCOUNT FOR INACTIVE PATCHES HAVING NO MEMBERS
    CHANGE UPDATE DYNAMICS SO INACTIVE PATCHES CAN BE INVADED BY M
    """
    InactivePatchIDs = []

    MNum = 0

    #Number of patches
    PNum = np.round(P * NodeNum).astype(int)

    for i in Nodes:#RandomGraph.nodes(data=True):
        """
        Dist = nx.shortest_path_length(RandomGraph,source=0,target=i)
        Degree = RandomGraph.degree[i]
        TimeInvadedList = []
        DataList.append([Dist,Degree,TimeInvadedList])
        """
        if NodesPlaced < PNum:
            RandomGraph.nodes(data=True)[i]["patch"] = 1
            RandomGraph.nodes(data=True)[i]["infection"] = "M"
            RandomGraph.nodes(data=True)[i]["label"] = "Z"
            RandomGraph.nodes(data=True)[i]["active"] = True
            MNum += 1

            #Make it an inactive site
            if NodesPlaced > 0:
                RandomGraph.nodes(data=True)[i]["active"] = False
                RandomGraph.nodes(data=True)[i]["infection"] = "WT"
                InactivePatchIDs.append(i)
                MNum -= 1
            """
            CompleteGraph.nodes(data=True)[i]["patch"] = 1
            CompleteGraph.nodes(data=True)[i]["infection"] = "M"
            CompleteGraph.nodes(data=True)[i]["label"] = "X"
            """
            MList.append(i)
            SepList.append(0)

            NodesPlaced += 1
            #GeoGraph.nodes(data=True)[i]["patch"] = 1
            #GeoGraph.nodes(data=True)[i]["infection"] = "M"
            #GeoGraph.nodes(data=True)[i]["label"] = "X"


        else:
            RandomGraph.nodes(data=True)[i]["patch"] = 0
            RandomGraph.nodes(data=True)[i]["infection"] = "WT"
            RandomGraph.nodes(data=True)[i]["label"] = ""
            """
            CompleteGraph.nodes(data=True)[i]["patch"] = 0
            CompleteGraph.nodes(data=True)[i]["infection"] = "WT"
            CompleteGraph.nodes(data=True)[i]["label"] = ""
            """
            #GeoGraph.nodes(data=True)[i]["patch"] = 0
            #GeoGraph.nodes(data=True)[i]["infection"] = "WT"
            #GeoGraph.nodes(data=True)[i]["label"] = ""

    #print(InactivePatchIDs)
    #sys.exit()

    InitDict = {
            "positions":positions,
            "CompleteGraph":CompleteGraph,
            "RandomGraph":RandomGraph,
            "InactivePatchIDs":InactivePatchIDs,
            "MNum":MNum,
            "NodeNum":NodeNum,
            "PNum":PNum}

    return InitDict#positions, CompleteGraph, RandomGraph, InactivePatchIDs


def Init(GraphDict):
    n = GraphDict["N"]
    P = GraphDict["P"]
    Type = GraphDict["Type"]
    SingleActive = GraphDict["SingleActive"]

    LargestComponentBool = True
    if "LargestComponent" in GraphDict:
        LargestComponentBool = GraphDict["LargestComponent"]


    #Create the graphs themselves
    if Type == "ER":
        p = GraphDict["p"]
        Graph = nx.gnp_random_graph(n, p, seed=None, directed=False)

    elif Type == "SmallWorld":
        k = GraphDict["k"]
        r = GraphDict["r"]
        t = GraphDict["t"]
        Graph = nx.connected_watts_strogatz_graph(n,k,r,t,seed=None)

    elif Type == "Geometric":
        radius = GraphDict["radius"]
        Graph = nx.random_geometric_graph(n,radius)

    elif Type == "Geometric_Torus":
        radius = GraphDict["radius"]
        Graph = nx.random_geometric_graph(n,radius)

        for node1 in Graph.nodes():
            n1x,n1y = Graph.nodes[node1]["pos"]
            #print(node1,n1x,n1y)
            
            for node2 in range(node1+1,len(Graph.nodes())):
                n2x,n2y = Graph.nodes[node2]["pos"]
                #print(node1,node2)

                edge_added = False
                for x in (-1,0,1):
                    for y in (-1,0,1):
                        dist = np.sqrt((n1x-n2x+x)**2 + (n1y-n2y+y)**2)
                        if dist <= radius:
                            Graph.add_edge(node1,node2)
                            edge_added = True
                            #print("Added:",(node1,node2))
                            #print((n1x,n1y),(n2x,n2y))
                            break
                    if edge_added:
                        break
    else:
        raise Exception("Graph Improperly Defined")


    
    ######################
    #Isolate the largest component of the Graph
    if LargestComponentBool:
        print("Isolate largest component only")
        LargestComponent = max(nx.connected_components(Graph), key=len)

        Nodes = set(Graph.nodes())

        Difference = Nodes - LargestComponent


        for i in Difference:
            Graph.remove_node(i)
    #####################
    

    Nodes = Graph.nodes()

    NodeNum = Graph.number_of_nodes()

    #positions = nx.spring_layout(Graph)#[]

    if Type == "SmallWorld":
        positions = nx.circular_layout(Graph)

    elif Type == "Geometric" or Type == "Geometric_Torus":
        positions = {node: data["pos"] for node, data in Graph.nodes(data=True)}
        #positions = nx.kamada_kawai_layout(Graph)

    else:
        positions = nx.spring_layout(Graph)

    CompleteGraph = 0
    MList = []
    SepList = []

    #Set patches and infections
    NodesPlaced = 0
    InactivePatchIDs = []

    MNum = 0

    #Number of patches
    PNum = np.round(P * NodeNum).astype(int)

    #Choose the IDs of the zealots
    ZList = random.sample(range(0,NodeNum),PNum)

    for i in Nodes:#RandomGraph.nodes(data=True):
        if i in ZList:
            Graph.nodes(data=True)[i]["patch"] = 1
            Graph.nodes(data=True)[i]["infection"] = "M"
            Graph.nodes(data=True)[i]["label"] = ""#"Z"
            Graph.nodes(data=True)[i]["active"] = True
            MNum += 1

            #Make it an inactive site
            if (NodesPlaced > 0) and SingleActive:
                Graph.nodes(data=True)[i]["active"] = False
                Graph.nodes(data=True)[i]["infection"] = "WT"
                InactivePatchIDs.append(i)
                MNum -= 1
            MList.append(i)
            SepList.append(0)

            NodesPlaced += 1

        else:
            Graph.nodes(data=True)[i]["patch"] = 0
            Graph.nodes(data=True)[i]["infection"] = "WT"
            Graph.nodes(data=True)[i]["label"] = ""


    
    #Set parameter 'changeable' in the next timestep, and make list of these
    ChangeableList = []
    
    for i in Nodes:
        #print("i:",i)
        if Graph.nodes(data=True)[i]['patch'] != 1:
            #print("Not patch")
            Graph.nodes(data=True)[i]["changeable"] = False
            for j in list(Graph.neighbors(i)):
                #print("Neighbour:",j)
                if Graph.nodes(data=True)[i]['infection'] != Graph.nodes(data=True)[j]['infection']:
                    Graph.nodes(data=True)[i]["changeable"] == True
                    #print("NOT SAME AS NEIGHBOUR")
                    ChangeableList.append(i)
                    break
            #if Graph.nodes(data=True)[i]["changeable"] == True:
            #    ChangeableList.append(i)
            #    print("ADDED NODE",i)

        
    print("Changeable list",ChangeableList)
    
    InitDict = {
            "positions":positions,
            "CompleteGraph":CompleteGraph,
            "Graph":Graph,
            "InactivePatchIDs":InactivePatchIDs,
            "MNum":MNum,
            "NodeNum":NodeNum,
            "PNum":PNum,
            "ChangeableList":ChangeableList}

    return InitDict#positions, CompleteGraph, RandomGraph, InactivePatchIDs







def Iterate(ParamDict):#t,Graph,F,InactivePatchIDs):
    """Iterate a single infection probability event

    """

    t = ParamDict["t"]
    Graph = ParamDict["Graph"]
    F = ParamDict["F"]
    InactivePatchIDs = ParamDict["InactivePatchIDs"]
    GraphMNum = ParamDict["MNum"]
    ChangeableList = ParamDict["ChangeableList"]

    #Create Graph node list now from profiling
    Nodes = Graph.nodes()

    NodeKeyList = list(Nodes.keys())

    #print("NODES",Nodes)
    #sys.exit()

    InactivePatchActivated = False

    #Choose a random index
    #print(ChangeableList)
    randindex = random.choice(NodeKeyList)#random.randint(0,len(Nodes)-1)
    randnode = Nodes[randindex]#random.choice(Graph.nodes())

    
    while randnode["patch"] and randnode["active"]:
        randindex =  random.choice(NodeKeyList)#random.randint(0,len(Nodes)-1)
        randnode = Nodes[randindex]
    
    if randnode in ChangeableList:
        MNum = 0
        Num = 0

        #Count yourself first
        InitialInfection = "WT"

        if randnode["infection"] == "M":
            #MNum += 1
            InitialInfection = "M"

        #Num += 1

        if (randnode["patch"]) and (not randnode["active"]):
            Num -= 1


        #Count your neighbours
        for j in list(Graph.neighbors(randindex)):
            Num += 1

            if Nodes[j]["infection"] == "M":
                MNum += 1

            if Nodes[j]["patch"] and (not Nodes[j]["active"]):
                Num -= 1


        if Num == 0:
            Num = 1
        #Adjust self
        FinalInfection = "M"
        if random.uniform(0,1) < MNum*F/((Num-MNum) + MNum*F):
            Nodes(data=True)[randindex]["infection"] = "M"

            """
            if randindex not in MList:
                MList.append(randindex)
                Sep = nx.shortest_path_length(Graph,source=0,target=randindex)
                SepDist.append(Sep)

                DataList[randindex][2].append(t)
            """
            if randnode["patch"]:
                randnode["active"] = 1
                InactivePatchIDs.remove(randindex)
                InactivePatchActivated = True
                #SepDist = MeasureSepDist(Graph,MList)

        else:
            Nodes(data=True)[randindex]["infection"] = "WT"
            FinalInfection = "WT"
            """
            if randindex in MList:
                index = MList.index(randindex)
                #MList.remove(randindex)
                del MList[index]
                del SepDist[index]
                #SepDist = MeasureSepDist(Graph,MList)
            """


        if InitialInfection != FinalInfection:
            if FinalInfection == "M":
                GraphMNum += 1
            else:
                GraphMNum -= 1

            
            #As there is a difference, make see if node and its neighbours are changeable

            #Self
            Nodes(data=True)[randindex]["changeable"] = False
            for j in list(Graph.neighbors(randindex)):
                if Nodes(data=True)[randindex]['infection'] != Nodes(data=True)[j]['infection']:
                    Nodes(data=True)[randindex]["changeable"] = True
                    break
            
            if (Nodes(data=True)[randindex]["changeable"] == False) and (randindex in ChangeableList):
                ChangeableList.remove(randindex)

            elif (Nodes(data=True)[randindex]["changeable"] == True) and (randindex not in ChangeableList):
                ChangeableList.append(randindex)

        
            #Neighbours
            for j in list(Graph.neighbors(randindex)):
                if Nodes(data=True)[j]['patch'] != 1:
                    Nodes(data=True)[j]["changeable"] = False
                    for k in list(Graph.neighbors(j)):
                        if Nodes(data=True)[j]['infection'] != Nodes(data=True)[k]['infection']:
                            Graph.nodes(data=True)[j]["changeable"] = True
                            break

                    if (Nodes(data=True)[j]["changeable"] == False) and (j in ChangeableList):
                        ChangeableList.remove(j)

                    elif (Nodes(data=True)[j]["changeable"] == True) and (j not in ChangeableList):
                        ChangeableList.append(j)
            
    """
    ResultsDict = {
            "Graph": Graph,
            "InactivePatchIDs": InactivePatchIDs,
            "InactivePatchActivated": InactivePatchActivated,
            "MNum": GraphMNum}
    """

    ParamDict["MNum"] = GraphMNum
    ParamDict["InactivePatchActivated"] = InactivePatchActivated

    ParamDict["ChangeableList"] = ChangeableList

    return ParamDict#Graph, InactivePatchIDs,InactivePatchActivated



def Iterate_2(ParamDict):
    """Iterate a single infection probability event

    """


    #################################################
    #Unpackage
    t = ParamDict["t"]
    GraphList = ParamDict["GraphList"]
    GraphNeighbourList = ParamDict["GraphNeighbourList"]
    F = ParamDict["F"]
    WEAKNUM = ParamDict["WEAKNUM"]
    ChangeableList = ParamDict["ChangeableList"]
    N = ParamDict["N"]
    
    ##################################################
    ##Process
    randnode = floor(random.random()*N)#random.randint(0, N-1)
    while GraphList[randnode] == 2:
        randnode = floor(random.random()*N)

    if randnode in ChangeableList:
        initialopinion = GraphList[randnode]

        #Iterate through neighbours, count number of weak:
        strongnum = 0
        for i in GraphNeighbourList[randnode]:
            if GraphList[i] == 0:
                strongnum += 1

        #Evaluate prob of adopting strong position
        probstrong = strongnum / (strongnum + F*(len(GraphNeighbourList[randnode])-strongnum))

        if random.uniform(0,1) < probstrong:
            GraphList[randnode] = 0

        else:
            GraphList[randnode] = 1


        if initialopinion != GraphList[randnode]:
            #Update WEAKNUM:
            if GraphList[randnode] == 0:
                WEAKNUM -= 1
            else:
                WEAKNUM += 1


            #Search through the neighbourlists to adjust changeablelists
            checklist = copy.copy(GraphNeighbourList[randnode])
            checklist.add(randnode)


            for i in checklist:
                inlist = False
                for j in GraphNeighbourList[i]:
                    if min(1,GraphList[i]) != min(1,GraphList[j]):
                        inlist = True
                        if GraphList[i] != 2 and (i not in ChangeableList):
                            ChangeableList.add(i)

                        if GraphList[j] != 2 and (j not in ChangeableList):
                            ChangeableList.add(j)

                        continue

                if not inlist:
                    if GraphList[i] != 2:
                        ChangeableList.remove(i)
                    continue



    #If changeablelist is empty, convert a random one
    if len(ChangeableList) == 0 and C>1/(1-z):
        randnode = floor(random.random()*N)#random.randint(0, N-1)
        while GraphList[randnode] != 1:
            randnode = floor(random.random()*N)#random.randint(0,N-1)


        GraphList[randnode] = 0
        WEAKNUM -= 1

        #Update changeable list
        for i in GraphNeighbourList[randnode]:
            if GraphList[i] != GraphList[randnode]:
                if GraphList[i] != 2 and (i not in ChangeableList):
                    ChangeableList.add(i)

                if GraphList[randnode] != 2 and (randnode not in ChangeableList):
                    ChangeableList.add(randnode)


    #################################################
    #Repackage
    ParamDict["WEAKNUM"] = WEAKNUM

    ParamDict["ChangeableList"] = ChangeableList

    return ParamDict


def Escape_Absorbing(ParamDict):
    """
    Choose a single random node and set it to be WT if the ChangeableList is empty (meaning the absorbing state is reached)
    """

    t = ParamDict["t"]
    Graph = ParamDict["Graph"]
    F = ParamDict["F"]
    InactivePatchIDs = ParamDict["InactivePatchIDs"]
    GraphMNum = ParamDict["MNum"]
    ChangeableList = ParamDict["ChangeableList"]

    #Create Graph node list now from profiling
    Nodes = Graph.nodes()

    NodeKeyList = list(Nodes.keys())

    #print("NODES",Nodes)
    #sys.exit()

    InactivePatchActivated = False

    #Choose a random index
    randindex = random.choice(NodeKeyList)#random.randint(0,len(Nodes)-1)
    randnode = Nodes[randindex]#random.choice(Graph.nodes())

    
    while (randnode["patch"] and randnode["active"]) or (randnode["infection"] == 'WT'):
        randindex =  random.choice(NodeKeyList)#random.randint(0,len(Nodes)-1)
        randnode = Nodes[randindex]
    

    #Set the node to be WT
    Nodes(data=True)[randindex]["infection"] = "WT"

    GraphMNum -= 1

    
    #As there is a difference, make see if node and its neighbours are changeable
    #Self
    Nodes(data=True)[randindex]["changeable"] = False
    for j in list(Graph.neighbors(randindex)):
        if Nodes(data=True)[randindex]['infection'] != Nodes(data=True)[j]['infection']:
            Nodes(data=True)[randindex]["changeable"] = True
            break

    if (Nodes(data=True)[randindex]["changeable"] == False) and (randindex in ChangeableList):
        ChangeableList.remove(randindex)

    elif (Nodes(data=True)[randindex]["changeable"] == True) and (randindex not in ChangeableList):
        ChangeableList.append(randindex)


    #Neighbours
    for j in list(Graph.neighbors(randindex)):
        if Nodes(data=True)[j]['patch'] != 1:
            Nodes(data=True)[j]["changeable"] = False
            for k in list(Graph.neighbors(j)):
                if Nodes(data=True)[j]['infection'] != Nodes(data=True)[k]['infection']:
                    Graph.nodes(data=True)[j]["changeable"] = True
                    break

            if (Nodes(data=True)[j]["changeable"] == False) and (j in ChangeableList):
                ChangeableList.remove(j)

            elif (Nodes(data=True)[j]["changeable"] == True) and (j not in ChangeableList):
                ChangeableList.append(j)

    
    #Package and return
    ParamDict["MNum"] = GraphMNum
    ParamDict["InactivePatchActivated"] = InactivePatchActivated
    ParamDict["ChangeableList"] = ChangeableList

    return ParamDict

def Observe(ObserveDict):#(t,Graph,positions,SaveDirName):
    """Save an image of the graphs

    """
    Graph = ObserveDict["Graph"]
    t = ObserveDict["t"]
    positions = ObserveDict["positions"]
    SaveDirName = ObserveDict["SaveDirName"]


    labels = nx.get_node_attributes(Graph, 'label')

    """
    print("Labels:",labels)

    print("Graph Nodes:",Graph.nodes(data=True))

    print("Positons:",positions)


    for i in Graph.nodes(data=True):
        if i[1]["infection"] == "WT":
            print("Blue")

        else:
            print("Red")


    print("Finished List")
    """
    colorlist = []
    for i in Graph.nodes(data=True):
        color = None
        if i[1]["infection"]=="WT":
            color = "#94d2e5"
        else:
            color = "#cc657f"

        if i[1]["patch"]  and not i[1]["active"]:
            color = "#CCCCCC"
       
        if i[1]["patch"] and i[1]["active"]:
            color = "#99324c"

        colorlist.append(color)


    nx.draw(Graph,
            positions,
            labels=labels,
            node_color=colorlist,
            node_size=50#100#['blue' if i[1]["infection"]=="WT" else 'red' for  i in Graph.nodes(data=True)]
            )

    print("MADE")
    savefig(SaveDirName + "/Snapshot_t_" + str(t).zfill(5))
    close()



"""
def Observe(t,CompleteGraph,RandomGraph,GeometricGraph,positions):

    labels = nx.get_node_attributes(CompleteGraph, 'label')

    #Draw Complete
    nx.draw(CompleteGraph,      #Graph
        positions,  #Defined positions
        labels = labels,
        node_color=['blue' if i[1]["infection"]=="WT" else 'red' for  i in CompleteGraph.nodes(data=True)]
        )
    savefig(SaveDirName + "/Complete_%d.png"%(t))
    close()


    #Draw Complete
    nx.draw(RandomGraph,      #Graph
        positions,  #Defined positions
        labels = labels,
        node_color=['blue' if i[1]["infection"]=="WT" else 'red' for  i in RandomGraph.nodes(data=True)]
        )
    savefig(SaveDirName + "/Random_%d.png"%(t))
    close()


    #Draw Geometric
    nx.draw(GeometricGraph,      #Graph
        positions,  #Defined positions
        labels = labels,
        node_color=['blue' if i[1]["infection"]=="WT" else 'red' for  i in GeometricGraph.nodes(data=True)]
        )
    savefig(SaveDirName + "/Geometric_%d.png"%(t))
    close()
"""


def MeasureMutants(Graph):
    """Count number of mutants in Graph
    """
    MNum = 0
    Num = 0

    for i in Graph.nodes(data=True):
        if i[1]["infection"] == "M":
            MNum += 1

        Num += 1

    return MNum


def MeasureSepDist(Graph,MList):
    """Generate a list of Mutant distances from initial node.
    """
    SepList = []

    for i in MList:
        if i!= 0:
            Sep = nx.shortest_path_length(Graph,source=0,target=i)
            SepList.append(Sep)

    return SepList




def Plot(PlotDict):

    plt.figure()
    plt.plot(PlotDict["xlist"],PlotDict["ylist"])
    plt.savefig(PlotDict["SaveDirName"] + PlotDict["FigName"])
    plt.close()

def GraphStats(Graph):
    import collections

    ###############################################
    #Find the Graph size
    GraphSize = Graph.number_of_nodes()
    ################################################

    ################################################
    #Find the mean Graph Clustering Coefficient
    MeanClusterCoeff = nx.average_clustering(Graph)
    ################################################

    ################################################
    #Find the Graph Degree Distribution
    degree_sequence = sorted([d for n, d in Graph.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, deg_cnt = zip(*degreeCount.items())
    ################################################

    deg = np.asarray(deg)
    deg_cnt = np.asarray(deg_cnt)

    ################################################
    #Use above to find Mean Graph Degree Distribution
    MeanDegree = np.sum(deg*deg_cnt)/sum(deg_cnt)
    ################################################


    ################################################
    #Cluster dist
    ComponentDist = [len(c) for c in sorted(nx.connected_components(Graph),key=len)]
    ################################################

    StatDict = {
            "GraphSize": GraphSize,
            "MeanClusterCoeff": MeanClusterCoeff,
            "deg_list": np.asarray(deg),
            "deg_cnt_list": np.asarray(deg_cnt),
            "MeanDegree": MeanDegree,
            "ComponentDist": ComponentDist
            }

    return StatDict
















def CompleteDist(N,F,Z):
    z = Z/N

    zs = 1/N

    IGNORESTATIONARY = False

    def P_Up(n,z,F):
        return (1-n-z-zs)/(1-z-zs) * F*(n+z) / (1-n-z+F*(n+z))

    def P_Down(n,z,F):
        return n/(1-z-zs) * (1-n-z) / (1-n-z+F*(n+z))



    def alpha(n,z,F):
        #return (1-z-zs-n)/(1-z-zs) * (F*(n+z)-n)/ (1-z-n+F*(n+z))
        return P_Up(n,z,F) - P_Down(n,z,F)

    def beta(n,z,F):
        #return (1/N)*(1-z-zs-n)/(1-z) * (F*(n+z)+n)/ (1-z-n+F*(n+z))
        return (P_Up(n,z,F) + P_Down(n,z,F))/N


    epsilon = zs
    nlist = np.linspace(0,1-z -epsilon,N)
    PList = []



    for n in nlist:

        integral = integrate.quad(lambda n: alpha(n,z,F)/beta(n,z,F),0,n)[0]

        if not IGNORESTATIONARY:
            print(2*integral - np.log(beta(n,z,F)))
            P = np.exp(2*integral - np.log(beta(n,z,F)))

        else:
            P = np.exp(2 * integral )

        PList.append( P)


    PList = np.asarray(PList)
    Normalisation = integrate.simpson(PList,nlist)
    PList = PList/Normalisation


    return (nlist,PList)

