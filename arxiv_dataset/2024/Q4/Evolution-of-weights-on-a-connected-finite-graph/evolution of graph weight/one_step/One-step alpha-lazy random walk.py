
import numpy as np
import math
import time
from sklearn import preprocessing, metrics
from multiprocessing import Pool, cpu_count
import community.community_louvain as community_louvain
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
import importlib
import cvxpy as cvx
import matplotlib.pyplot as plt
import networkx as nx
import heapq

_nbr_topk = 3000
_apsp = {}
_max_ari={}
_max_nmi={}
_max_modularity={}



#定义alpha-RhoNormalize RicciCurvature类，包括曲率的计算和flow的计算
class RhoNormalizeCurvature:
    
    def __init__(self, G, alpha=0.5, weight="weight", proc=cpu_count()):
        self.G = G.copy()
        self.alpha = alpha
        self.weight = weight
        self.proc = proc
        self.lengths = {}  # all pair shortest path dictionary
        self.densities = {}  # density distribution dictionary
        self.EPSILON = 1e-7  # to prevent divided by zero
        self.base = math.e
        self.exp_power = 2
        self.label_dict = {}
        

   

    def _get_all_pairs_shortest_path(self):
        # Construct the all pair shortest path lookup
        lengths = dict(nx.all_pairs_dijkstra_path_length(self.G, weight=self.weight))
        return lengths

    def _get_single_node_neighbors_distributions(self, node):
        neighbors = list(self.G.neighbors(node))
        heap_weight_node_pair = []
        for nbr in neighbors:
            w=self.G[node][nbr][self.weight]
            if len(heap_weight_node_pair) < _nbr_topk:
                heapq.heappush(heap_weight_node_pair, (w, nbr))
            else:
                heapq.heappushpop(heap_weight_node_pair, (w, nbr))
        nbr_edge_weight_sum = sum([x[0] for x in heap_weight_node_pair])
        if not neighbors:
            return [1],[node]
        if nbr_edge_weight_sum > self.EPSILON:
            distributions = [(1.0 - self.alpha) * w / nbr_edge_weight_sum for w, _ in heap_weight_node_pair]

        else:
            # Sum too small, just evenly distribute to every neighbors
            
            distributions = [(1.0 - self.alpha) / len(heap_weight_node_pair)] * len(heap_weight_node_pair)
        nbr = [x[1] for x in heap_weight_node_pair]
        return distributions + [self.alpha], nbr + [node]
    def _get_edge_density_distributions(self):
        densities = dict()

        
        for x in self.G.nodes():
            densities[x] = self._get_single_node_neighbors_distributions(x)

    def _optimal_transportation_distance(self, x, y, d):
        rho = cvx.Variable((len(y), len(x)))  

        # objective function d(x,y) * rho * x, need to do element-wise multiply here
        obj = cvx.Minimize(cvx.sum(cvx.multiply(np.multiply(d.T, x.T), rho)))

        # \sigma_i rho_{ij}=[1,1,...,1]
        source_sum = cvx.sum(rho, axis=0, keepdims=True)
        constrains = [rho * x == y, source_sum == np.ones((1, (len(x)))), 0 <= rho, rho <= 1]
        prob = cvx.Problem(obj, constrains)

        m = prob.solve(solver="ECOS")  # change solver here if you want
        return m

    def _distribute_densities(self, source, target):
        # Append source and target node into weight distribution matrix x,y
        x,source_topknbr=self._get_single_node_neighbors_distributions(source)
        y,target_topknbr=self._get_single_node_neighbors_distributions(target)
        d = []
        for src in source_topknbr:
            tmp = []
            for tgt in target_topknbr:
                tmp.append(self.lengths[src][tgt])
            d.append(tmp)
        d = np.array(d)
        x=np.array(x)
        y=np.array(y)
        return x,y,d    

    def _compute_ricci_curvature_single_edge(self, source, target):

        assert source != target, "Self loop is not allowed."  # to prevent self loop

        # If the weight of edge is too small, return 0 instead.
        if self.lengths[source][target] < self.EPSILON:
            print("Zero weight edge detected for edge (%s,%s), return Ricci Curvature as 0 instead." % (source, target))
            return {(source, target): 0}

        # compute transportation distance
        m = 1  # assign an initial cost

        x, y, d = self._distribute_densities(source, target)
        m = self._optimal_transportation_distance(x, y, d)

        # compute Ricci curvature: k=1-(m_{x,y})/d(x,y)
        result = 1-m / self.lengths[source][target]  # Divided by the length of d(i, j)
        #print("Ricci curvature (%s,%s) = %f" % (source, target, result))

        return {(source, target): result}

    def _wrap_compute_single_edge(self, stuff):
        return self._compute_ricci_curvature_single_edge(*stuff)

    def compute_ricci_curvature_edges(self, edge_list=None):
        
        if not edge_list:
            edge_list = []

        # Construct the all pair shortest path dictionary
        #if not self.lengths:
        self.lengths = self._get_all_pairs_shortest_path()

        # Construct the density distribution
        if not self.densities:
            self.densities = self._get_edge_density_distributions()
        
       
            
        # Compute Ricci curvature for edges
        args = [(source, target) for source, target in edge_list]

        result = [self._wrap_compute_single_edge(arg) for arg in args]
            
      

        return result

    def compute_ricci_curvature(self):
        
        if not nx.get_edge_attributes(self.G, self.weight):
            print('Edge weight not detected in graph, use "weight" as edge weight.')
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][self.weight] = 1.0

                
        edge_ricci = self.compute_ricci_curvature_edges(self.G.edges())
        
        # Assign edge Ricci curvature from result to graph G
        for rc in edge_ricci:
            for k in list(rc.keys()):
                source, target = k
                self.G[source][target]['ricciCurvature'] = rc[k]

        # Compute node Ricci curvature
        for n in self.G.nodes():
            rc_sum = 0  # sum of the neighbor Ricci curvature
            if self.G.degree(n) != 0:
                for nbr in self.G.neighbors(n):
                    if 'ricciCurvature' in self.G[n][nbr]:
                        rc_sum += self.G[n][nbr]['ricciCurvature']

                # Assign the node Ricci curvature to be the average of node's adjacency edges
                self.G.nodes[n]['ricciCurvature'] = rc_sum / self.G.degree(n)


    def compute_ricci_flow(self, iterations=5, step=0.01, delta=1e-6):
        if not nx.is_connected(self.G):
            print("Not connected graph detected, compute on the largest connected component instead.")
            self.G = nx.Graph(max([self.G.subgraph(c) for c in nx.connected_components(self.G)], key=len))
            print('---------------------------')
            print(self.G)

        self.G.remove_edges_from(nx.selfloop_edges(self.G))
        

        
        # Start compute edge Ricci flow
        t0 = time.time()

        if nx.get_edge_attributes(self.G, "original_RC"):
            print("original_RC detected, continue to refine the ricci flow.")
        else:
            self.compute_ricci_curvature()

            for (v1, v2) in self.G.edges():
                self.G[v1][v2]["original_RC"] = self.G[v1][v2]["ricciCurvature"]

        # Start the Ricci flow process
        self.rc_diff = []
        for i in range(iterations):
            
            w = nx.get_edge_attributes(self.G, self.weight)
            
            sum_K_R = sum(self.G[v1][v2]["ricciCurvature"] * self.lengths[v1][v2] for (v1, v2) in self.G.edges())
            sumr=0
            for (v1, v2) in self.G.edges():
                sumr+=self.G[v1][v2][self.weight]
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][self.weight] -= step * (self.G[v1][v2]["ricciCurvature"]) * (self.lengths[v1][v2])
                self.G[v1][v2][self.weight] += step*(sum_K_R)/sumr  
           
            #Merge really adjacent node
            G1 = self.G.copy()
            merged = True
            while merged:
                merged = False
                for v1,v2 in G1.edges():
                    if G1[v1][v2][self.weight] < delta * 10:
                        # if self.find(v1)!= self.find(v2):
                        #     self.union(v1, v2)
                        G1 = nx.contracted_edge(G1, (v1, v2), self_loops=False)
                        merged = True
                        break
            self.G = G1

            self.compute_ricci_curvature()
           # print("=== Ricciflow iteration % d ===" % int(i+1))
           # nx.write_gexf(self.G, rf"C:\Users\20978\Desktop\evolution of graph weight\one_step\a quasi-normalized version\data\iteration\karate\iteration_{i + 1}.gexf")
            rc = nx.get_edge_attributes(self.G, "ricciCurvature")
            if rc:
                diff = max(rc.values()) - min(rc.values())
                print("Ricci curvature difference: %f" % diff)
                print("max:%f, min:%f | maxw:%f, minw:%f" % (max(rc.values()), min(rc.values()), max(w.values()), min(w.values())))
            else:
                diff = 0

           

            if diff < delta:
                print("Ricci curvature converged, process terminated.")
                break

            
                
            # clear the APSP and densities since the graph have changed.
            self.densities = {}

      #  nx.write_gexf(self.G, os.path.join(save_gexf_dir, "%d.gexf"%iterations))
        #show_results(self.G)
        print("\n%8f secs for Ricci flow computation." % (time.time() - t0))


#定义alpha-Rho RicciCurvature类，包括曲率的计算和flow的计算
class RhoCurvature:
    
    def __init__(self, G, alpha=0.5, weight="weight", proc=cpu_count()):
        self.G = G.copy()
        self.alpha = alpha
        self.weight = weight
        self.proc = proc
        self.lengths = {}  # all pair shortest path dictionary
        self.densities = {}  # density distribution dictionary
        self.EPSILON = 1e-7  # to prevent divided by zero
        self.base = math.e
        self.exp_power = 2
        self.label_dict = {}
        self.rank = {}  # 用于 union-find 的 rank 字典
        for i, node in enumerate(self.G.nodes()):
            self.label_dict[node] = i
            self.rank[i] = 0  # 初始化 rank 为 0

   


    def _get_all_pairs_shortest_path(self):
        # Construct the all pair shortest path lookup
        lengths = dict(nx.all_pairs_dijkstra_path_length(self.G, weight=self.weight))
        return lengths
    def _get_single_node_neighbors_distributions(self, node):
        neighbors = list(self.G.neighbors(node))
        heap_weight_node_pair = []
        for nbr in neighbors:
            w=self.G[node][nbr][self.weight]
            if len(heap_weight_node_pair) < _nbr_topk:
                heapq.heappush(heap_weight_node_pair, (w, nbr))
            else:
                heapq.heappushpop(heap_weight_node_pair, (w, nbr))
        nbr_edge_weight_sum = sum([x[0] for x in heap_weight_node_pair])
        if not neighbors:
            return [1],[node]
        if nbr_edge_weight_sum > self.EPSILON:
            distributions = [(1.0 - self.alpha) * w / nbr_edge_weight_sum for w, _ in heap_weight_node_pair]

        else:
            # Sum too small, just evenly distribute to every neighbors
           
            distributions = [(1.0 - self.alpha) / len(heap_weight_node_pair)] * len(heap_weight_node_pair)
        nbr = [x[1] for x in heap_weight_node_pair]
        return distributions + [self.alpha], nbr + [node]
    def _get_edge_density_distributions(self):
        densities = dict()

        
        for x in self.G.nodes():
            densities[x] = self._get_single_node_neighbors_distributions(x)

    def _optimal_transportation_distance(self, x, y, d):
        rho = cvx.Variable((len(y), len(x)))  

        # objective function d(x,y) * rho * x, need to do element-wise multiply here
        obj = cvx.Minimize(cvx.sum(cvx.multiply(np.multiply(d.T, x.T), rho)))

        # \sigma_i rho_{ij}=[1,1,...,1]
        source_sum = cvx.sum(rho, axis=0, keepdims=True)
        constrains = [rho * x == y, source_sum == np.ones((1, (len(x)))), 0 <= rho, rho <= 1]
        prob = cvx.Problem(obj, constrains)

        m = prob.solve(solver="ECOS")  # change solver here if you want
        return m

    def _distribute_densities(self, source, target):
        # Append source and target node into weight distribution matrix x,y
        x,source_topknbr=self._get_single_node_neighbors_distributions(source)
        y,target_topknbr=self._get_single_node_neighbors_distributions(target)
        d = []
        for src in source_topknbr:
            tmp = []
            for tgt in target_topknbr:
                tmp.append(self.lengths[src][tgt])
            d.append(tmp)
        d = np.array(d)
        x=np.array(x)
        y=np.array(y)
        return x,y,d    
        

    def _compute_ricci_curvature_single_edge(self, source, target):

        assert source != target, "Self loop is not allowed."  # to prevent self loop

        # If the weight of edge is too small, return 0 instead.
        if self.lengths[source][target] < self.EPSILON:
            print("Zero weight edge detected for edge (%s,%s), return Ricci Curvature as 0 instead." % (source, target))
            return {(source, target): 0}

        # compute transportation distance
        m = 1  # assign an initial cost

        x, y, d = self._distribute_densities(source, target)
        m = self._optimal_transportation_distance(x, y, d)

        # compute Ricci curvature: k=1-(m_{x,y})/d(x,y)
        result =1- m / self.lengths[source][target]  # Divided by the length of d(i, j)
        #print("Ricci curvature (%s,%s) = %f" % (source, target, result))

        return {(source, target): result}

    def _wrap_compute_single_edge(self, stuff):
        return self._compute_ricci_curvature_single_edge(*stuff)

    def compute_ricci_curvature_edges(self, edge_list=None):
        
        if not edge_list:
            edge_list = []

        # Construct the all pair shortest path dictionary
        #if not self.lengths:
        self.lengths = self._get_all_pairs_shortest_path()

        # Construct the density distribution
        if not self.densities:
            self.densities = self._get_edge_density_distributions()
       
            
        # Compute Ricci curvature for edges
        args = [(source, target) for source, target in edge_list]

        result = [self._wrap_compute_single_edge(arg) for arg in args]
            
        return result

    def compute_ricci_curvature(self):
        
        if not nx.get_edge_attributes(self.G, self.weight):
            print('Edge weight not detected in graph, use "weight" as edge weight.')
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][self.weight] = 1.0

                
        edge_ricci = self.compute_ricci_curvature_edges(self.G.edges())
        
        # Assign edge Ricci curvature from result to graph G
        for rc in edge_ricci:
            for k in list(rc.keys()):
                source, target = k
                self.G[source][target]['ricciCurvature'] = rc[k]

        # Compute node Ricci curvature
        for n in self.G.nodes():
            rc_sum = 0  # sum of the neighbor Ricci curvature
            if self.G.degree(n) != 0:
                for nbr in self.G.neighbors(n):
                    if 'ricciCurvature' in self.G[n][nbr]:
                        rc_sum += self.G[n][nbr]['ricciCurvature']

                # Assign the node Ricci curvature to be the average of node's adjacency edges
                self.G.nodes[n]['ricciCurvature'] = rc_sum / self.G.degree(n)


    def compute_ricci_flow(self, iterations=5, step=0.01, delta=1e-6):
        if not nx.is_connected(self.G):
            print("Not connected graph detected, compute on the largest connected component instead.")
            self.G = nx.Graph(max([self.G.subgraph(c) for c in nx.connected_components(self.G)], key=len))
            print('---------------------------')
            print(self.G)

        self.G.remove_edges_from(nx.selfloop_edges(self.G))
        

        
        # Start compute edge Ricci flow
        t0 = time.time()

        if nx.get_edge_attributes(self.G, "original_RC"):
            print("original_RC detected, continue to refine the ricci flow.")
        else:
            self.compute_ricci_curvature()

            for (v1, v2) in self.G.edges():
                self.G[v1][v2]["original_RC"] = self.G[v1][v2]["ricciCurvature"]

        # Start the Ricci flow process
        self.rc_diff = []
        for i in range(iterations):
            
            w = nx.get_edge_attributes(self.G, self.weight)
            

            for (v1, v2) in self.G.edges():
                self.G[v1][v2][self.weight] -= step * (self.G[v1][v2]["ricciCurvature"]) * (self.lengths[v1][v2])
           
            #Merge really adjacent node
            G1 = self.G.copy()
            merged = True
            while merged:
                merged = False
                for v1,v2 in G1.edges():
                    if G1[v1][v2][self.weight] < delta * 10:
                        G1 = nx.contracted_edge(G1, (v1, v2), self_loops=False)
                        merged = True
                        break
            self.G = G1

            self.compute_ricci_curvature()
            print("=== Ricciflow iteration % d ===" % int(i+1))
            #nx.write_gexf(self.G, rf"C:\Users\20978\Desktop\evolution of graph weight\one_step\modification of Ollivier’s Ricci\data\iteration\karate\iteration_{i + 1}.gexf")
            rc = nx.get_edge_attributes(self.G, "ricciCurvature")
            if rc:
                diff = max(rc.values()) - min(rc.values())
                print("Ricci curvature difference: %f" % diff)
                print("max:%f, min:%f | maxw:%f, minw:%f" % (max(rc.values()), min(rc.values()), max(w.values()), min(w.values())))
            else:
                diff = 0

           

            if diff < delta:
                print("Ricci curvature converged, process terminated.")
                break

           
                
            # clear the APSP and densities since the graph have changed.
            self.densities = {}

      #  nx.write_gexf(self.G, os.path.join(save_gexf_dir, "%d.gexf"%iterations))
        #show_results(self.G)
        print("\n%8f secs for Ricci flow computation." % (time.time() - t0))

def show_results(G, curvature="ricciCurvature"):

    # Print the first five results
    #print("Karate Club Graph, first 5 edges: ")
   # for n1,n2 in list(G.edges())[:5]:
      #  print("Ricci curvature of edge (%s,%s) is %f" % (n1 ,n2, G[n1][n2][curvature]))

    # Plot the histogram of Ricci curvatures
    plt.subplot(2, 1, 1)
    ricci_curvtures = nx.get_edge_attributes(G, curvature).values()
    plt.hist(ricci_curvtures,bins=20)
    plt.xlabel('Ricci curvature')
    plt.title("Histogram of Ricci Curvatures ")

    # Plot the histogram of edge weights
    plt.subplot(2, 1, 2)
    weights = nx.get_edge_attributes(G, "weight").values()
    plt.hist(weights,bins=20)
    plt.xlabel('Edge weight')
    plt.title("Histogram of Edge weights ")

    plt.tight_layout()
    plt.show()


    return G


def draw_graph(G, weight="weight",clustering_label="club",cutoff=1.0):
    """
    A helper function to draw a nx graph with community.
    """
    edge_trim_list = []
    for n1, n2 in G.edges():
        if G[n1][n2][weight] > cutoff:
            edge_trim_list.append((n1, n2))
    G.remove_edges_from(edge_trim_list)
    complex_list = nx.get_node_attributes(G, clustering_label)
    le = preprocessing.LabelEncoder()
    node_color = le.fit_transform(list(complex_list.values()))
    pos=nx.spring_layout(G)
    nx.draw_spring(G,nodelist=G.nodes(),
                   node_color=node_color,
                   cmap=plt.cm.rainbow,
                   alpha=0.8)
    plt.show()

    

def ARI(G, clustering, clustering_label="club"):

    complex_list = nx.get_node_attributes(G, clustering_label)
    le = preprocessing.LabelEncoder()
    y_true = le.fit_transform(list(complex_list.values()))

    if isinstance(clustering, dict):
        # python-louvain partition format
        y_pred = np.array([clustering[v] for v in complex_list.keys()])
    elif isinstance(clustering[0], set):
        # networkx partition format
        predict_dict = {c: idx for idx, comp in enumerate(clustering) for c in comp}
        y_pred = np.array([predict_dict[v] for v in complex_list.keys()])
    elif isinstance(clustering, list):
        # sklearn partition format
        y_pred = clustering
    else:
        return -1

    return metrics.adjusted_rand_score(y_true, y_pred)


def NMI(G, clustering, clustering_label="club"):
    
    
    complex_list = nx.get_node_attributes(G, clustering_label)
    
    le = preprocessing.LabelEncoder()
    y_true = le.fit_transform(list(complex_list.values()))
    
    if isinstance(clustering, dict):
        # python-louvain partition format
        y_pred = np.array([clustering[v] for v in complex_list.keys()])
    elif isinstance(clustering[0], set):
        # networkx partition format
        predict_dict = {c: idx for idx, comp in enumerate(clustering) for c in comp}
        y_pred = np.array([predict_dict[v] for v in complex_list.keys()])
    elif isinstance(clustering, list):
        # sklearn partition format
        y_pred = clustering
    else:
        return -1
    
    return metrics.normalized_mutual_info_score(y_true, y_pred)

#my_surgery(_rc.G, weight="weight", cut=1.0)

def check_accuracy(G_origin, weight="weight", clustering_label="club", plot_cut=True):
    """To check the clustering quality while cut the edges with weight using different threshold

    Parameters
    ----------
    G_origin : NetworkX graph
        A graph with ``weight`` as Ricci flow metric to cut.
    weight: float
        The edge weight used as Ricci flow metric. (Default value = "weight")
    clustering_label : str
        Node attribute name for ground truth.
    plot_cut: bool
        To plot the good guessed cut or not.

    """
    G = G_origin.copy()
    modularity, ari ,nmi = [], [], []
    maxw = max(nx.get_edge_attributes(G, weight).values())
    minw = min(nx.get_edge_attributes(G, weight).values())
    cutoff_range = np.arange(maxw, minw , -0.01)
    for cutoff in cutoff_range:
        
        edge_trim_list = []
        for n1, n2 in G.edges():
            if G[n1][n2][weight] > cutoff:
                edge_trim_list.append((n1, n2))
        G.remove_edges_from(edge_trim_list)
        if G.number_of_edges() == 0:
            cutoff_range=np.arange(maxw,minw , -0.01)
            print("No edges left in the graph. Exiting the loop.")
            break
        # Get connected component after cut as clustering
        clustering = {c: idx for idx, comp in enumerate(nx.connected_components(G)) for c in comp}
       

        # Compute modularity and ari 
        c_communities=list(nx.connected_components(G))
        modularity.append(nx.community.modularity(G, c_communities))
        
        ari.append(ARI(G, clustering, clustering_label=clustering_label))
        nmi.append(NMI(G, clustering, clustering_label=clustering_label))

    plt.xlim(maxw, 0)
    plt.xlabel("Edge weight cutoff")
    plt.plot(cutoff_range, modularity, alpha=0.8)
    plt.plot(cutoff_range, ari, alpha=0.8)
    plt.plot(cutoff_range, nmi, alpha=0.8)

    if plot_cut==False:
        plt.legend(['Modularity', 'Adjust Rand Index',"NMI"])
    
    print("max ari:", max(ari))
    print("max nmi:", max(nmi))
    print("max modularity:", max(modularity))
    plt.show()
    return max(ari), max(nmi), max(modularity)



    
  

def main():
    
    G = nx.read_gexf(r"C:\Users\20978\Desktop\test\test3\data\original\karate.gexf")
    
    G_1 = G.copy()
    
    _rnc = RhoCurvature(G_1)
    _rnc.compute_ricci_curvature()
    #nx.write_gexf(_rnc.G, (r"C:\Users\20978\Desktop\evolution of graph weight\one_step\modification of Ollivier’s Ricci\data\updated\karate.gexf_ricci.gexf"))
    _rnc.compute_ricci_flow(iterations=30, step=0.01, delta=1e-6)

    _rnc=RhoNormalizeCurvature(G_1)
    _rnc.compute_ricci_curvature()
    #nx.write_gexf(_rnc.G, (r"C:\Users\20978\Desktop\evolution of graph weight\one_step\a quasi-normalized version\data\updated\karate.gexf_ricci.gexf"))
    _rnc.compute_ricci_flow(iterations=30, step=0.01, delta=1e-6)
    
if __name__ == "__main__":
    main()

