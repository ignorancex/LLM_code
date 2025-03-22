import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
fm.fontManager.addfont('times_with_simsun.ttf')
from sklearn.manifold import MDS
from itertools import combinations
from collections import Counter
class SemanticMap:
    def __init__(self, tfM, featNames, adjM=None, GT_adj=None, zeroOcc=0):
        self.tfM = tfM # (N, D)
        self.featNames = {ix:i for ix, i in enumerate(featNames)}
        self.n_nodes = len(featNames)
        self.n_instances = tfM.shape[0]
        self.ins2feats = {i : [self.featNames[j] for j in range(self.n_nodes) if self.tfM[i,j] != 0] for i in range(self.n_instances)}
        self.mergeFeat()
        self.GT_adj = GT_adj
        if not self.GT_adj is None:
            self.visualizeSM(nx.from_numpy_array(self.GT_adj), savePath="output/GT.pdf")
        self.zeroOcc = zeroOcc
        self.adjM = adjM

    def mergeFeat(self):
        # merge the feature columns if they are totally the same
        unique_featNames = []
        unique_columns = []
        for i, col in enumerate(self.tfM.T.tolist()):
            if col not in unique_columns or sum(col) == 0:
                unique_columns.append(col)
                unique_featNames.append(self.featNames[i])
            else:
                ix = unique_columns.index(col)
                unique_featNames[ix] += "/" + self.featNames[i]
        self.featNames = {ix:i for ix, i in enumerate(unique_featNames)}
        self.tfM = np.stack(unique_columns,axis=1)
        print(self.featNames) # for debug
        print(self.tfM)

    def constG(self, norm=False):
        '''
        To construct a undirected acyclic (fully connected) graph according to a term-feature matrix
        '''
        mat = self.tfM.T @ self.tfM + (1 - self.tfM).T @ (1 - self.tfM) * self.zeroOcc # 0 has the fastest convergence.
        print(mat.shape)
        if norm:
            mat = mat.astype("float")
            mat /= (np.sum(self.tfM.T,axis=1,keepdims=True) + np.sum(self.tfM, axis=0,keepdims=True)-mat)
            mat *= 100
        if self.adjM is None:
            self.adjM = np.triu(mat)

        np.fill_diagonal(self.adjM, val=0) # (D, D)

        print(self.adjM)
        self.G = nx.from_numpy_array(self.adjM)
        # self.visualizeSM(self.G)
        for u, v in self.G.edges():
            weight = self.G.get_edge_data(u, v)['weight']
            # print(f"Edge ({u}, {v}) has weight {weight}.")
        components = list(nx.connected_components(self.G))
        print("There are %d connected components at the previous Graph"%(len(components)))
        if len(components) != 1: # Due to the zero-occurrence, the components are always connected.
            print(components)

    def accWithGT(self, T):
        # print((nx.to_numpy_array(T)!=0))
        # print(self.GT_adj)
        return np.sum((nx.to_numpy_array(T)!=0) == self.GT_adj) / (self.GT_adj.shape[0] ** 2)
    
    def get_subGraph_connected(self, T):
        nodes = list(T.nodes)
        self.poss_subG_list = []

        for r in range(2, len(nodes)+1):
            for node_combination in combinations(nodes, r):
                sub_graph = T.subgraph(node_combination)
                if nx.is_connected(sub_graph):
                    self.poss_subG_list.append(sub_graph)

    def check_subGraph_connectivity(self, G, selected_ins, onlyUnconn=False):
        self.get_subGraph_connected(G)
        connected_flag = []
        self.subG_list = []
        if onlyUnconn:
            selected_ins = [i for i,j in zip(selected_ins, self.connected_flag) if not j]
            print("Selected inx: ", selected_ins)
            for si in selected_ins:
                print(f"{si}: {'-'.join(self.ins2feats[si])}")
        for i in self.tfM[selected_ins, :]: # O(N)
            subG = G.subgraph({ix for ix,j in enumerate(i) if j==1})
            connected = nx.is_connected(subG)
            connected_flag.append(connected)
            self.subG_list.append(subG)

        recall = sum(connected_flag) / len(self.subG_list)
        prec = sum(connected_flag) / len(self.poss_subG_list)
        print(sum(connected_flag), len(self.poss_subG_list))
        F1 = 2 * (recall * prec) / (recall + prec + 1e-10)
        # prec = 0
        # F1 = 0
        Deg = [d for _, d in G.degree()]
        Deg_mean = np.mean(Deg)
        Deg_std = np.std(Deg)

        self.metrics = (prec, recall, F1, Deg_mean, Deg_std)
        if onlyUnconn:
            for ix, f in zip(selected_ins, connected_flag):
                self.connected_flag[ix] = f
        else:
            self.connected_flag = connected_flag

    def highlight_subgraphs(self, G, subgraphs, linestyles=None, colors=None):
        """ TODO: NOT COMPETE!
        Highlight connected subgraphs in a NetworkX graph by drawing lines around them.

        Args:
            G: The NetworkX graph.
            subgraphs: A list of subgraphs, where each subgraph is a list of nodes.
            linestyles: A list of linestyles for the bounding lines (optional).
            colors: A list of colors for the bounding lines (optional).
        """

        if linestyles is None:
            linestyles = ["solid"] * len(subgraphs)
        if colors is None:
            colors = ["red", "blue", "green"]  # Cycle through colors for visibility

        lay = nx.planar_layout(G)
        nx.set_node_attributes(G, lay, 'pos')  # Assign positions to the 'pos' attribute
        # Calculate bounding boxes for each subgraph
        bounding_boxes = []
        for subgraph in subgraphs:
            pos = nx.get_node_attributes(G, 'pos')
            xs = [pos[node][0] for node in subgraph]
            ys = [pos[node][1] for node in subgraph]
            xmin, ymin = min(xs), min(ys)
            xmax, ymax = max(xs), max(ys)
            padding = 0.1  # Add padding around the subgraph
            bounding_boxes.append([xmin - padding, ymin - padding, xmax + padding, ymax + padding])

        # Draw the graph
        # nx.draw(G, with_labels=True)
        nx.draw(G, lay, labels=self.featNames,with_labels=True, node_size=150, font_size=10,alpha=0.8, font_family='times_with_simsun.ttf')
        # nx.draw_networkx_labels(T, pos, labels=self.featNames, font_family=font_path)  # Draw labels
        weights = {k:round(nx.get_edge_attributes(G, 'weight')[k],1) for k,v in nx.get_edge_attributes(G, 'weight').items()}
        nx.draw_networkx_edge_labels(G,lay,edge_labels=weights, font_size=5)

        # Draw bounding lines for each subgraph
        for i, (xmin, ymin, xmax, ymax) in enumerate(bounding_boxes):
            ls = linestyles[i % len(linestyles)]  # Cycle through linestyles
            c = colors[i % len(colors)]  # Cycle through colors
            plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], ls=ls, color=c, linewidth=2)

        plt.show()

    def draw_multiple_subgraphs(self, subgraphs, subtitles):
        num_subgraphs = len(subgraphs)
        fig, axes = plt.subplots(num_subgraphs, 1, figsize=(4*num_subgraphs, 15))
        
        for i, (subgraph, subtitle, ax) in enumerate(zip(subgraphs, subtitles, axes)):
            # Draw the subgraph
            nx.draw(subgraph, ax=ax, with_labels=True)

            # Add subtitle
            ax.set_title(subtitle, fontsize=12)

        plt.show()

    def visualizeSM(self, T, hit=0, acc_GT=0.0, showIns=False, savePath=None):
        '''
        To visualize a semantic map with instances on it.
        '''
        font_path = 'times_with_simsun.ttf'
        chinese_font = fm.FontProperties(fname=font_path)
        font_names = ['Noto Sans Mandaic', 'Arial', 'Comic Sans MS', ]
        plt.rcParams['font.family'] = [ 'Heiti TC','Arial', 'Helvetica', 'Times New Roman']
        fig = plt.figure()#figsize=(100,100))
        # if not acc_GT is None:
        #     plt.title("Semantic Map with weights %.2f; %d hit;\n Recall: %.2f; Precision: %.2f; F1: %.2f; Std degree: %.2f \n ACC: %.2f"%(T.size(weight="weight"), hit,self.metrics[1], self.metrics[0], self.metrics[2], self.metrics[4], acc_GT))
        # else:
        #     plt.title("Semantic Map with weights %.2f; %d hit;\n Recall: %.2f; Precision: %.2f; F1: %.2f; Std degree: %.2f"%(T.size(weight="weight"), hit,self.metrics[1], self.metrics[0], self.metrics[2], self.metrics[4]))
        pos = nx.spring_layout(T, k=.5, seed=42)
        nx.draw(T, pos, labels=self.featNames,with_labels=True, node_size=400, font_size=10,alpha=0.8, font_family='Times New Roman + SimSun')
        # nx.draw_networkx_labels(T, pos, labels=self.featNames, font_family=font_path)  # Draw labels
        weights = {k:round(nx.get_edge_attributes(T, 'weight')[k],1) for k,v in nx.get_edge_attributes(T, 'weight').items()}
        edge_width = [T[u][v]['weight'] * 1.0 for u, v in T.edges()]
        nx.draw_networkx_edges(T, pos, width=edge_width, edge_color="plum")
        if not self.GT_adj is None:
            nx.draw_networkx_edges(nx.from_numpy_array(self.GT_adj), pos, style="-.", width=0.8,alpha=0.7)
            # 创建图例条目
            # 在边末端添加点（突出）
            for u, v in nx.from_numpy_array(self.GT_adj).edges():
                x_end, y_end = pos[v]  # 获取边的末端坐标
                plt.scatter(x_end, y_end, s=15, color='black')  # 在末端绘制红色圆点
            
            legend_elements = [
                Line2D([0], [0], color='plum', lw=2, label='Generated Network', ),  # 实线
                Line2D([0], [0], color='black', lw=1, linestyle='--', label='True Network')  # 虚线
            ]
            font_properties = fm.FontProperties(family=['Times New Roman'], size=12)
            plt.legend(handles=legend_elements, loc='upper left', prop=font_properties)
        nx.draw_networkx_edge_labels(T,pos,edge_labels=weights, font_size=5)


        if savePath:
            plt.savefig(savePath, dpi=300)
        plt.show()
        plt.close()

        if showIns:
            subgraphs = self.subG_list
            num_subgraphs = len(self.subG_list)
            subtitles = [str(i) for i in range(num_subgraphs)]
            fig, axes = plt.subplots(num_subgraphs//2, 2, figsize=(num_subgraphs*3, 5))
            axes = [aa for a in axes for aa in a]
            
            for i, (subgraph, subtitle, ax) in enumerate(zip(subgraphs, subtitles, axes)):
                # nx.draw(subgraph, ax=ax, with_labels=True)
                print(ax)
                nx.draw(subgraph, pos, ax=ax,labels=self.featNames,with_labels=True, node_size=100, font_size=7,alpha=0.8, font_family='Times New Roman + SimSun')
                ax.set_title(subtitle, fontsize=6)
                weights = {k:nx.get_edge_attributes(subgraph, 'weight')[k] for k,v in nx.get_edge_attributes(subgraph, 'weight').items()}
                nx.draw_networkx_edge_labels(subgraph,pos,ax=ax,edge_labels=weights, font_size=5)

            plt.show()

    def visualizeSM_2D(self):
        self.constG()
        # Perform MDS to embed features into 2D space
        mds = MDS(n_components=2)
        feature_positions = mds.fit_transform(self.tfM.T)

        # Plot the features as points
        plt.figure(figsize=(8, 6))
        plt.scatter(feature_positions[:, 0], feature_positions[:, 1])

        # Label the points (optional)
        font_path = 'times_with_simsun.ttf'
        chinese_font = fm.FontProperties(fname=font_path)
        feature_names = [v for k,v in self.featNames.items()]
        annotations = []
        for i, name in enumerate(feature_names):
            annotations.append(plt.text(feature_positions[i, 0], feature_positions[i, 1], name,font_properties=chinese_font))
        # adjustText.adjust_text(annotations)#,arrowprops=dict(arrowstyle='->', color='red'))   

        plt.title("Feature Similarity Visualization using MDS")
        plt.show()

    def get_optimal_SpanningTrees(self, acc_thr = 1.0, figPath=None, moreEdges=False):
        '''
        To get all the spanning trees given a graph. The trees should be ordered according to its sum of weights.

        moreEdges: whether to add more edges (circles) to realize the 100% coverage rate.
        '''
        self.constG()
        # sys.exit(0)
        if not self.GT_adj is None:
            acc_GT_0 = self.accWithGT(self.G)
            print(f"Accuracy in terms of GT before becoming a tree: {acc_GT_0}")
            print("Precision: %.5f" % (self.n_nodes / (2 ** (self.n_nodes*(self.n_nodes-1)/2) - 1)))
            print("Size: %d" % self.G.size(weight="weight"))
        if  not self.GT_adj is None: # GT
            print("---Statistics of GT:")
            self.check_subGraph_connectivity(nx.from_numpy_array(self.GT_adj), range(len(self.tfM)))
            print(self.metrics)
            print("Size: %d" % (nx.from_numpy_array(self.GT_adj * nx.adjacency_matrix(self.G).toarray()).size(weight="weight")))
            print("---")

        trees = nx.algorithms.tree.mst.SpanningTreeIterator(self.G, minimum=False)
        selected_ins = range(len(self.tfM))#[0,1,8,-1,-6]##[0,1]
        optimal_trees = []
        std_list = []
        prec_list = []
        recall_list = []
        weight_list = []
        acc_list = []
        for ix, t in enumerate(trees):
            if ix ==0:#% 10000 == 0:# and ix > 30000:
                print(f"This is the id of the spanning tree: {ix}")
                self.check_subGraph_connectivity(t, selected_ins)
                print(ix, self.connected_flag)

                # Add more edges to cover all the instances
                if moreEdges:
                    # sorted_edges_by_weights = sorted(self.G.edges.data(), key=lambda x:x[2]['weight'], reverse=True)
                    possible_edges = []
                    for si, flag in zip(selected_ins, self.connected_flag):
                        if not flag:
                            jx_list = [jx for jx in range(self.n_nodes) if self.tfM[si, jx] != 0]
                            possible_edges += [(jx_list[jjx], jx_list[kkx], {'weight': self.G.get_edge_data(jx_list[jjx], jx_list[kkx])['weight']}) for jjx in range(len(jx_list)) for kkx in range(jjx+1, len(jx_list))]

                    # 统计边的次数（只考虑节点对，忽略权重）
                    edges_without_attributes = [(u, v) for u, v, _ in possible_edges]
                    edge_counts = Counter(edges_without_attributes)

                    # 构建一个包含次数和最大权重的结构
                    edge_info = {}
                    for u, v, attr in possible_edges:
                        edge = (u, v)
                        weight = attr['weight']
                        if edge not in edge_info:
                            edge_info[edge] = {"count": edge_counts[edge], "max_weight": weight}
                        else:
                            edge_info[edge]["max_weight"] = max(edge_info[edge]["max_weight"], weight)

                    # 排序：按次数和最大权重降序
                    sorted_edges_by_number = sorted(
                        edge_info.items(),
                        key=lambda x: x[1]["count"] + x[1]["max_weight"],
                        reverse=True,
                    )
                    sorted_edges_by_number = [(edge[0], edge[1], {'weight': info['max_weight'], 'count': info['count']}) for edge, info in sorted_edges_by_number]
                    # 'CO-CD'
                    for iix, e in enumerate(sorted_edges_by_number):
                        if sum(self.connected_flag) == len(self.connected_flag):
                            break
                        if (e[0], e[1]) in t.edges or (e[1], e[0]) in t.edges:
                            continue
                        else:
                            print(f"The added edge is between {self.featNames[e[0]]} and {self.featNames[e[1]]}.")
                            t.add_edge(e[0], e[1], **e[2])
                            self.check_subGraph_connectivity(t, selected_ins, onlyUnconn=True)
                            print(iix, self.connected_flag)
                            merged_graph = nx.Graph()
                            for subG in self.subG_list:
                                merged_graph = nx.compose(merged_graph, subG)
                            self.visualizeSM(t, hit=ix, acc_GT=1.1, showIns=False, savePath=figPath[:-4]+f"_{iix}_Subcompose"+figPath[-4:] if figPath else None)

                print(">>> Intrinsic Evaluation >>>")
                self.check_subGraph_connectivity(t, selected_ins)
                print(f"Precision: {self.metrics[0]} \t Recall: {self.metrics[1]} \t F1: {self.metrics[2]}")
                print("Summed Weight: %d" % t.size(weight="weight"))
                print(f"Network typology of degree mean: {self.metrics[3]} \t std: {self.metrics[4]}")
                # prec_list.append(self.metrics[0])
                recall_list.append(self.metrics[1])
                weight_list.append(t.size(weight="weight"))
                std_list.append(self.metrics[4])

                # For merged graph
                print(">>>>>>For merged graph>>>>>")
                merged_graph = nx.Graph()
                for subG in self.subG_list:
                    merged_graph = nx.compose(merged_graph, subG)
                self.check_subGraph_connectivity(merged_graph, selected_ins)
                print(f"Precision: {self.metrics[0]} \t Recall: {self.metrics[1]} \t F1: {self.metrics[2]}")
                print("Summed Weight: %d" % t.size(weight="weight"))
                print(f"Network typology of degree mean: {self.metrics[3]} \t std: {self.metrics[4]}")
                print(">>>>>>END For merged graph>>>>>--")
                print(">>> Extrinsic Evaluation >>>")
                if not self.GT_adj is None:
                    acc_GT = self.accWithGT(t)
                    print(f"ACC_GT: {acc_GT}")
                    acc_list.append(acc_GT)
                else:
                    acc_GT = None
                # self.highlight_subgraphs(t, subG_list)
                self.wrongcases = [ix for ix,i in enumerate(self.connected_flag) if not i]
                print("Wrong instances id: %s" % self.wrongcases)
                

                if self.metrics[1] >= acc_thr: # recall
                    print(self.metrics[1], acc_thr)
                    optimal_trees.append(t)
                    self.visualizeSM(t, hit=ix, acc_GT=acc_GT, showIns=False, savePath=figPath[:-4]+f"_{ix}"+figPath[-4:] if figPath else None)
                if ix == 0:
                    print(f"End to the maximum iteration: {ix}")
                    break
                # self.visualizeSM(t)

        print("There are %d optimal trees" % len(optimal_trees))
        # _ = [self.visualizeSM(t) for t in optimal_trees]

        print("Correlation to recall: %f" % np.corrcoef(acc_list, recall_list)[0, 1])
        print("Correlation to std: %f" % np.corrcoef(acc_list, std_list)[0, 1])
        print("Correlation to size: %f" % np.corrcoef(acc_list, weight_list)[0, 1])
