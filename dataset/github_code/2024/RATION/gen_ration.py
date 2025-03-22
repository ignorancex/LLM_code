# -*- coding: utf-8 -*-
"""
Created on Sat May 27 11:38:05 2023

@author: jacki
"""

import json
import pickle
import numpy as np
#from sklearn.manifold import MDS
#import matplotlib.pyplot as plt
import copy
import itertools
import random
from scipy import stats
import argparse
from sklearn.metrics.pairwise import cosine_similarity
random.seed(0)
argparser=argparse.ArgumentParser()
argparser.add_argument('--rela',action='store_true')
argparser.add_argument('--path_rela', action='store_true')
argparser.add_argument('--clique_rela',action='store_true')
argparser.add_argument('--div_weight',type=float)
argparser.add_argument('--div',type=str)
argparser.add_argument('--popu',action='store_true')
argparser.add_argument('--spec',action='store_true')
argparser.add_argument('--k',type=int)
argparser.add_argument('--input_file',type=str)
argparser.add_argument('--entail_file', type=str)
argparser.add_argument('--output_file',type=str)
argparser.add_argument('--entail_thres',type=float)
args = argparser.parse_args()
class Graph:
    # init function to declare class variables
    def __init__(self, V):
        self.V = V
        self.adj = [[] for i in range(V)]
 
    def DFSUtil(self, temp, v, visited):
 
        # Mark the current vertex as visited
        visited[v] = True
 
        # Store the vertex to list
        temp.append(v)
 
        # Repeat for all vertices adjacent
        # to this vertex v
        for i in self.adj[v]:
            if visited[i] == False:
 
                # Update the list
                temp = self.DFSUtil(temp, i, visited)
        return temp
 
    # method to add an undirected edge
    def addEdge(self, v, w):
        self.adj[v].append(w)
        self.adj[w].append(v)
 
    # Method to retrieve connected components
    # in an undirected graph
    def connectedComponents(self):
        visited = []
        cc = []
        for i in range(self.V):
            visited.append(False)
        for v in range(self.V):
            if visited[v] == False:
                temp = []
                cc.append(self.DFSUtil(temp, v, visited))
        return cc
def prim(graph):
    INF=9999
    N=graph.shape[0]
    selected_node = [0]*N
    no_edge = 0
    edges=[]
    selected_node[0] = True
    dist_node=[0]*N
    while (no_edge < N - 1):
        minimum = INF
        a = 0
        b = 0
        for m in range(N):
            if selected_node[m]:
                for n in range(N):
                    if ((not selected_node[n]) and graph[m][n]):  
                        # not in selected and there is an edge
                        if minimum > graph[m][n]:
                            minimum = graph[m][n]
                            a = m
                            b = n
                            
        edges.append([a,b])
        dist_node[b]=dist_node[a]+1
        selected_node[b] = True
        no_edge += 1
    return edges,dist_node
with open('entailmenttree_space_spec.pickle', 'rb') as file:
    a = pickle.load(file)
with open('entailmenttree_space_tree.pickle', 'rb') as file:
    b = pickle.load(file)

#assert 0==1
def cond_prob(graph,indexes,prob,weight):
    assert len(indexes)<graph.shape[0]
    """
    const_prob=np.sum(prob[indexes])
    #print(const_prob)
    if len(indexes)>1:
        for i in itertools.product(indexes,repeat=2):
            if i[0]<i[1]:
                const_prob-=graph[i[0],i[1]]*weight
                #print(const_prob)
    print(const_prob)
    """
    cand_prob=np.zeros(graph.shape[0],dtype=float)
    for i in range(graph.shape[0]):
        cand_prob[i]+=prob[i]
        #print('a')
        #print(prob[i])
        media_graph=graph[i,indexes]
        cand_prob[i]-=weight*(np.sum(media_graph))
        #print(b)
        #print(weight*(np.sum(media_graph)))
        #cand_prob[i]+=const_prob
    #print('b')
    #print(weight*(np.sum(media_graph)))
    #print(cand_prob)
    #cand_prob=np.clip(cand_prob,0,1)
    #print(cand_prob)
    cand_prob=cand_prob-np.min(cand_prob)
    #print(cand_prob)
    cand_prob=np.clip(cand_prob,-10,1)
    cand_prob=np.exp(cand_prob*100)
    #print(cand_prob)
    #print(indexes)
    for i in range(graph.shape[0]):
        if i in indexes:
            cand_prob[i]=0.0
    #print(cand_prob)
    cand_prob=cand_prob/np.sum(cand_prob,axis=0,keepdims=True)
    #print(cand_prob)
    #print(cand_prob)
    return cand_prob
def gibbs_sampling(graph,prob,weight,num,iters=200):
    assert num<graph.shape[0]
    init_idx=random.sample(range(graph.shape[0]),num)
    result_counting=[]
    for i in range(iters):
        for j in range(num):
            media_idx=init_idx[:j]+init_idx[(j+1):]
            media_prob=cond_prob(graph,media_idx,prob,weight)
            select_idx=random.choices(range(graph.shape[0]),weights=media_prob,k=1)[0]
            init_idx[j]=select_idx
            if i>=100:
                #print(media_prob)
                media_idx=sorted(init_idx)
                media_idx=[str(k) for k in media_idx]
                result_counting.append('+'.join(media_idx))
    return result_counting
class Graph_Dijkstra():
 
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]
 
    def printSolution(self, dist):
        print("Vertex \t Distance from Source")
        for node in range(self.V):
            print(node, "\t\t", dist[node])
 
    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minDistance(self, dist, sptSet):
 
        # Initialize minimum distance for next node
        min = 0
 
        # Search not nearest vertex not in the
        # shortest path tree
        for v in range(self.V):
            if dist[v] > min and sptSet[v] == False:
                min = dist[v]
                min_index = v
 
        return min_index
 
    # Function that implements Dijkstra's single source
    # shortest path algorithm for a graph represented
    # using adjacency matrix representation
    def dijkstra(self, src):
 
        dist = [0] * self.V
        dist[src] = 1
        sptSet = [False] * self.V
 
        for cout in range(self.V):
 
            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minDistance(dist, sptSet)
 
            # Put the minimum distance vertex in the
            # shortest path tree
            sptSet[u] = True
 
            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for v in range(self.V):
                if (self.graph[u][v] > 0 and
                   sptSet[v] == False and
                   dist[v] < dist[u] * self.graph[u][v]):
                    dist[v] = dist[u] * self.graph[u][v]
        return dist
#assert 0==1
result={}
ratio_p=[]
ratio_r=[]
ratio_f=[]
spearman=[]
def normalize(array):
    min_value=np.min(array)
    max_value=np.max(array)
    min_value=min_value-(max_value-min_value)*0.01
    if max_value==min_value:
        return np.ones(array.shape[0])
    else:
        array=(array-min_value)/(max_value-min_value)
        return array
#b={'100585_7':b['100585_7']}
for i in b.keys():
    #print(i)
    clique_value=a[i]['clique_value']
    value_matrix=np.vstack([np.expand_dims(j,axis=0) for j in a[i]['value']])
    graph=Graph(b[i][0].shape[0]+1)
    dist_for_prim=np.zeros((b[i][0].shape[0],b[i][0].shape[0]))
    dist=np.zeros((b[i][0].shape[0],b[i][0].shape[0]))
    sent_prim=a[i]['sents']
    for j in range(dist_for_prim.shape[0]):
        for k in range(j+1,dist_for_prim.shape[0]):
            #if np.argmax(b[i][0][j,k])==2:
            if b[i][0][j,k,1]>args.entail_thres:
                if b[i][0][j,k,1]>dist_for_prim[j,k]:
                    dist_for_prim[j,k]=b[i][0][j,k,1]
                    dist_for_prim[k,j]=b[i][0][j,k,1]
            if b[i][0][j,k,1]>dist[j,k]:
                dist[j,k]=b[i][0][j,k,1]
                dist[k,j]=b[i][0][j,k,1]
    value_prim=np.zeros((value_matrix.shape[0],value_matrix.shape[1]))
    value=np.zeros((value_matrix.shape[0],value_matrix.shape[1]))
    for j in range(value_matrix.shape[0]):
        for k in range(value_matrix.shape[1]):
            #if np.argmax(value_matrix[j,k])==2:
            if value_matrix[j,k,1]>0.59:
                value_prim[j,k]=value_matrix[j,k,1]
            value[j,k]=value_matrix[j,k,1]    
    idx=np.argmax(np.sum(value_prim,axis=0))
    value_prim=value_prim[:,idx:(idx+1)]
    value=value[:,idx]
    repre=a[i]['clique'][idx]
    #print(repre)
    #sent_prim=[a[i]['clique'][idx]]+sent_prim
    dist_for_prim=np.concatenate((value_prim,dist_for_prim),axis=1)
    media_zero=np.concatenate((np.array([[0.0]]),value_prim.T),axis=1)
    dist_for_prim=np.concatenate((media_zero,dist_for_prim),axis=0)
    for j in range(dist_for_prim.shape[0]):
        for k in range(j+1,dist_for_prim.shape[0]):
            if dist_for_prim[j,k]>0:
                graph.addEdge(j,k)
    cc=graph.connectedComponents()
    cc=sorted(cc,key=lambda x:-len(x))
    #print(cc)
    cc=sorted(cc[0])
    dist_for_prim=dist_for_prim[cc,:]
    dist_for_prim=dist_for_prim[:,cc]
    clique_value=np.array(clique_value)
    cc=[j-1 for j in cc[1:]]
    assert len(cc)>0
    clique_value=clique_value[cc]
    sent_prim=[sent_prim[j] for j in cc]
    value1=value[cc]
    dist=dist[cc,:]
    dist=dist[:,cc]
    graph_dijkstra=Graph_Dijkstra(dist_for_prim.shape[0])
    graph_dijkstra.graph=dist_for_prim
    media_value=graph_dijkstra.dijkstra(0)[1:]
    #media_value=normalize(media_value[1:])
    centrality=np.ones(dist_for_prim.shape[0])
    speci=a[i]['speci']
    speci=np.array([speci[j] for j in cc])
    speci=normalize(speci)
    d=0.85
    for j in range(25):
        new_centrality=copy.deepcopy(centrality)
        media=np.sum(dist_for_prim,axis=1)+0.001
        for k in range(dist_for_prim.shape[0]):
            new_centrality[k]=1-d+d*np.sum(dist_for_prim[:,k]*centrality/media)
        #print(new_centrality[0:10])
        centrality=new_centrality
    #len_list=np.array([float(len(j.split())) for j in sent_prim])
    #len_list=np.log(len_list)
    centrality=centrality[1:]
    #len_list=normalize(len_list)
    centrality=normalize(centrality)
    #value=value1
    prob=np.ones(value1.shape)
    value=np.ones(value1.shape)
    if args.popu:
        prob=prob*centrality
    if args.spec:
        prob=prob*speci
    if args.rela:
        value=value*value1
    if args.path_rela:
        value=media_value*value
    if args.clique_rela:
        value=value*clique_value
    if type(value)!=int:
        value=normalize(value)
    prob=prob*value
    if args.k==1:
        sent_prim=list(zip(sent_prim,prob,cc))
        sent_prim1=sorted(sent_prim,key=lambda x:-x[1])
        media_sent=sent_prim1[0][0]
        count=False
        #a[i]['sents']=[j[0].upper()+j[1:] for j in a[i]['sents']]
        ratio_p.append(a[i]['ratio_p'][sent_prim1[0][2]])
        ratio_r.append(a[i]['ratio_r'][sent_prim1[0][2]])
        if a[i]['ratio_p'][sent_prim1[0][2]]!=0.0:
            ratio_f.append(2*a[i]['ratio_p'][sent_prim1[0][2]]*a[i]['ratio_r'][sent_prim1[0][2]]/(a[i]['ratio_p'][sent_prim1[0][2]]+a[i]['ratio_r'][sent_prim1[0][2]]))
        else:
            ratio_f.append(0.0)
        result[i]=[repre,[media_sent]]
    else:
        sent_prim=list(zip(sent_prim,prob))
        if args.div!='top':
            if args.div=='path':
                dist_dijkstra=np.zeros((dist_for_prim.shape[0],dist_for_prim.shape[1]))
                graph_dijkstra=Graph_Dijkstra(dist_for_prim.shape[0])
                graph_dijkstra.graph=dist_for_prim
                for j in range(dist_for_prim.shape[0]):
                    dist_dijkstra[j]=graph_dijkstra.dijkstra(j)
                dist_dijkstra=dist_dijkstra[1:,1:]
                indexes=gibbs_sampling(dist_dijkstra,prob,args.div_weight,args.k)
            elif args.div=='value':
                indexes=gibbs_sampling(dist,prob,args.div_weight,args.k)
            elif args.div=='token':
                #dist_token=np.zeros((dist_for_prim.shape[0],dist_for_prim.shape[1]))
                count=0
                token2id={}
                for j in a[i]['topic_value'].keys():
                    token2id[j]=count
                    count+=1
                emb_token=np.zeros((dist_for_prim.shape[0]-1,len(token2id)))
                topic_sent_prim=[a[i]['topic_sents'][j] for j in cc]
                assert emb_token.shape[0]==len(topic_sent_prim)
                for j in range(len(topic_sent_prim)):
                    for k in topic_sent_prim[j]:
                        if k in token2id.keys():
                            emb_token[j,token2id[k]]+=1
                dist_token= cosine_similarity(emb_token,emb_token)
                #print(dist_token)
                indexes=gibbs_sampling(dist_token,prob,args.div_weight,args.k)
            stat={}
            for j in indexes:
                if j not in stat.keys():
                    stat[j]=0
                stat[j]+=1
            #print(stat)
            stat=[(j,stat[j]) for j in stat.keys()]
            stat=sorted(stat,key=lambda x:-x[1])
            stat=stat[0][0].split('+')
            #print(stat)
            sent_prim_multiple=[sent_prim[int(j)][0] for j in stat]
        else:
            sent_prim1=sorted(sent_prim,key=lambda x:-x[1])
            sent_prim_multiple=[j[0] for j in sent_prim1[0:args.k]]
        topic_sent_prim_multiple=[]
        count=0
        #media_sents=[j[0].lower() for j in a[i]['sents']]
        #print(sent_prim_multiple)
        #print(a[i]['sents'])
        for j in range(len(a[i]['sents'])):
            if a[i]['sents'][j] in sent_prim_multiple:
                topic_sent_prim_multiple.extend(a[i]['topic_sents'][j])
                count+=1
                if count==3:
                    break
        if count!=3:
            assert 0==1
        keyword_sum=sum([a[i]['topic_value'][j] for j in a[i]['keyword']])
        word_sum=0
        keyword_sum1=0
        for j in topic_sent_prim_multiple:
            if j in a[i]['keyword']:
                keyword_sum1+=a[i]['topic_value'][j]
            if j in a[i]['topic_value'].keys():
                word_sum+=a[i]['topic_value'][j]
        if word_sum==0:
            ratio_r.append(0)
        else:
            ratio_r.append(keyword_sum1/word_sum)
        media1=set(topic_sent_prim_multiple).intersection(set(a[i]['keyword']))
        ratio_p.append(sum([a[i]['topic_value'][k] for k in media1])/keyword_sum)
        if ratio_p[-1]!=0.0:
            ratio_f.append(2*ratio_p[-1]*ratio_r[-1]/(ratio_p[-1]+ratio_r[-1]))
        else:
            ratio_f.append(0.0)
        result[i]=[repre,sent_prim_multiple]
    #break
print('Specificity:'+str(np.mean(ratio_p)))
print('Popularity:'+str(np.mean(ratio_r)))
#print(np.mean(ratio_f))
#assert 0==1
name=args.output_file
if args.popu:
    name=name+'popu'
if args.spec:
    name=name+'spec'
if args.rela:
    name=name+'rela'
if args.path_rela:
    name=name+'path_rela'
if args.clique_rela:
    name=name+'clique_rela'
name=name+str(args.div_weight)+args.div+'_'+str(args.k)+'rationales'
with open(name+'.json','w') as file:
    json.dump(result,file)

