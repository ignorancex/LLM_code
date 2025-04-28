
import matplotlib.pyplot as plt
from gem.utils import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from gem.evaluation import evaluate_graph_reconstruction as gr
from time import time
import pandas as pd
import pickle
from gem.embedding.hope import HOPE
import os
# File that contains the edges. Format: source target
# Optionally, you can add weights as third column: source target weight
# Specify whether the edges are not directed
"""isDirected =False
d = "[5]"
beta = "[0.00390625]"
edge_f="/media/DATA2/sinem/25-10-2023/HOPER/data/hoper_PPI/PPI_example_data/example.edgelist"
protein_id="/media/DATA2/sinem/25-10-2023/HOPER/data/hoper_PPI/PPI_example_data/proteins_id.csv"
"""

import ast
def hope_repesentation_call(edge_f,protein_ids,isDirected,d_lst,beta_lst):
# Load graph
    d = ast.literal_eval(d_lst)
    beta = ast.literal_eval(beta_lst)
    
    protein_id=pd.read_csv(protein_ids)
    G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed =isDirected )
    G = G.to_directed()



    for x in d:

        for y in beta:
      
            models = []

        #HOPE takes embedding dimension (d) and decay factor (beta) as inputs
            models.append(HOPE(d=x, beta=y))
        
            for embedding in models:
                print ('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges()))
                t1 = time()
            # Learn embedding - accepts a networkx graph or file with edge list
                Y, t = embedding.learn_embedding(graph=G, is_weighted=False, no_python=True)
                print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
            # Evaluate on graph reconstruction
                MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G, embedding, Y, None)
            #---------------------------------------------------------------------------------
                print(("\tMAP: {} \t precision curve: {}\n\n\n\n"+'-'*100).format(MAP,prec_curv[:5]))
                emb= embedding.get_embedding()
                print(emb)
            emb_list=list(emb)
            protein_id_list=protein_id['0'].tolist()
            ent_vec = {'Entry':protein_id_list,'Vector':emb_list}
            ent_vec_data_frame = pd.DataFrame(ent_vec)
            path=os.getcwd()
            path_data = os.path.join(path, "data") 
            if os.path.exists(path_data):  
                output = open(os.path.join(path_data,'HOPE_'+'d_'+str(x) + '_' +'beta_'+str(y) +'.pkl'), 'wb')
                pickle.dump(ent_vec_data_frame, output)
                output.close()
            else:
                os.mkdir(path_data)
                output = open(os.path.join(path_data,'HOPE_'+'d_'+str(x) + '_' +'beta_'+str(y) +'.pkl'), 'wb')
                pickle.dump(ent_vec_data_frame, output)
                output.close()
          
#emb=hope_repesentation_call(edge_f,protein_id,isDirected,d,beta)            
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 6:
        print("Usage: python script.py param1 param2 param3 param4 param5")
    else:
        param1 = sys.argv[1]
        param2 = sys.argv[2]
        param3 = sys.argv[3]
        param4 = sys.argv[4]
        param5 = sys.argv[5]
        hope_repesentation_call(param1,param2,param3,param4,param5)
        
  

