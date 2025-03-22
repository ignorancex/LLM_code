from train import train_model, extract_embeddings
import igraph as ig
import pandas as pd
import numpy as np

def pipeline(graph, negsamplesize=5, dimension=128, batchsize=5, epochs=1, learning_rate=0.025, negativepower=0.75):

    model1 = train_model(
        graph=graph,
        order=1,
        negsamplesize=negsamplesize,
        dimension=dimension,
        batchsize=batchsize,
        epochs=epochs,
        learning_rate=learning_rate,
        negativepower=negativepower
    )

    model2 = train_model(
        graph=graph,
        order=2,
        negsamplesize=negsamplesize,
        dimension=dimension,
        batchsize=batchsize,
        epochs=epochs,
        learning_rate=learning_rate,
        negativepower=negativepower
    )

    embedding1 = extract_embeddings(model1)
    embedding2 = extract_embeddings(model2)

    embedding = np.concatenate((embedding1, embedding2), axis=1)

    node_ids = graph.vs['name']
    node_labels = graph.vs['label']
    embeddings_df = pd.DataFrame(embedding)
    embeddings_df.insert(0, 'node_id', node_ids)
    embeddings_df['label'] = node_labels
    num_features = embedding.shape[1]
    embeddings_df.columns = ['node_id'] + [f'feature_{i}' for i in range(num_features)] + ['label']

    embeddings_df.to_csv('Model_Evaluation/Model/LINE/embedding.csv', index=False)
    return embeddings_df

#graph_ = r"/Users/georgeilli/Desktop/NeurIPS/NeurIPS-2024-GANTAT/Network Construction/less_Tclin_Signalink_PIN_graph.graphml"
#graph = ig.Graph.Load(graph_, format='graphml')


#pipeline(graph, negsamplesize=5, dimension=128, batchsize=5, epochs=30, learning_rate=0.01, negativepower=0.75)