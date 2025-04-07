from tqdm import tqdm
import torch
import pandas as pd
import numpy as np

from .simtune_autoencoder import SimtuneAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nearest_neighbour(train_features, test_features, train_labels, test_labels, model):
    train_inputs = torch.Tensor(train_features.to_numpy())
    train_inputs = train_inputs.to(device)

    test_inputs = torch.Tensor(test_features.to_numpy()).to(device)

    model = model.to(device)

    train_embeddings = model(train_inputs)
    test_embeddings = model(test_inputs)

    def get_label(idx, labels):
        return (labels.iloc[idx][0], labels.iloc[idx][1])
    
    def manhattan_distance(historic_embeddings, embedding):
        embedding = embedding.repeat(historic_embeddings.size(0), 1)
        return torch.sum(torch.abs(historic_embeddings - embedding), 1)

    result = []

    for i, row in tqdm(enumerate(test_embeddings), total=len(test_embeddings)):
        distances = manhattan_distance(train_embeddings, row)
        test_label = get_label(i, test_labels)
        closest = torch.argmin(distances)
        closest_label = get_label(closest.item(), train_labels)
        
        data = {
            "source_query": test_label[0], 
            "source_query_id": test_label[1], 
            "neighbour_query": closest_label[0], 
            "neighbour_query_id": closest_label[1],
            "distance": distances[closest].item()
        }
        result.append(data)
    
    return result

model = SimtuneAutoencoder()
checkpoint = torch.load('src/simtune/simtune.model', map_location=device)
model.load_state_dict(checkpoint())

model.eval()

train_features = pd.read_csv("src/simtune/features_train.csv")
test_features = pd.read_csv("src/simtune/features_test.csv")

train_labels = pd.read_csv("src/simtune/labels_train.csv")
test_labels = pd.read_csv("src/simtune/labels_test.csv")


test_neighbours = nearest_neighbour(train_features, test_features, train_labels, test_labels, model)

pd.DataFrame(test_neighbours).to_csv("src/simtune/neighbours_test.csv", index=False)
