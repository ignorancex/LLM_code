import argparse
from utils import *
from line import Line
from tqdm import trange
import torch
import torch.optim as optim
import neptune.new as neptune
import sys
import pickle


def train_model(graph, order=2, negsamplesize=5, dimension=128, batchsize=5, epochs=1, learning_rate=0.025, negativepower=0.75):

    np.random.seed(42)
    torch.manual_seed(42)

    # Create dict of distribution when opening file
    edgedistdict, nodedistdict, weights, nodedegrees, maxindex = makeDist(graph, negativepower)

    edgesaliassampler = VoseAlias(edgedistdict)
    nodesaliassampler = VoseAlias(nodedistdict)

    batchrange = int(len(edgedistdict) / batchsize)
    print(maxindex)
    line = Line(maxindex + 1, embed_dim=dimension, order=order)

    opt = optim.SGD(line.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lossdata = {"it": [], "loss": []}
    it = 0

    print("\nTraining on {}...\n".format(device))
    for epoch in range(epochs):
        print("Epoch {}".format(epoch))
        for b in trange(batchrange):
            samplededges = edgesaliassampler.sample_n(batchsize)
            batch = list(makeData(samplededges, negsamplesize, weights, nodedegrees, nodesaliassampler))
            batch = torch.LongTensor(batch)
            v_i = batch[:, 0]
            v_j = batch[:, 1]
            negsamples = batch[:, 2:]
            line.zero_grad()
            loss = line(v_i, v_j, negsamples, device)
            loss.backward()
            opt.step()

            lossdata["loss"].append(loss.item())
            lossdata["it"].append(it)
            it += 1

    return line

def extract_embeddings(model):
    # Extract the embeddings from the model
    node_embeddings = model.nodes_embeddings.weight.data.cpu().numpy()
    if model.order == 2:
        context_embeddings = model.contextnodes_embeddings.weight.data.cpu().numpy()
        return context_embeddings

    return node_embeddings