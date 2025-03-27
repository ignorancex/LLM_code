import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from gensim.models import KeyedVectors
from models import TripletNet, SimpleEmbedding

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
words2vec = KeyedVectors.load_word2vec_format('/Users/liuyucong/Downloads/wikipedia-pubmed-and-PMC-w2v.bin', binary=True)
# 1biowords2vec = KeyedVectors.load_word2vec_format('/Users/liuyucong/Downloads/BioWordVec_PubMed_MIMICIII_d200.vec.bin', binary=True)


def word2vec(mention):
    mention = mention.split(' ')
    embedding = [0 for i in range(200)]
    for i in mention:
        if i in words2vec.key_to_index:
            embedding += words2vec[i]
        # elif i in biowords2vec.key_to_index:
        #     embedding += biowords2vec[i]i
    return torch.Tensor(embedding)


def sum_embedding(sentence_embedding):
    return sentence_embedding.sum(axis=0)


def readdictionary(file_path):
    pmids = []
    info = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.split("||")
        if len(line) == 2:
            pmid, mention = line
        else:
            raise NotImplementedError()
        if '|' in pmid:
            pmid = pmid.split('|')
        else:
            pmid = [pmid]
        info.append([pmid, mention, word2vec(mention)])
    return info


def readfile(train_file, test_file):
    train_data = readdictionary(train_file)
    test_data = readdictionary(test_file)
    return train_data, test_data


def candidate_set1(pmid, mention, vec, data, threshold, size):
    candidate = []
    # m = word2vec(mention)
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    positive = 1
    for i in data:
        if pmid in i[0]:
            positive = 1
        else:
            positive = 0
        if i[1] != mention:
            # print(i[0],i[1])
            # x = word2vec(i[1])
            # print(m, x)
            # print(cos(m, x))
            candidate.append([i[1], i[2], cos(vec, i[2]).item(), positive])
    candidate.sort(key=lambda x: x[2], reverse=True)
    candidate = candidate[:size]
    # print(candidate)
    return [x for x in candidate if x[2] >= threshold]


def jaccard_overlap(x, y):
    # print(x, y)
    x = x.split(' ')
    y = y.split(' ')
    sec = 0
    uni = 0
    uni_set = list(set().union(x, y))
    sec_set = list(set(x) & set(y))
    # print(len(sec_set)/len(uni_set))
    return len(sec_set)/len(uni_set)


def candidate_set2(pmid, mention, data, threshold, size):
    candidate = []
    positive = 1
    for i in data:
        if pmid in i[0]:
            positive = 1
        else:
            positive = 0
        if i[1] != mention:
            candidate.append([i[1], i[2], jaccard_overlap(mention, i[1]), positive])
    candidate.sort(key=lambda x: x[2], reverse=True)
    candidate = candidate[:size]
    # print(candidate)
    return [x for x in candidate if x[2] >= threshold]


t1 = 0.7
t2 = 0.1
k1 = 3
k2 = 7
# print(jaccard_overlap('bacteremic infections due to neisseria', 'dna virus infections'))
# print(jaccard_overlap('bacteremic infections due to neisseria', 'screw worm infections'))
train_data, test_data = readfile('/Users/liuyucong/Downloads/ncbi-disease/test_dictionary.txt', '/Users/liuyucong/Downloads/ncbi-disease/test_dictionary.txt')
print(candidate_set2('D016870', 'bacteremic infections due to neisseria', train_data, t2, k2))
train_set = []
count = 0
for i in train_data:
    count += 1
    if count > 100:
        break
    print(i[0], i[1])
    for j in i[0]:
        l1 = candidate_set1(j, i[1], i[2], train_data, t1, k1)
        l2 = candidate_set2(j, i[1], train_data, t2, k2)
        # mention, vector, value, positive
        positive_candidates = []
        negative_candidates = []
        for posi in l1:
            if posi[3] == 1:
                if posi[0] not in positive_candidates:
                    positive_candidates.append(posi[1])
            if posi[3] == 0:
                if posi[0] not in negative_candidates:
                    negative_candidates.append(posi[1])
        for posi in l2:
            if posi[3] == 1:
                if posi[0] not in positive_candidates:
                    positive_candidates.append(posi[1])
            if posi[3] == 0:
                if posi[0] not in negative_candidates:
                    negative_candidates.append(posi[1])
        for posi in positive_candidates:
            # train_pos_set = []
            # train_mention_set = []
            # train_neg_set = []
            for nega in negative_candidates:
                # print(torch.cat((posi, i[2], nega), 0))
                # train_pos_set.append(posi.numpy().tolist())
                # train_mention_set.append(i[2].numpy().tolist())
                # train_neg_set.append(nega.numpy().tolist())
                train_set.append(torch.cat((posi, i[2], nega), 0))

test_set = []
count = 0
for i in test_data:
    count += 1
    if count > 100:
        break
    # print(i[0], i[1])
    for j in i[0]:
        l1 = candidate_set1(j, i[1], i[2], test_data, t1, k1)
        # l2 = candidate_set2(j, i[1], train_data, t2, k2)
        # mention, vector, value, positive
        positive_candidates = []
        negative_candidates = []
        for posi in l1:
            if posi[3] == 1:
                if posi[0] not in positive_candidates:
                    positive_candidates.append(posi[1])
            if posi[3] == 0:
                if posi[0] not in negative_candidates:
                    negative_candidates.append(posi[1])
        test_set.append([positive_candidates, i[2], negative_candidates])

simplenet = SimpleEmbedding()
net = TripletNet(simplenet).to(device)
print(net)
criterion = nn.MSELoss()
lr = 0.001
epochs = 20
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
alpha = 0
# Train
correct = 0
total = 0
for epoch in range(epochs):
    print(epoch, '/', epochs)
    running_loss = 0.0
    jacobian_loss = 0.0
    correct = 0
    total = 0
    for i, candidate in enumerate(train_set, 0):
        # if times >= 10000:
        #     break
        # x = torch.Tensor(candidate[0])[None,:,:]
        # y = torch.Tensor(candidate[1])[None,:,:]
        # z = torch.Tensor(candidate[2])[None,:,:]
        # print(x.size())
        x, y, z = candidate[:200].resize_(1, 1, 200), candidate[200:400].resize_(1, 1, 200), candidate[400:].resize_(1, 1, 200)
        x, y, z = x.to(device), y.to(device), z.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        x_outputs, y_outputs, z_outputs = net(x, y, z)
        print(x.size(), x_outputs.size())
        loss_1 = criterion(x_outputs, y_outputs)
        loss_2 = criterion(y_outputs, z_outputs)
        loss = loss_2 - loss_1 + alpha
        if loss < 0:
            loss = torch.tensor(0.0, requires_grad=True)
        loss.backward()
        optimizer.step()

TP = 0
FP = 0
for i in test_set:
    positive_candidates = i[0]
    mention = i[1]
    negative_candidates = i[2]
    candidates = []
    for j in positive_candidates:
        x, y, z = j.resize_(1, 1, 200), mention.resize_(1, 1, 200), mention.resize(1, 1, 200)
        x_outputs, y_outputs, z_outputs = net(x, y, z)
        candidates.append([criterion(x_outputs, y_outputs).item(), 1])
    for j in negative_candidates:
        x, y, z = j.resize_(1, 1, 200), mention.resize_(1, 1, 200), mention.resize(1, 1, 200)
        x_outputs, y_outputs, z_outputs = net(x, y, z)
        candidates.append([criterion(x_outputs, y_outputs).item(), 0])
    candidates.sort(key=lambda x: x[0], reverse=True)
    print('candidates', candidates)
    if len(candidates) != 0:
        if candidates[0][1] == 1:
            TP += 1
        else:
            FP += 0
print("Accuracy = ", TP / (TP + FP))