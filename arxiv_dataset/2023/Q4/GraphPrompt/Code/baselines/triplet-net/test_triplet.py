from dataset import  get_all_data, load_data, data_split
import ssl
import heapq
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from gensim.models import KeyedVectors
from models import TripletNet, SimpleEmbedding

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
words2vec = KeyedVectors.load_word2vec_format('./wikipedia-pubmed-and-PMC-w2v.bin', binary=True)
biowords2vec = KeyedVectors.load_word2vec_format('./BioWordVec_PubMed_MIMICIII_d200.vec.bin', binary=True)

pdist = nn.PairwiseDistance(p=2, eps=1e-50)

def word2vec(mention):
    mention = mention.split(' ')
    embedding = [0 for i in range(200)]
    for i in mention:
        if i in words2vec.key_to_index:
            embedding += words2vec[i]
        elif i in biowords2vec.key_to_index:
            embedding += biowords2vec[i]
        # else:
        #     print('not found word', i)
    if True in torch.isnan(torch.Tensor(embedding)):
        print(mention, '有毛病')
    return embedding

def cosine_similarity(a, b):
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a / a_norm, (b / b_norm).T)

def candidate_set1(id, mention, vec, data, threshold, size):
    candidate = []
    # m = word2vec(mention)
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    positive = 1
    for i in data:
        entity_id = i[1]
        entity = i[0]
        if id == entity_id:
            positive = 1
        else:
            positive = 0
        vecc = word2vec_dict[entity]
        candidate.append([entity_id, vecc, cos(vec, vecc).item(), positive])
    candidate.sort(key=lambda x: x[2], reverse=True)
    candidate = candidate[:size]
    # print(candidate)
    return [x for x in candidate if x[2] >= threshold]

# calculate jaccard overlap of two mentions
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

def candidate_set2(id, mention, data, threshold, size):
    candidate = []
    positive = 1
    for i in data:
        entity_id = i[1]
        entity = i[0]
        if id == entity_id:
            positive = 1
        else:
            positive = 0
        candidate.append([id, entity_id, jaccard_overlap(mention, entity), positive])
    candidate.sort(key=lambda x: x[2], reverse=True)
    candidate = candidate[:size]
    # print(candidate)
    return [x for x in candidate if x[2] >= threshold]


t1 = 0
t2 = 0
k1 = 50
k2 = 10
# ff = open('debug.txt', 'a+', encoding='utf-8')
# for file in ['ddpheno.obo', 'ceph.obo', 'cdno.obo', 'eco.obo', 'xao.obo', 'cl.obo', 'hp.obo', 'doid.obo', 'fbbt.obo', 'mp.obo']:
for file in ['fbbt.obo', 'mp.obo']:
    with open('result.txt', 'a', encoding='utf-8') as f:
        f.write('%s \n' % file)
    print(file)
    word2vec_dict = {}
    # ssl._create_default_https_context = ssl._create_unverified_context
    # ssl._create_default_http_context = ssl._create_unverified_context
    # get_all_data()
    name_array, query_id_array, mention2id, edge_index = load_data(filename='./data/datasets/'+file)
    # print(query_id_array)
    for (mention, id) in query_id_array:
        word2vec_dict[mention] = word2vec(mention)
    for entity in name_array:
        word2vec_dict[entity] = word2vec(entity)
    print('get word vectors')
    ll = []
    for i in word2vec_dict.values():
        ll.append(i)
    mean = np.mean(ll, axis=0)
    std = np.std(ll, axis=0)
    for i in word2vec_dict.keys():
        word2vec_dict[i] = (word2vec_dict[i] - mean) / std
    # print(np.mean(word2vec_dict.values()))
    queries_train, queries_valid, queries_test = data_split(query_id_array, is_unseen=True, test_size=0.33, seed=0)
    entity = []
    for i in range(len(name_array)):
        entity.append([name_array[i], i])

    # print(entity)
    # print(queries_train)
    train_vec = []
    for i in queries_train:
        train_vec.append(word2vec_dict[i])
    train_vec = np.array(train_vec)
    test_vec = []
    for i in queries_test:
        test_vec.append(word2vec_dict[i])
    test_vec = np.array(test_vec)
    valid_vec = []
    for i in queries_valid:
        valid_vec.append(word2vec_dict[i])
    valid_vec = np.array(valid_vec)
    name_vec = []
    for i in name_array:
        name_vec.append(word2vec_dict[i])
    name_vec = np.array(name_vec)
    train_cos = cosine_similarity(train_vec, name_vec)
    valid_cos = cosine_similarity(valid_vec, name_vec)
    test_cos = cosine_similarity(test_vec, name_vec)
    print(train_cos.shape)
    train_set = []
    for count, i in enumerate(queries_train):
        # if count % 100 == 0:
        #     print(count, '/', len(queries_train))
        # print(i, mention2id[i])
        vec = word2vec_dict[i]
        l1 = []
        candidate1 = heapq.nlargest(k1, range(len(train_cos[count])), train_cos[count].take)
        for count_j in candidate1:
            if train_cos[count][count_j] >= t1:
                if mention2id[i] == count_j:
                    l1.append([count, count_j, 1])
                else:
                    l1.append([count, count_j, 0])
        # l1 = candidate_set1(mention2id[i], i, vec, entity, t1, k1)
        l2 = candidate_set2(mention2id[i], i, entity, t2, k2)
        # mention, vector, value, positive
        positive_candidates = []
        negative_candidates = []
        for posi in l1:
            # print(posi)
            if posi[2] == 1:
                if posi[1] not in positive_candidates:
                    positive_candidates.append(posi[1])
            if posi[2] == 0:
                if posi[1] not in negative_candidates:
                    negative_candidates.append(posi[1])
        for posi in l2:
            if posi[3] == 1:
                if posi[1] not in positive_candidates:
                    positive_candidates.append(posi[1])
            if posi[3] == 0:
                if posi[1] not in negative_candidates:
                    negative_candidates.append(posi[1])
        if mention2id[i] not in positive_candidates:
            positive_candidates.append(mention2id[i])
        for posi in positive_candidates:
            for nega in negative_candidates:
                train_set.append([word2vec_dict[name_array[posi]], vec, word2vec_dict[name_array[nega]]])
        # if True in torch.isnan(torch.Tensor(vec)):
        #     print('我的天', i, vec)
            # for nega in negative_candidates:
            #     # print(posi, i, nega)
            #     train_set.append(torch.cat((torch.Tensor(word2vec_dict[name_array[posi]]), torch.Tensor(vec), torch.Tensor(word2vec_dict[name_array[nega]])), 0))
    #print(train_set)
    print('get train data', len(train_set))

    # valid_set = []
    # for count, i in enumerate(queries_valid):
    #     # if count % 1000 == 0:
    #     #     print(count, '/', len(queries_valid))
    #     # print(i[0], i[1])
    #     vec = word2vec_dict[i]
    #     # l1 = candidate_set1(mention2id[i], i, vec, entity, 0, len(name_array))
    #     # l2 = candidate_set2(j, i[1], train_data, t2, k2)
    #     # mention, vector, value, positive
    #     positive_candidates = []
    #     negative_candidates = []
    #     for j in entity:
    #         name = j[0]
    #         id = j[1]
    #         if id == mention2id[i]:
    #             positive_candidates.append(id)
    #         else:
    #             negative_candidates.append(id)
    #     valid_set.append([positive_candidates, vec, negative_candidates])
    # print('get valid data')
    #
    # test_set = []
    # for count, i in enumerate(queries_test):
    #     # if count % 1000 == 0:
    #     #     print(count, '/', len(queries_test))
    #     # print(i[0], i[1])
    #     vec = word2vec_dict[i]
    #     # l1 = candidate_set1(mention2id[i], i, vec, entity, 0, len(name_array))
    #     # l2 = candidate_set2(j, i[1], train_data, t2, k2)
    #     # mention, vector, value, positive
    #     positive_candidates = []
    #     negative_candidates = []
    #     for j in entity:
    #         name = j[0]
    #         id = j[1]
    #         if id == mention2id[i]:
    #             positive_candidates.append(id)
    #         else:
    #             negative_candidates.append(id)
    #     test_set.append([positive_candidates, vec, negative_candidates])
    # print('get test data')

    simplenet = SimpleEmbedding().to(device)
    net = TripletNet(simplenet).to(device)
    print(net)
    criterion = nn.MSELoss(reduction='sum')
    # criterion = nn.CrossEntropyLoss()
    lr = 0.1
    epochs = 10
    largest = 50
    batchsize = 16
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.1, nesterov=True)
    alpha = 0
    # Train
    correct = 0
    total = 0
    best_accuracy = 0
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad()
        loss = torch.zeros(1)
        loss.requires_grad = True
        data_x = []
        data_y = []
        data_z = []
        for i, candidate in enumerate(train_set, 0):
            batch_count = 0
            # loss = torch.zeros(1)
            # loss.requires_grad = True
            data_x.append(candidate[0])
            data_y.append(candidate[1])
            data_z.append(candidate[2])
            # x, y, z = candidate[:200].resize_(1, 1, 200), candidate[200:400].resize_(1, 1, 200), candidate[400:].resize_(1, 1, 200)
            # x, y, z = x.to(device), y.to(device), z.to(device)
            # x_outputs, y_outputs, z_outputs = net(x, y, z)
            # loss_1 = criterion(x_outputs, y_outputs)
            # loss_2 = criterion(y_outputs, z_outputs)
            # if True in torch.isnan(x_outputs) or True in torch.isnan(y_outputs) or True in torch.isnan(z_outputs):
            #     print('出大问题', i, epoch)
            #     print('x', x, x_outputs)
            #     print('y', y, y_outputs)
            #     print('z', z, z_outputs)
            #     for _, param in net.named_parameters():
            #         print(param.grad)
            #     raise RuntimeError('完蛋')
            # output = torch.dot(x_outputs.squeeze(0).squeeze(0), y_outputs.squeeze(0).squeeze(0)).unsqueeze(0)
            # label = [0]
            # for z in candidate[1]:
            #     # print('before', z)
            #     z = torch.Tensor(word2vec_dict[name_array[z]]).resize_(1, 1, 200)
            #     # print('after', z)
            #     # x, y, z = x.to(device), y.to(device), z.to(device)
            #     z = z.to(device)
            #     # x_outputs, y_outputs, z_outputs = net(x, y, z)
            #     z_outputs = simplenet(z)
            #     # x_outputs = torch.sigmoid(x_outputs)
            #     # y_outputs = torch.sigmoid(y_outputs)
            #     # z_outputs = torch.sigmoid(z_outputs)
            #     output = torch.cat([output, torch.dot(z_outputs.squeeze(0).squeeze(0), y_outputs.squeeze(0).squeeze(0)).unsqueeze(0)], 0)
            #     label = torch.zeros(1)
            #     if True in torch.isnan(x_outputs) or True in torch.isnan(y_outputs) or True in torch.isnan(z_outputs):
            #         print('出大问题', i, epoch)
            #         print('x', x, x_outputs)
            #         print('y', y, y_outputs)
            #         print('z', z, z_outputs)
            #         for _, param in net.named_parameters():
            #             print(param.grad)
            #         raise RuntimeError('完蛋')
                # print(x.size(), x_outputs.size())
                # loss_1 = torch.dot(x_outputs.squeeze(0).squeeze(0), y_outputs.squeeze(0).squeeze(0))
                # loss_2 = torch.dot(y_outputs.squeeze(0).squeeze(0), z_outputs.squeeze(0).squeeze(0))
                # loss = loss + loss_1 - loss_2
                # batch_count += 1
                # print('loss', loss, 'batch count', batch_count, 'loss1', loss_1, 'loss2', loss_2)
                # if loss_1 - loss_2 > alpha:
                #     continue
                # else:
                #     loss = loss + loss_1 - loss_2
                #     batch_count += 1
                    # print('loss', loss, 'batch count', batch_count, 'loss1', loss_1, 'loss2', loss_2)
            # if batch_count == 0:
            #     continue
            # print('loss', loss)
            # loss = criterion(torch.unsqueeze(output, 0), label.long())
            # loss = torch.sigmoid(loss)
            # print('loss', loss, 'batch count', batch_count)
            if (i + 1) % batchsize == 0:
                #  or i == len(train_set) - 1
                x = torch.Tensor(np.array(data_x)).to(device)
                y = torch.Tensor(np.array(data_y)).to(device)
                z = torch.Tensor(np.array(data_z)).to(device)
                # print(len(data_x), x.shape)
                x_outputs, y_outputs, z_outputs = net(x, y, z)
                data_x = []
                data_y = []
                data_z = []
                loss = criterion(x_outputs, y_outputs) - criterion(y_outputs, z_outputs)
                if i // batchsize == 0:
                    loss = loss / batchsize
                else:
                    loss = loss / (i // batchsize)
                # print('epoch', epoch, 'loss', loss, file=ff)
                loss.backward()
                torch.nn.utils.clip_grad_value_(parameters=net.parameters(), clip_value=1)
                #torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10, norm_type=2)
                # for p in net.parameters():
                #     nn.utils.clip_grad(p, 1000)
                optimizer.step()
                # optimizer.zero_grad()
                # loss = torch.zeros(1)
                # loss.requires_grad = True
            # print('output', output, file=ff)
            # LogSoftmax = nn.LogSoftmax(dim=1)
            # NLLLoss = nn.NLLLoss()
            # logoutput = LogSoftmax(torch.unsqueeze(output, 0))
            # print('softmax', logoutput, file=ff)
            # print('手算', NLLLoss(logoutput, label.long()), file=ff)
            # for _, param in net.named_parameters():
            #     print(epoch, file=ff)
            #     print('param', param, file=ff)
            #     print('grad', param.grad, file=ff)
            #     if True in torch.isnan(param):
            #         break
        with torch.no_grad():
            TP = 0
            FP = 0
            for count, i in enumerate(queries_valid):
                id = mention2id[i]
                result = []
                candidate = heapq.nlargest(largest, range(len(valid_cos[count])), valid_cos[count].take)
                data_x = []
                data_y = []
                if mention2id[i] not in candidate:
                    candidate.append(mention2id[i])
                for num, count_j in enumerate(candidate):
                    # data_x.append(word2vec_dict[name_array[count_j]])
                    # data_y.append(word2vec_dict[i])
                    x, y = torch.Tensor(word2vec_dict[name_array[count_j]]).resize_(1, 200),  torch.Tensor(word2vec_dict[i]).resize_(1, 200)
                    x, y, z = x.to(device), y.to(device), z.to(device)
                    x_outputs, y_outputs, z_outputs = net(x, y, x)
                    result.append([criterion(x_outputs, y_outputs), count_j])
                result.sort(key=lambda x: x[0], reverse=False)
                # print(result)
                if result[0][1] == id:
                    TP += 1
                else:
                    FP += 1
            print(epoch, '/', epochs, "Valid Accuracy = ", TP / (TP + FP))
            if TP / (TP + FP) > best_accuracy:
                best_accuracy = TP / (TP + FP)
                torch.save(net.state_dict(), file[0:-4] + 'seen.pkl')
            else:
                for p in optimizer.param_groups:
                    p['lr'] /= 4
    net.load_state_dict(torch.load(file[0:-4] + 'seen.pkl'))
    with torch.no_grad():
        TP = 0
        FP = 0
        TP_10 = 0
        result = []
        for count, i in enumerate(queries_valid):
            id = mention2id[i]
            result = []
            candidate = heapq.nlargest(largest, range(len(valid_cos[count])), valid_cos[count].take)
            if mention2id[i] not in candidate:
                candidate.append(mention2id[i])
            for count_j in candidate:
                x, y = torch.Tensor(word2vec_dict[name_array[count_j]]).resize_(1, 200), torch.Tensor(
                    word2vec_dict[i]).resize_(1,
                                              200)
                x, y, z = x.to(device), y.to(device), z.to(device)
                x_outputs, y_outputs, z_outputs = net(x, y, x)
                result.append([criterion(x_outputs, y_outputs), count_j])
            result.sort(key=lambda x: x[0], reverse=False)
            # print(result[0][0])
            if result[0][1] == id:
                TP += 1
            else:
                FP += 1
            l = min(len(result), 50)
            for j in range(l):
                if result[j][1] == id:
                    TP_10 += 1
                    break
        with open('result.txt', 'a', encoding='utf-8') as f:
            f.write("Valid Accuracy = %.8f  \n" % (TP / (TP + FP)))
            f.write("Valid Accuracy10 = %.8f  \n" % (TP_10 / (TP + FP)))
        print("Valid Accuracy = ", TP / (TP + FP))
        print("Valid Accuracy10 = ", TP_10 / (TP + FP))
        TP = 0
        FP = 0
        TP_10 = 0
        for count, i in enumerate(queries_test):
            id = mention2id[i]
            result = []
            candidate = heapq.nlargest(largest, range(len(test_cos[count])), test_cos[count].take)
            if mention2id[i] not in candidate:
                candidate.append(mention2id[i])
            for count_j in candidate:
                x, y = torch.Tensor(word2vec_dict[name_array[count_j]]).resize_(1, 200), torch.Tensor(
                    word2vec_dict[i]).resize_(1,
                                             200)
                x, y, z = x.to(device), y.to(device), z.to(device)
                x_outputs, y_outputs, z_outputs = net(x, y, x)
                result.append([criterion(x_outputs, y_outputs), count_j])
            result.sort(key=lambda x: x[0], reverse=False)
            if result[0][1] == id:
                TP += 1
            else:
                FP += 1
            l = min(len(result), 50)
            for j in range(l):
                if result[j][1] == id:
                    TP_10 += 1
                    break
        with open('result.txt', 'a', encoding='utf-8') as f:
            f.write("Test Accuracy = %.8f  \n" % (TP / (TP + FP)))
            f.write("Test Accuracy10 = %.8f  \n" % (TP_10 / (TP + FP)))
        print("Test Accuracy = ", TP / (TP + FP))
        print("Test Accuracy10 = ", TP_10 / (TP + FP))
