import numpy as np
import torch
import torch.utils.data as data
from sklearn.metrics import ndcg_score
from tqdm import tqdm

import os
DEBUG=os.getenv("DEBUG")

class EarlyStop:
    def __init__(self, num_steps, ascending=False, thold_stop=None):
        self.num_steps = num_steps
        self.ascending = ascending
        self.num_evals = 0
        self.best_eval_num = 0
        if ascending:
            self.best_eval = -1000000
        else:
            self.best_eval = +1000000
        self.check_thold = False
        if thold_stop is not None:
            self.check_thold = True
            arr = thold_stop.split(':')
            self.thold_step = int(arr[0])
            self.thold_value = float(arr[1])
            
          

    def step(self, value):
        self.num_evals = self.num_evals + 1
        if self.check_thold and (self.num_evals > self.thold_step):
            if ((self.ascending and (max(value, self.best_eval) < self.thold_value))
                or (not self.ascending and (min(value, self.best_eval) > self.thold_value))
                ):
                print("THOLD CHECK HIT")
                return True

        if (self.ascending and value > self.best_eval) or (not self.ascending and value < self.best_eval):
            self.best_eval_num = self.num_evals
            self.best_eval = value
            return False
        else:
            if (self.num_evals - self.best_eval_num ) > self.num_steps:
                print("TREND CHECK HIT")
                return True
        return False

def count_parameters(model):
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += p.numel()

    return total

def get_compression(cmp_str):
    '''
        helper to give concise input for compression config
    '''
    if cmp_str.startswith('2:'):
        x = cmp_str.split(':')[1]
        power = int(x)
    else:
        raise NotImplementedError
    return 1.0/2**power



def train(model, train_dataset, epochs, batch_size, args, device):
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
    for epoch in range(epochs):
        model.train()
        losses = []
        norms = []
        num_samples = 0
        for batch_idx, (users, pos, neg) in enumerate(train_loader):
            model.zero_grad()
            loss1, loss2 = model.bpr_loss(torch.LongTensor(users).to(device), 
                                          torch.LongTensor(pos).to(device), torch.LongTensor(neg).to(device))
            loss = loss1 + args.reg_lambda * loss2
            loss.backward()
            optimizer.step()
            losses.append(float(loss1.detach().cpu()))
            norms.append(float(loss2.detach().cpu()))
            num_samples += len(users)
        if DEBUG:
            print(epoch,  "-->", np.mean(losses), np.mean(norms), "samples in epoch", num_samples)


def evaluate(model, train_dataset, test_dataset, total_items, device, user_batch=10, full=False):
    model = model.to(device)
    with torch.no_grad():
        all_items = torch.from_numpy(np.arange(total_items)).to(device).long()
        zeros_tensor = torch.zeros(total_items, device=device, dtype=torch.int64)

        users = test_dataset.unique_users
        if not full:
            users = users[:user_batch]
        ndcgs = []
        topk = []

        batches = (len(users) + user_batch - 1) // user_batch

        for i in tqdm(range(batches)):
            users_batch = torch.from_numpy(users[i*user_batch:(i+1)*user_batch]).to(device).long().reshape(-1,1)
            user_tensor = zeros_tensor + users_batch
            items_tensor = all_items.repeat((users_batch.numel(),1))
            fshape = user_tensor.shape

            scores = np.array(model(user_tensor.reshape(-1), items_tensor.reshape(-1)).cpu())
            scores = scores.reshape(*fshape)
            
            mask = np.zeros(scores.shape)
            target = np.zeros(scores.shape)

            for j in range(users_batch.numel()):
                u = users[i*user_batch:(i+1)*user_batch][j]
                train_items = train_dataset.pos_items[u]
                test_items = test_dataset.pos_items[u]
                # mask the train items
                scores[j][train_items] = -10000
                target[j][test_items] = 1
            
            # compute metrics
            ndgc = ndcg_score(target, scores, k=20)
            #meanret = np.mean(target[np.argsort(scores)[-20:]])
            ndcgs.append(ndgc)
            #topk.append(meanret)

        #print("[",full,"]Evaluation: ndgc:", np.mean(ndcgs))
        model.cpu()
        return np.mean(ndcgs)


import re


## partition data into batches ##
def partition_list_by_size(_l, _n):
    return [_l[b * _n : (b + 1) * _n] for b in range((len(_l) + _n - 1) // _n)]


## get sample weight per each device (from sampled set of devices) ##
def get_norm_weights_devices(_devices):
    norm_weights = {}
    Z1 = 0.0
    device_ids = []
    for dev_i in _devices:
        id = dev_i.get_id()
        Z1 += len(dev_i.localdata_idx)
        device_ids.append(id)
    for dev_i in _devices:
        norm_weights[dev_i.get_id()] = float(len(dev_i.localdata_idx)) / Z1

    return norm_weights


def split_line(line):
    return re.findall(r"[\w']+|[.,!?;]", line)


def line_to_indices(line, word2id, max_words=25):
    unk_id = len(word2id)
    line_list = split_line(line)  # split phrase in words
    indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
    indl += [unk_id] * (max_words - len(indl))
    return indl


ALL_LETTERS = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
)


def word_to_indices(word, max_words=80):
    indices = []
    for c in word[:max_words]:
        indices.append(ALL_LETTERS.find(c))
    indices += [80] * (max_words - len(indices))
    return indices


# def word_to_indices(word, max_words = 80):
#     indices = []
#     for c in word[:]:
#         indices.append(ALL_LETTERS.find(c))
#     #indices += [80]*(max_words-len(indices))
#     print(word)
#     print(indices)
#     print(len(indices))

#     return indices

#
#

if __name__ == '__main__':
    ea = EarlyStop(2, False)
    values = [0.1, 0.11, 0.09, 0.12, 0.11, 0.11, 0.11, 0.11, 0.10, 0.09]
    for v in values:
        c = ea.step(v)
        print(v)
        if c :
            break

#
