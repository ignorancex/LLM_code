import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
# from test_balance import TG
import pandas
import math

def tri_to_adj(triple, n):
    A = torch.sparse_coo_tensor(triple[:, :2].T, triple[:, 2], size=[n, n]).to_dense()
    A = A + A.T - torch.diag(torch.diag(A))
    return A

def get_meta_grad(triple_copy):
    edges = Variable(triple_copy[:,2:], requires_grad = True)
    triple_torch = torch.cat((triple_copy[:, :2], edges), 1)
    adj = tri_to_adj(triple_torch, len(triple_copy))
    adj_unsign = torch.abs(adj)
    atk_loss = (0.5 * (1 + (torch.trace(torch.matmul(torch.matmul(adj.to_dense(), adj.to_dense()), adj.to_dense())) / torch.trace(torch.matmul(torch.matmul(adj_unsign.to_dense(), adj_unsign.to_dense()), adj_unsign.to_dense())))))
    print("atk_loss=", atk_loss.item())
    atk_loss.requires_grad_(True)
    atk_loss.backward()
    meta_grad = edges.grad.data.cpu().numpy()
    return np.concatenate((triple_copy[:,:2], meta_grad), 1)

def gradmax(triple, ptb_rate):
    budget = int(ptb_rate * len(triple))
    triple_copy = torch.from_numpy(triple.copy())
    flag = True
    perturb = []
    with tqdm(total=budget) as pbar:
        for i in range(math.ceil(budget / 10)):
            pbar.update(10)
            if flag:
                flag = False
            else:
                triple_copy = torch.from_numpy(triple_copy)
            meta_grad = get_meta_grad(triple_copy)
            v_grad = np.zeros((len(meta_grad), 3))
            for j in range(len(meta_grad)):
                v_grad[j,0] = meta_grad[j,0]
                v_grad[j,1] = meta_grad[j,1]
                if triple_copy[j,2] == -1 and meta_grad[j,2] < 0:
                    v_grad[j,2] = meta_grad[j,2]
                elif triple_copy[j,2] == 1 and meta_grad[j,2] > 0:
                    v_grad[j,2] = meta_grad[j,2]
                else:
                    continue

            v_grad = v_grad[np.abs(v_grad[:,2]).argsort()]
            K = -1
            triple_copy = triple_copy.data.numpy()
            for k in range(20):
                while v_grad[K][:2].astype('int').tolist() in perturb:
                    K -= 1
                target_grad = v_grad[int(K)]
                target_index = np.where(np.all((triple[:,:2] == target_grad[:2]), axis=1))[0][0]
                triple_copy[target_index,2] -= 2 * np.sign(target_grad[2])
                perturb.append([int(target_grad[0]), int(target_grad[1])])
                K -= 1

        new_triple_index = triple_copy[:,:2].T
        new_triple_value = triple_copy[:,2].T
        new_adj = torch.sparse_coo_tensor(indices=new_triple_index, values=new_triple_value, size=[len(triple), len(triple)])

    np.savetxt("bitcoinalpha_train_balance_attack_" + str(ptb_rate) + ".txt", triple_copy, fmt='%d')
    print("finish")

mask_ratio = 0.1
triple = np.loadtxt('bitcoinalpha_train.txt')
print(triple)

gradmax(triple, mask_ratio)
