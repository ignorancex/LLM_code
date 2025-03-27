import torch
import numpy as np
n_way, shots_per_way, n_ul = 5, 4, 16
nun = np.arange(0, 25).reshape(5, 5)
a = torch.from_numpy(nun)
#a = a.sum(dim=1).unsqueeze(1).repeat(1, 5)
print(a)
print(a.diag())
print(a.diag().diag())



n_pos = 2
n_l = n_way * shots_per_way
# positive mask
T1 = np.eye(int(n_l/n_pos))
T2 = np.ones((n_pos, n_pos))
mask_pos_lab = torch.FloatTensor(np.kron(T1, T2))
# print(mask_pos_lab)
# print(mask_pos_lab.size())  20, 20
T3 = torch.cat([mask_pos_lab, torch.zeros(n_l, n_ul)], dim=1)
T4 = torch.zeros(n_ul, n_l+n_ul)
mask_pos = torch.cat([T3, T4], dim=0)
# print(mask_pos.size())  20+16, 20+16
T1 = 1-np.eye(n_way)

T2 = np.ones((shots_per_way, shots_per_way))
mask_neg_lab = torch.FloatTensor(np.kron(T1, T2))
# print(mask_neg_lab)
# print('mask_neg_lab:', mask_neg_lab.size())  # 20, 20
T3 = torch.cat([mask_neg_lab, torch.ones(n_l, n_ul)], dim=1)
T4 = torch.ones(n_ul, n_l + n_ul)  # dummy
mask_neg = torch.cat([T3, T4], dim=0)
# print(mask_neg)
T3 = torch.cat([torch.zeros(n_l, n_l), torch.ones(n_l, n_ul)], dim=1)
mask_distract = torch.cat([T3, T4], dim=0)
# print(mask_distract)
