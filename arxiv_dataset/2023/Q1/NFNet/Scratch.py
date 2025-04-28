import torch
import torch.nn as nn
import math
from Utils_torch_version import Network, make_bas_thin, make_bas
import numpy as np

# random = torch.rand(3, dtype=torch.cfloat)
# print('random', random)

# b_ij_list = nn.Parameter(torch.tensor(math.pi) * torch.rand(3, dtype=torch.cfloat))

# print('b_ij_list', b_ij_list)
# print('b_ij_list[1]',b_ij_list[1])

# J = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
# print("J", J)

# N=4
# B_l = torch.zeros((N, N), dtype=torch.cfloat)
# i,j = [0,2]
# # This
# B_l[i,i], B_l[j,j], B_l[i,j], B_l[j,i]= b_ij_list[0], b_ij_list[1], b_ij_list[2], torch.conj(b_ij_list[2])

# print('B_l', B_l)
# jc = torch.complex(torch.tensor(0.0), torch.tensor(1.0))

# V_l = torch.matrix_exp(jc*B_l) # This might potentially be a wrong scipy result
# print('V_l', V_l)


# jc = torch.complex(torch.tensor(0.0), torch.tensor(1.0))

# class V_l(torch.nn.Module):
#     def __init__(self, conn, N):
#         super().__init__()
#         self.b_ij_list = nn.Parameter(torch.tensor(math.pi) * torch.rand(3, dtype=torch.cfloat))
#         self.conn, self.N = conn, N 
#         self.V_l = self.set_V_l(self.b_ij_list)

#     # Return the V_l matrix of this layer, which is to be multiplied with other V_l's from other layers
#     # V_l is calculated as exp(iB_l), where B_l matrix is Hermitian, as prepared by the values in b_ij_list
#     def get_V_l(self): 
#         return self.V_l

#     # In a variational setting the values of the b_ij's should be conveniently resettable
#     def set_V_l(self, b_ij_list):
#         self.b_ij_list = b_ij_list
#         B_l = torch.zeros((self.N, self.N), dtype=torch.cfloat)
#         i,j = self.conn
#         # This
#         B_l[i,i], B_l[j,j], B_l[i,j], B_l[j,i]= self.b_ij_list[0], self.b_ij_list[1], self.b_ij_list[2], torch.conj(self.b_ij_list[2])

#         self.V_l = torch.matrix_exp(jc*B_l) # This might potentially be a wrong scipy result
#         return

# V_l_list = [] # A list of modules
# conn_list = [[0,1], [1,2]]
# for l in range(len(conn_list)):
#     V_l_list.append(V_l(conn_list[l], N))

# # Convert to modulelist
# V_l_list = nn.ModuleList(V_l_list)
# print('V_l_list', V_l_list)

# print('V_l_list params', V_l_list.parameters())
# for param in V_l_list.parameters():
#     print('param', param)
#     print('reset')
#     param.data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.cfloat)

# for param in V_l_list.parameters():
#     print('param.weight', param.data)


# print('test true', torch.tensor(1.0)==1)
# # Testing out the pytorch integration
# N = 4
# conn_list = [[0,1], [1,2], [2,3]]
# L = len(conn_list)
# net = Network(conn_list, N)
# y = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
# x = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, .0, 1.0]])
# amps = net(y, x)
# print('amps', amps)

# b_ij_list = torch.tensor(math.pi) * torch.rand(L, 3, dtype=torch.cfloat)
# net.manual_set_params(b_ij_list)


# x = torch.tensor([[1,0]*int(N/2)]) # The 1010101... single basis state
# x_batch = x.repeat(10, 1)
# print('x_batch', x_batch)

# A = np.array([ [[1,2,3],
#         [4,5,6]],

#         [[10, 20, 30],
#         [40, 50, 60]]
#          ])
# B = np.zeros((3,3))
# B[:, [0,1]]=1
# print(B)

# print(A[[0,1],[0,1]])

# print(make_bas_thin(geometry=[4,4]))

# N=4
# B_l = nn.Parameter(torch.zeros((N, N)), requires_grad=False) # First freeze all parameters
# print('B_l[1,1]', B_l[1,1])
# print('B_l', B_l)
# # B_l[0,0] = nn.Parameter(torch.tensor(1.5)) # requires gradient by default
# # print('B_l', B_l)
# parameter = nn.Parameter(torch.tensor(1.5)) # This requires gradient, so the gradient to this scalar will be copied through.
# B_l[0,0] = parameter
# print('B_l', B_l)
# print('B_l[0,0].requires_grad', B_l[0,0].requires_grad)
# s = torch.sum(B_l)
# s.backward()
# print(B_l[0,0].grad)
# print('B_l.requires_grad', B_l.requires_grad)
# print('B_l.grad', B_l.grad)
# print('B_l[0,0].is_leaf', B_l[0,0].is_leaf)
# print('parameter.grad', parameter.grad)


# A = torch.tensor([ [[1,2,3],
#         [4,5,6],
#         [5,5,5]],

#         [[10, 20, 30],
#         [40, 50, 60],
#         [9,9,1]],

#         [[0, 0, 0],
#         [1, 1, 610],
#         [10, 3, 5]]
#          ])

# print(  A[[0,1,1],[1,1,1]]  )
# print(A[0][[0,1],:][:,[1,2]])
# print(torch.arange(19,-1,-1))

# M = torch.ones((2**3, 2**3), dtype=torch.cfloat)


# M = torch.from_numpy(np.arange(64)).float().view(8, 8)

# print('test0',M[[0,1,2],[[2],[3],[4]]]    )
# print('test1',M[ [[0],[1],[2]],[2,3,4] ]    )
# print('test2', M[[0,1,2], [2,3,4]])
# print('M', M)

# a = torch.Tensor([[3, 5],
#                 [1, 2],
#                 [5, 7]])
# # print("a:", a)
# # a[0:2, :] = torch.Tensor([[4, 4],
# #                             [1, 1]])
# # a[[0,2], :][:, :] = torch.eye(2)
# # print("After modifying a[0:2, 1]:", a)

# testlist = torch.tensor([1,2,3,4])
# mask = torch.tensor([1,0,0,1], dtype=torch.bool)
# print(testlist.masked_select( mask))
# test = torch.ones(5, dtype=torch.int)
# print(test.bool())
# print([1,0]+[1])
# print([1,0]*int(3/2)+1)


# bas_samples = make_bas(geometry=(2,2))
# print('bas_samples', bas_samples)

K = torch.tensor([[1,0],
				[1,1]])
x = torch.tensor([5, 9])
y = torch.tensor([[1], [5]])
print(x@K@y)





