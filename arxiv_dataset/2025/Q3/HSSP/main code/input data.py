import numpy as np 
import pandas as pd
from pandas import Series,DataFrame
from numpy import random,mat
data = pd.read_excel('./15.xlsx',header=None) # 读取文件路径
data = data.values
Mlen = len(np.unique([i for j in data for i in j]))
#Mlen = len(np.unique(data))
#M=np.zeros((Mlen,Mlen),dtype=float)#行列数是所有节点的个数
#print(Mlen)
Max = data.max()+1
M=np.zeros((Max,Max),dtype=float)#行列数是所有节点的个数
print(Max)

u=0
p=0.02
noise = []
parent1 = []
child = []
Input_one = []
Input_two = []

for j in range(1):
    n = np.random.normal(u,p,M.shape[1])#n = np.random.normal(u,p,10)
    c = np.random.randint(25, size=M.shape[1])#uniform,randint
    input_one = n + c
    input_one = input_one.tolist()
    input_two = input_one.copy()
    Input_one.append(input_one)
    Input_two.append(input_two)
    for i in data:
        print(i)
        node_a = i[1]
        node_b = i[0]
        print(node_a)
        print(node_b)
        M[node_a][node_b]=1
        M[node_b][node_a]=1
        A = input_one[node_a] + input_one[node_b]
        input_one[node_b] = A
        parent1.append(A)
        input_three = input_two + input_one
    child.append(input_three)
print(M)

#parent = [parent1[i:i + len(data)] for i in range(0, len(parent1), len(data))]
nested_i = child
nested_d = Input_one
leni=len(child[0])
#lend=len(parent[0])
nested_d1=[i for j in nested_d for i in j]
nested_i1=[i for j in nested_d for i in j]

parent = []
for n in range(len(Input_one)):
    A = [x for x in Input_one[n] if x not in Input_two[n]]
    parent.append(A)

lend=len(parent[0])
lend
