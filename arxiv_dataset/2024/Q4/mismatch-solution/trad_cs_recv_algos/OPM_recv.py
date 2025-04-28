import torch
import numpy as np
import cv2
from tqdm import tqdm


def cs_omp(y,Phi,N,K,Psi=None, device="cpu", dtype=torch.float32,verbose=True):
    residual = y.to(device)  #初始化残差
    Phi = Phi.to(device)
    index = torch.zeros(N,dtype=int,device=device)
    for i in range(N): #第i列被选中就是1，未选中就是-1
        index[i]= -1
    result= torch.zeros((N,1),dtype=dtype,device=device)
    for j in tqdm(range(K)):  #迭代次数
        
        product=torch.abs(Phi.T @ residual)
        pos=torch.argmax(product)  #最大投影系数对应的位置        
        index[pos]=1 #对应的位置取1
        inv_d=Phi[:,index>=0]
        
        inv=inv_d.T @ inv_d
        my=torch.linalg.inv(inv) @ inv_d.T
        
        a=my @ y # 最小二乘,看参考文献  
        residual=y-inv_d @ a
        if verbose:
            print(residual)
        
        # if abs(residual).mean()<1e-5:
        #     break    
    
    result[index>=0]=a.reshape((-1,1))
    if verbose:
        print(result)
    Candidate = torch.where(index>=0) #返回所有选中的列
    if Psi is not None:
        result = torch.real(Psi.to(device) @ result)
        
    return result, Candidate