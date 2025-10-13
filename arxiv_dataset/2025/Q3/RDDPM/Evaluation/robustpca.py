# Written by Mehrdad Moradi
import numpy as np
from copy import deepcopy
class RPCA:
    def __init__(self, M: np.array, max_iter: int =1000):
        self.M = M
        self.L = np.zeros(self.M.shape)
        self.S = np.zeros(self.M.shape)
        self.Y = np.zeros(self.M.shape)
        self.max_iter = max_iter
        pass
    def shrinkage_operator (self, X: np.array, threshold: float):
        return np.sign(X) * np.maximum((np.abs(X) - threshold), np.zeros(X.shape))
    def singular_value_thresholding (self, X: np.array, threshold: float):
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        return np.dot(U,np.dot(np.diag(self.shrinkage_operator(S, threshold)), Vh))
    def coherency (self,L: np.array=None):
        if L is None:
            return print('Please provide the low rank component')

        else:
            U , S , Vh = np.linalg.svd(L, full_matrices=False)
            n1, n2 = L.shape
            r=S.shape[0]
            umax=np.max([np.linalg.norm(U.T[:,ii],ord=2)**2 for ii in range(n1)])*(n1/r)
            vmax=np.max([np.linalg.norm(Vh[:,ii],ord=2)**2 for ii in range(n2)])*(n2/r)
            uv_max=(np.max(U@Vh)**2)*(n1*n2)*(1/r)
            return umax, vmax, uv_max
        
        ### A function to check if the low rank component is sparse or not. If its sparse, we cannot use this method###
    def fit (self):
        ### 
        # Function to solve the principle component pursuit problem based on the original candes et al. paper of Robust PCA
        # Assumptions: Low rank component is not sparse and sparse component is not low rank.
        # Input: Matrix M to be decomposed. For anomaly detection approaches, it is vertically long.
        # Based on the original paper, we solve the problem by alternating direction method.
        ###
        L0=self.L
        S0=self.S
        Y0=self.Y
        M=self.M
        n1, n2 = M.shape
        #lamda=1/max(n1,n2) # based on the paper
        #lamda=10**-7
        lamda=1/np.sqrt(max(n1,n2))
        mu=(n1*n2)/(4*np.linalg.norm(M,1))
        iter=0
        while np.linalg.norm(M-L0-S0,'fro') > lamda*np.linalg.norm(M,'fro') and iter<self.max_iter:
            iter+=1
            L1=self.singular_value_thresholding(M-S0+Y0/mu, mu)
            S1=self.shrinkage_operator(M-L1+Y0/mu, mu*lamda)
            Y1=Y0+mu*(M-L1-S1)
            L0=deepcopy(L1)
            S0=deepcopy(S1)
            Y0=deepcopy(Y1)
            #print('iter:',iter, 'left_tol:', np.linalg.norm(M-L0-S0,'fro'), 'right_tol:', lamda*np.linalg.norm(M,'fro'))
            #print(iter)
        return L0, S0