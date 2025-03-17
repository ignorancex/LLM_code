import os
import torch
import torch.nn as nn
import numpy as np




relu = nn.ReLU()

if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")   

class Barycenter:
   
   
   
    # The init method or constructor
    def __init__(self,p0,p1, ADMM_opts, M, N, D_op, m1, m0, r1, r0, d1, d0, q, orig ):
     
        # Instance Variable    
        self.ADMM_opts = ADMM_opts
        self.p0 = p0
        self.p1 = p1
        self.M = M
        self.N = N
        self.D_op = D_op
        self.m1 = m1
        self.m0 = m0
        self.r1 = r1
        self.r0=r0
        self.d1 = d1
        self.d0 = d0
        self.q = q
        self.orig = orig

    def get_bary(self):
        n = self.M*self.N;
        
        # m11 = (torch.zeros(n,1)) 
        # m00 = (torch.zeros(n,1))
        # m1 = torch.complex(m11,m00).to(device)
        # m0 = torch.complex(m11,m00).to(device)

        # r1 = torch.zeros(n,1).to(device)
        # r0 = torch.zeros(n,1).to(device)

        # q = torch.zeros(n,1).to(device)
        # d3 = torch.zeros(n,1).to(device)
        # d2 = torch.zeros(n,1).to(device)
        # d1 = torch.zeros(n,1).to(device)
        # d0 = torch.zeros(n,1).to(device)

        x0 = self.p0
        x1 = self.p1
        D_op = self.D_op
        d0 = self.d0
        d1 = self.d1
        q = self.q
        m0 = self.m0
        m1 = self.m1
        r0 = self.r0
        r1 = self.r1
        K_0 =torch.real(torch.sparse.mm(torch.conj(D_op),m0)) + x0 - q - r0
        K_1 = torch.real(torch.sparse.mm(torch.conj(D_op),m1)) + x1 - q - r1
        
        for k in range(self.ADMM_opts.maxiter):
            prevm0 = m0; prevm1 = m1; 
            prevx0 = x0; prevx1 = x1; 
            prevr0 = r0;prevr1 = r1;
            prevq = q; 
            prev_K_op_mxr_0 = K_0;prev_K_op_mxr_1 = K_1;
          

           

            temp = (torch.transpose(D_op,0,1))
            matmulreal = torch.sparse.mm(torch.real(temp.to_dense()).to_sparse(),d0)
            matmulimag = torch.sparse.mm(torch.imag(temp.to_dense()).to_sparse(),d0)
            matmulcomp = torch.complex(matmulreal,matmulimag)
            m0 = prevm0 - self.ADMM_opts.tau1*matmulcomp
            abs_m = torch.abs(m0);
            m0 = torch.mul((1 - (torch.div(self.ADMM_opts.tau1,abs_m))), m0)
            ind = abs_m < self.ADMM_opts.tau1
            m0[ind] = 0

            temp = (torch.transpose(D_op,0,1))
            matmulreal = torch.sparse.mm(torch.real(temp.to_dense()).to_sparse(),d1)
            matmulimag = torch.sparse.mm(torch.imag(temp.to_dense()).to_sparse(),d1)
            matmulcomp = torch.complex(matmulreal,matmulimag)
            m1 = prevm1-self.ADMM_opts.tau1*matmulcomp
            abs_m = torch.abs(m1);
            m1 = torch.mul((1 - (torch.div(self.ADMM_opts.tau1,abs_m))), m1)
            ind = abs_m < self.ADMM_opts.tau1
            m1[ind] = 0
            
            x0 = relu( ((self.ADMM_opts.rho*self.ADMM_opts.tau1)/(1+self.ADMM_opts.rho*self.ADMM_opts.tau1))*self.p0 + (1/(1+self.ADMM_opts.rho*self.ADMM_opts.tau1))*(prevx0-self.ADMM_opts.tau1*(d0)) )
          
            x1 = relu( ((self.ADMM_opts.rho*self.ADMM_opts.tau1)/(1+self.ADMM_opts.rho*self.ADMM_opts.tau1))*self.p1 + (1/(1+self.ADMM_opts.rho*self.ADMM_opts.tau1))*(prevx1-self.ADMM_opts.tau1*d1) )
          
          
            r0 = prevr0+self.ADMM_opts.tau1*d0
            r0 = torch.mul(torch.sign(r0),relu(torch.abs(r0)-self.ADMM_opts.mu*self.ADMM_opts.tau1))
            r1 = prevr1+self.ADMM_opts.tau1*d1
            r1 = torch.mul(torch.sign(r1),relu(torch.abs(r1)-self.ADMM_opts.mu*self.ADMM_opts.tau1))
          
            q = prevq+self.ADMM_opts.tau1*(d0+d1)
            q = torch.mul(torch.sign(q), relu(torch.abs(q)-self.ADMM_opts.theta*self.ADMM_opts.tau1))
            q = torch.clamp(q,min=0,max=1)
            K_0 = torch.real(torch.sparse.mm(torch.conj(D_op),m0)) + x0 - q -r0
            K_1 = torch.real(torch.sparse.mm(torch.conj(D_op),m1)) + x1 - q -r1

            d0 = d0 + self.ADMM_opts.tau2*( 2*K_0 - prev_K_op_mxr_0 )
            d1 = d1 + self.ADMM_opts.tau2*( 2*K_1 - prev_K_op_mxr_1 )

            
        return torch.clamp(q,min=0,max=1)

          
  
 

    
