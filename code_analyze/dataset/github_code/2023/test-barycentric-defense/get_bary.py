import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
# from divergence import get_divergence
from BeckOT import Barycenter
import torch.nn.functional as F

relu = nn.ReLU()

if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")   



# D_op, m1, m0, r1, r0, q, d1, d0 = get_divergence(x,n,batch_size)
def get_bary(inputs,inputs_,ADMM_opts,batch_size,flag,n=1024, x=32):
        # n = 1024*4
        x = torch.tensor([x,x])



        sz = torch.mul(x[1],x[0])
        y=torch.eye(sz)


        y = y.to_sparse()
        s=y.clone()
        # print(y,s)
        z=y.to_dense()
        z =torch.roll(y.to_dense(),1,0)
        r = s.to_dense() - z


        # print(r.to_sparse())
        r[0,sz-1] = 0
        ind = torch.arange(2, sz, x[0])

        r[:,ind] = torch.zeros(sz,x[1])


        dz =torch.roll(y.to_dense(),x[1].item(),0)  
        dr =  s.to_dense() - dz
        ind = torch.arange(sz-x[0], sz, 1)

        dr[:,ind] = torch.zeros(sz,x[0])

        D_op = torch.complex(r,dr)
        D_op = D_op.to_sparse().to(device)
        m11 = (torch.zeros(n,batch_size)) 
        m00 = (torch.zeros(n,batch_size))
        m1 = torch.complex(m11,m00).to(device)
        m0 = torch.complex(m11,m00).to(device)

        r1 = torch.zeros(n,batch_size).to(device)
        r0 = torch.zeros(n,batch_size).to(device)

        q = torch.zeros(n,batch_size).to(device)

        d1 = torch.zeros(n,batch_size).to(device)
        d0 = torch.zeros(n,batch_size).to(device)
        [B, C, W, H] = inputs.size()
        baryinputs = torch.zeros(B,C,W,H).to(device)

        if flag:
          p0 = torchvision.transforms.functional.rotate(inputs,4, transforms.InterpolationMode.NEAREST)
          p0 = F.interpolate(p0, size=W)

          p1 = torchvision.transforms.functional.rotate(inputs, -4, transforms.InterpolationMode.NEAREST)
          p1 = F.interpolate(p0, size=W)
        else:
          p0 = inputs
          p1=inputs_

        [B, C, W, H] = p0.size()
      
        orig =inputs[:,0,:,:].clone().view(W*H,B)
        rC0 = p0[:,0,:,:].clone().view(W*H,B)
        rC1 = p1[:,0,:,:].clone().view(W*H,B)
        rc = Barycenter(rC0,rC1,ADMM_opts, W, H,D_op, m1, m0, r1, r0, d1, d0, q,  orig )
        rC = rc.get_bary()

        orig =inputs[:,1,:,:].clone().view(W*H,B)
        rC0 = p0[:,1,:,:].clone().view(W*H,B)
        rC1 = p1[:,1,:,:].clone().view(W*H,B)
        gc = Barycenter(rC0,rC1,ADMM_opts, W, H,D_op, m1, m0, r1, r0, d1, d0, q, orig  )
        gC = gc.get_bary()

        orig =inputs[:,2,:,:].clone().view(W*H,B)
        rC0 = p0[:,2,:,:].clone().view(W*H,B)
        rC1 = p1[:,2,:,:].clone().view(W*H,B)
        bc = Barycenter(rC0,rC1,ADMM_opts, W, H,D_op, m1, m0, r1, r0, d1, d0, q, orig  )
        bC = bc.get_bary()

        baryinputs[:,0,:,:] = rC.clone().view(B,W,H)
        baryinputs[:,1,:,:] = gC.clone().view(B,W,H)
        baryinputs[:,2,:,:] = bC.clone().view(B,W,H)

        return baryinputs