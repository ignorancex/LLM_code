#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:33:27 2020

@author: kflouris
"""

from macros import jacobian, hessian_NEW, doublenabla

import torch
from torch import nn 
import sys

sys.path.append("torchdiffeq-master-amd")

from torchdiffeq import odeint
from torchdiffeq import odeint_amd
# from torchdiffeq import odeint
from macros import jacobian
import numpy as np

 #---------------------------------------------------------------------------------------------------------------------------------
# Method
#---------------------------------------------------------------------------------------------------------------------------------
def step_size_func(t, tau, alpha): 
     # return alpha* torch.exp(-2*t/tau)
    return alpha 

def time_steps_func(tau, steps, base=4): 
    # return (tau/base)*(np.logspace(0, 1,  num=steps, base=base)-1)
    return np.linspace(0,tau, num=steps)
 
class GFlow_approx:
    def __init__(self, net, indim, floss, tau=1.0, alpha=1.0, rTOL=1e-8, aTOL=1e-9, int_tsteps=200, method='rk4', device='cuda' ): 
        self.net = net.to(device)
        self.indim = indim
        self.device=device
        self.floss = floss
        self.x_data = None
        self.tau = tau
        self.alpha=alpha
        self.int_tsteps=int_tsteps
        self.z0 = torch.zeros((self.indim), requires_grad=True).to(self.device)
        self.ztau = None
        self.tlist = torch.from_numpy(time_steps_func(self.tau,self.int_tsteps ))        
        self.z_t = lambda  t, z0  : odeint(self.z_model, 
                                           z0, 
                                           torch.from_numpy(time_steps_func(t, self.int_tsteps )), 
                                           rtol=rTOL, atol=aTOL, method=method)
 
            
    def set_z0(self, z0=None, batch_size=None): 
        batch_size=self.x_data.shape[0]
        if  z0 is None:      z0_ = torch.zeros((self.x_data.shape[0], self.indim), requires_grad=True).to(self.device)
        elif z0 == 'det':    z0_ = self.determine_zs(self.x_data, 10000)
        elif z0 == 'update': z0_ = self.z0.repeat(batch_size, 1)
        else: z0_ = z0
        return z0_

    def determine_zs_allsteps(self, z0=None): 
        z0_ = self.set_z0(z0=z0, batch_size=self.x_data.shape[0])
        self.ztau = self.z_t(self.tau, z0_)
        return self.ztau
 
    def determine_zs(self, z0=None): 
        z0_ = self.set_z0(z0=z0, batch_size=self.x_data.shape[0])
        self.ztau = self.z_t(self.tau, z0_)[-1]
        return self.ztau
 
    def z_model(self, t, z): 
        zt = z.clone()
        x_model = self.net(zt)[0]  
        loss=self.floss(x_model, self.x_data)
        dLdz = -1.0 * step_size_func(t, self.tau,  self.alpha) *torch.autograd.grad(loss,zt,retain_graph=True)[0]    
        return dLdz   
    
class GFlow_AMD_approx:
    def __init__(self, net, indim, floss, amd_floss, tau=1.0, alpha=1.0, rTOL=1e-8, aTOL=1e-9, int_tsteps=200, 
                 dtstep=1, dtmaxstep=10, dtfactor=0.75, dtiter=50, conv_graddt=1e-9, conv_dtstep=100 ,conv_percentagetau=0.9, armijosigma=1e-5,
                 method='amd', device='cuda' ): 
        self.net = net.to(device)
        self.indim = indim
        self.device=device
        self.floss =floss
        self.amd_floss=amd_floss
        self.x_data = None
        self.tau = tau
        self.alpha=alpha
        self.int_tsteps=int_tsteps
        # self.z0=torch.ones((100, self.indim), requires_grad=True).to(self.device)
        # self.z0 = torch.zeros((self.indim), requires_grad=True).to(self.device)
        # self.ztau = None
        # self.tlist = torch.from_numpy(time_steps_func(self.tau,self.int_tsteps ))        
        self.inttlist = []
        self.zlist = []
        self.index = 0
        # options={_dtstep:dtstep, _dtmaxstep:dtmaxstep, _dtfactor:dtmaxstep, _dtiter:dtiter, _conv_graddt:conv_graddt
        #                                     ,_conv_percentagetau:conv_percentagetau}
        self.z_t = lambda  t, z0  : odeint_amd(self.z_model, self.amdloss,
                                           z0, 
                                           torch.from_numpy(time_steps_func(t, self.int_tsteps )),
                                           rtol=rTOL, atol=aTOL, method='amd',
                                           options={'dtstep':dtstep,'dtmaxstep':dtmaxstep,'dtfactor':dtfactor,'dtiter':dtiter,
                                                    'conv_graddt':conv_graddt,'conv_dtstep':conv_dtstep,'conv_percentagetau':conv_percentagetau,'tau':tau,
                                                    'armijosigma':armijosigma})
    def zero_lists(self): 
        self.inttlist = []
        self.zlist = []
            
    def set_z0(self, z0=None, batch_size=None): 
        batch_size=self.x_data.shape[0]
        if  z0 is None:      z0_ = torch.zeros((self.x_data.shape[0], self.indim), requires_grad=True).to(self.device)
        # elif z0 == 'det':    z0_ = self.determine_zs(10000)
        elif z0 == 'update': z0_ = self.z0.repeat(batch_size, 1)
        elif z0 == 'ones': z0_ = torch.ones((self.x_data.shape[0], self.indim), requires_grad=True).to(self.device)
        elif z0 == 'rand': z0_ = torch.rand((self.x_data.shape[0], self.indim), requires_grad=True).to(self.device)
        elif z0 == 'fill': z0_ = torch.ones((self.x_data.shape[0], self.indim), requires_grad=True).to(self.device)*0.5
        else: z0_ = z0
        return z0_
        
    def determine_zs(self, z0=None): 
        z0_ = self.set_z0(z0=z0, batch_size=self.x_data.shape[0])
        self.zt, _ = self.z_t(self.tau, z0_)
        return self.zt[-1]

    def determine_zs_allsteps(self, z0=None): 
        z0_ = self.set_z0(z0=z0, batch_size=self.x_data.shape[0])
        self.ztau, _ = self.z_t(self.tau, z0_)
        return self.zt[-1]
 
    # def z_model(self, t, z):
    #     print('t: ', t)
    #     zt = z.clone()
    #     x_model = self.net(zt)[0]
    #     loss=self.floss(x_model, self.x_data)
    #     #dLdz = -1.0 * step_size_func(t, self.tau, self.alpha) * torch.autograd.grad(loss,zt,retain_graph=True)[0]
    #     dLdz = -1.0 * step_size_func(t, self.tau, self.alpha) * torch.autograd.grad(loss,zt,retain_graph=True,only_inputs=True)[0]
    #     loss_av = 0.0
    #     print(loss.shape)

    #     for j in range(self.x_data.shape[0]):
    #         zt_ = z[j].clone()[None,:]
    #         loss_av += self.floss(self.net(zt_)[0], self.x_data[j][None,:])
    #         # print(loss_av / 5.0, loss)
    #         zt_ = z[j].clone()[None,:]
    #         x_model_ = self.net(zt_)[0]
    #         loss_ = self.floss(x_model_, self.x_data[j][None,:])
    #         dLdz_ = -1.0 * step_size_func(t, self.tau, self.alpha) * torch.autograd.grad(loss_,zt_,retain_graph=True,only_inputs=True)[0]
    #         # print('derivative: ', dLdz[j])
    #         # print('single derivative: ', dLdz_)
    #         # print('single derivative: ', dLdz_/self.x_data.shape[0])
    #         # print('diff: ', dLdz[j]-dLdz_/self.x_data.shape[0])
    #         print('diff: ', dLdz[j]-dLdz_)

    #         # print('single derivative loss: ', loss_.item())
            
    #         # print(dLdz.shape)
    #         zt = zt_ + dLdz_
    #         # print(self.floss(self.net(zt)[0], self.x_data[0][None,:]), loss_)
    #         _zt = zt + dLdz
    #         # print(self.floss(self.net(_zt)[0][0], self.x_data[0]))
    #     import sys
    #     sys.exit(1)
    #     #print('z_model gradient: ', dLdz[3], ' at ', t)
    #     return dLdz
 
    
    
    def z_model(self, t, z): 
        zt = z.clone()
        x_model = self.net(zt)[0]  
        loss=self.floss(x_model, self.x_data)
        dLdz = -1.0 * step_size_func(t, self.tau,  self.alpha) *torch.autograd.grad(loss,zt,retain_graph=True)[0]  
        # print(torch.mean(dLdz))
        # import sys
        # sys.exit()
        return dLdz  

    def amdloss(self, z): 
        zt = z.clone()
        x_model = self.net(zt)[0] 
        loss= self.amd_floss(x_model, self.x_data)
        return loss 
    
class GFlow_mom:
    def __init__(self, net, indim, floss, tau=1.0, alpha=1.0, rTOL=1e-8, aTOL=1e-9, int_tsteps=200, device='cuda', beta=0.9 ):
        
        self.net = net.to(device)
        self.indim = indim
        self.device=device
        self.floss = floss
        self.x_data = None
        self.tau = tau
        self.alpha=alpha
        self.int_tsteps=int_tsteps
        self.z0 = torch.zeros((self.indim), requires_grad=True).to(self.device)
        self.ztau = None
        self.tlist = torch.from_numpy(time_steps_func(self.tau,self.int_tsteps ))        
        self.inttlist = []
        self.zlist = []
        self.index = 0
        self.z_t = lambda  t, z0  : odeint(self.z_model, 
                                           z0, 
                                           torch.from_numpy(time_steps_func(t, self.int_tsteps )), 
                                           rtol=rTOL, atol=aTOL, method='rk4')
        self.beta = beta
    def forward(self, v): 
        return self.net(v)
    
    def set_z0(self, z0=None, batch_size=None): 
        batch_size=self.x_data.shape[0]
        v0_ = torch.zeros((batch_size, self.net.decoder[0].in_features), requires_grad=True).to(self.device)
        m0_ = torch.zeros((batch_size, self.net.decoder[0].in_features), requires_grad=True).to(self.device)
        z0_ = torch.stack([v0_, m0_], dim=0)
        return z0_
    
    def zero_lists(self): 
        self.inttlist = []
        self.zlist = []

    def determine_zs_allsteps(self, z0=None): 
        z0_ = self.set_z0(z0=z0, batch_size=self.x_data.shape[0])
        self.ztau = self.z_t(self.tau, z0_)
        return self.ztau
                    
    def determine_zs(self, z0=None): 
        z0_ = self.set_z0(z0=z0, batch_size=self.x_data.shape[0])
        self.ztau = self.z_t(self.tau, z0_)[-1]
        return self.ztau[0] #to choose the first part of the dldz as the other is to allow double integration
 
    def z_model(self, t, z): 
        # self.inttlist.append(t)
        # self.zlist.append(z.clone())
        vt = z[0].clone()
        mt = z[1].clone()
        x_model = self.net(vt)[0]   
        loss=self.floss(x_model, self.x_data)
        dmdt = -3/(t+1e-5)*mt - step_size_func(t, self.tau, self.alpha) * torch.autograd.grad(loss,vt,retain_graph=True)[0]    
        dvdt = mt
        dLdz = torch.stack([dvdt, dmdt], dim=0)
        return dLdz
    
class GFlow_AMD_mom:
    def __init__(self, net, indim, floss,amd_floss, tau=1.0, alpha=1.0, rTOL=1e-8, aTOL=1e-9, int_tsteps=200, 
                 dtstep=1, dtmaxstep=10, dtfactor=0.75, dtiter=50, conv_graddt=1e-9 ,conv_percentagetau=0.9, armijosigma=1e-5,
                 method='amd', device='cuda' ): 
        self.net = net.to(device)
        self.indim = indim
        self.device=device
        self.floss =floss
        self.amd_floss=amd_floss
        self.x_data = None
        self.tau = tau
        self.alpha=alpha
        self.int_tsteps=int_tsteps
        self.z0 = torch.zeros((self.indim), requires_grad=True).to(self.device)
        self.ztau = None
        self.tlist = torch.from_numpy(time_steps_func(self.tau,self.int_tsteps ))        
        self.inttlist = []
        self.zlist = []
        self.index = 0
        # options={_dtstep:dtstep, _dtmaxstep:dtmaxstep, _dtfactor:dtmaxstep, _dtiter:dtiter, _conv_graddt:conv_graddt
        #                                     ,_conv_percentagetau:conv_percentagetau}
        self.z_t = lambda  t, z0  : odeint_amd(self.z_model, self.amdloss,
                                           z0, 
                                           torch.from_numpy(time_steps_func(t, self.int_tsteps )),
                                           rtol=rTOL, atol=aTOL, method=method,
                                           options={'dtstep':dtstep,'dtmaxstep':dtmaxstep,'dtfactor':dtfactor,'dtiter':dtiter,
                                                    'conv_graddt':conv_graddt,'conv_percentagetau':conv_percentagetau,'tau':tau,
                                                    'armijosigma':armijosigma})
    def forward(self, v): 
        return self.net(v)
    
    def set_z0(self, z0=None, batch_size=None): 
        batch_size=self.x_data.shape[0]
        v0_ = torch.zeros((batch_size, self.net.decoder[0].in_features), requires_grad=True).to(self.device)
        m0_ = torch.zeros((batch_size, self.net.decoder[0].in_features), requires_grad=True).to(self.device)
        z0_ = torch.stack([v0_, m0_], dim=0)
        return z0_
    
    def zero_lists(self): 
        self.inttlist = []
        self.zlist = []
                    
    def determine_zs(self, z0=None): 
        z0_ = self.set_z0(z0=z0, batch_size=self.x_data.shape[0])
        self.ztau = self.z_t(self.tau, z0_)[-1]
        return self.ztau[0] #to choose the first part of the dldz as the other is to allow double integration
    
    def determine_zs_allsteps(self, z0=None): 
        z0_ = self.set_z0(z0=z0, batch_size=self.x_data.shape[0])
        self.ztau = self.z_t(self.tau, z0_)
        return self.ztau
    
    def z_model(self, t, z): 
        # self.inttlist.append(t)
        # self.zlist.append(z.clone())
        vt = z[0].clone()
        mt = z[1].clone()
        x_model = self.net(vt)[0]   
        loss=self.floss(x_model, self.x_data)
        dmdt = -3/(t+1e-5)*mt - step_size_func(t, self.tau, self.alpha) * torch.autograd.grad(loss,vt,retain_graph=True)[0]    
        dvdt = mt
        dLdz = torch.stack([dvdt, dmdt], dim=0)
        return dLdz
    
    def amdloss(self, z): 
        zt = z.clone()[0]
        x_model = self.net(zt)[0] 
        loss=self.amd_floss(x_model, self.x_data)
        return loss 
    
 
class AE_GFlow_approx:
    def __init__(self, net, indim, floss, tau=1.0, rTOL=1e-8, aTOL=1e-9, int_tsteps=200, device='cuda' ): 
        self.net = net.to(device)
        self.indim = indim
        self.device=device
        self.floss = floss
        self.x_data = None
        self.tau = tau
        self.int_tsteps=int_tsteps
        self.z0 = torch.zeros((self.indim), requires_grad=True).to(self.device)
        self.ztau = None
        self.tlist = torch.from_numpy(time_steps_func(self.tau,self.int_tsteps ))        
        self.inttlist = []
        self.zlist = []
        self.index = 0
        self.z_t = lambda  t, z0  : odeint(self.z_model, 
                                           z0, 
                                           torch.from_numpy(time_steps_func(t, self.int_tsteps )), 
                                           rtol=rTOL, atol=aTOL, method='rk4')
    def zero_lists(self): 
        self.inttlist = []
        self.zlist = []
            
    def set_z0(self, z0=None, batch_size=None): 
        batch_size=self.x_data.shape[0]
        if  z0 is None:      z0_ = torch.zeros((self.x_data.shape[0], self.indim), requires_grad=True).to(self.device)
        elif z0 == 'det':    z0_ = self.determine_zs(self.x_data, 10000)
        elif z0 == 'update': z0_ = self.z0.repeat(batch_size, 1)
        else: z0_ = z0
        return z0_
        
    def determine_zs(self, z0=None): 
        z0_ = self.set_z0(z0=z0, batch_size=self.x_data.shape[0])
        self.ztau = self.z_t(self.tau, z0_)[-1]
        return self.ztau
 
    def z_model(self, t, z): 
        zt = z.clone()
        x_model = self.net(zt)[0]  
        loss=self.floss(x_model, self.x_data)
        dLdz = -1.0 * step_size_func(t, self.tau) *torch.autograd.grad(loss,zt,retain_graph=True)[0]    
        return dLdz        
#================= Full ========================
class GFlow_adj:
    def __init__(self, net, indim, floss, tau=1.0, errdt=1, rTOL=1e-8, alpha=1.0, aTOL=1e-9, int_tsteps=200, device='cuda' ): 
        self.device=device
        self.net = net.to(self.device)
        self.indim = indim
        self.floss = floss
        self.x_data = None
        self.tau = tau
        self.alpha=alpha
        self.int_tsteps=int_tsteps
        self.z0 = torch.zeros((self.indim), requires_grad=True).to(self.device)
        self.ztau = None
        self.vtau = None
        self.phi = None
        self.losstau = None
        self.tlist = torch.from_numpy(time_steps_func(self.tau,self.int_tsteps )).to(self.device)        
        self.inttlist = []
        self.zlist = []
        self.index = 0
        self.errdt=errdt
        self.z_t = lambda  t, z0  : odeint(self.z_model, 
                                           z0, 
                                           # torch.from_numpy(np.linspace(0, t, self.err_tsteps)),
                                           torch.from_numpy(time_steps_func(t, self.int_tsteps )), 
                                           rtol=rTOL, atol=aTOL, method='rk4')
        
        self.v_t = lambda t, v0: odeint(self.v_model, v0, 
                                        t, 
                                        rtol=rTOL, atol=aTOL, method='rk4')
        
        # self.err_t = lambda  t, err0: odeint(self.err_model,  
        #                                       err0, 
        #                                       torch.tensor([0, t]).float(),  
        #                                       rtol=rTOL, atol=aTOL)[-1,:]      
        
        # self.err_t = lambda  t, err0: odeint(self.err_model,  
        #                                      err0, 
        #                                      torch.tensor([0, t]).float(),  
        #                                      rtol=rTOL, atol=aTOL, method='rk4')[-1,:]
         
        self.counter_z=0
        self.counter_error=0
        self.counter_adj=0    
    def set_z0(self, z0=None, batch_size=None): 
        if  z0 is None:      z0_ = torch.zeros((self.x_data.shape[0], self.indim), requires_grad=True).to(self.device)
        elif z0 == 'det':    z0_ = self.determine_zs(self.x_data, 10000)
        elif z0 == 'update': z0_ = self.z0.repeat(self.x_data.shape[0], 1)
        else: z0_ = z0
        return z0_


    def forward(self):
        self.zs = self.z_t(self.tau, torch.zeros((self.x_data.shape[0], self.indim), requires_grad=True).to(self.device) )
        self.ztau = self.zs[-1]
        self.inttlist = torch.Tensor(self.inttlist)
        x_model, log_px = self.net(self.ztau)
        self.losstau = self.floss( x_model , self.x_data)     
        self.vtau= jacobian(self.floss(x_model, self.x_data), self.ztau)
        self.index = self.inttlist.shape[0]
        vtlist = torch.flip(self.tlist,dims=[0])
        vs = self.v_t(vtlist, self.vtau) 
        self.zvt = torch.stack([self.zs, torch.flip(vs, dims=[0])], dim=1)
        self.dt = self.tlist[1]-self.tlist[0]

    def backward(self, z0=None): 
        DlDphi_list = []
        for i, self.phi in enumerate(self.net.parameters()):
            dldphi=jacobian(self.losstau, self.phi) 
            adjoint_model = self.err_model(self.tlist[1]-self.tlist[0], self.zvt).detach()
            full_derivative = -1.0*adjoint_model + dldphi
            DlDphi_list.append(full_derivative)
        # DlDphi_list.append(torch.sum(vs[-1], dim=0)) #ask ender 
        self.zero_lists()
        print('counter err={},adj={},z={}'.format( self.counter_error, self.counter_adj, self.counter_z))
        return DlDphi_list

    def backward_approx(self, z0=None): 
        DlDphi_list = []
        x_model, log_px = self.net(self.ztau)
        self.losstau = self.floss( x_model , self.x_data)     
        for i, self.phi in enumerate(self.net.parameters()):
            dldphi=jacobian(self.losstau, self.phi) 
            # adjoint_model = self.err_model(self.tlist[1]-self.tlist[0], self.zvt).detach()
            full_derivative = dldphi
            DlDphi_list.append(full_derivative)
        # DlDphi_list.append(torch.sum(vs[-1], dim=0)) #ask ender 
        self.zero_lists()
        return DlDphi_list
        
    def zero_lists(self): 
        self.inttlist = []
        self.zlist = []
        
    def determine_zs(self, z0=None): 
        z0_ = self.set_z0(z0=z0, batch_size=self.x_data.shape[0])
        self.ztau = self.z_t(self.tau, z0_)[-1]
        self.zero_lists()
        return self.ztau
 
    def z_model(self, t, z):
        self.counter_z+=1
        self.inttlist.append(t)
        self.zlist.append(z.clone())
        zt = z.clone()
        x_model, log_px = self.net(zt)    
        loss=self.floss(x_model ,self.x_data)
        dLdz = -1.0 * step_size_func(t,self.tau, self.alpha) * jacobian(loss,zt);    
        return dLdz
    
    def v_model(self, t, v):
        self.counter_adj+=1
        vt = v.clone()
        self.index = self.index-1
        zt = self.zlist[self.index]
        x_model, log_px = self.net(zt)    
        loss=self.floss(x_model,self.x_data)
        hess=hessian_NEW(loss*torch.ones(1,dtype=torch.float)[0], zt, vt)
        ddLdz = step_size_func(t,self.tau, self.alpha)*hess
        
        return ddLdz
    
    
    def err_model(self, dt, zvt):  
        dLdzdphi = torch.zeros_like(self.phi)
        self.counter_error+=1
        for j in range(0, zvt.shape[0], self.errdt):
            zt = zvt[j,0].clone()
            vt = zvt[j,1].clone()
            x_model, log_px = self.net(zt)    
            loss=self.floss(x_model ,self.x_data)
            ddLdzdphi = doublenabla(loss * torch.ones(1, dtype=torch.float)[0], 
                                       zt, 
                                       self.phi, 
                                       vt)
            dLdzdphi += step_size_func(self.tlist[j],self.tau, self.alpha) * ddLdzdphi * dt * self.errdt
        return dLdzdphi

class GFlow_AMD_adj:
    def __init__(self, net, indim, floss, amd_floss, errdt=4, tau=1.0, alpha=1.0, rTOL=1e-8, aTOL=1e-9, int_tsteps=200, 
                 dtstep=1, dtmaxstep=10, dtfactor=0.75, dtiter=50, conv_graddt=1e-9, conv_dtstep=100,conv_percentagetau=0.9, armijosigma=1e-5,
                 method='amd', device='cuda' ): 
        
        self.device=device
        self.errdt=errdt
        self.net = net.to(self.device)
        self.x_data = None
        self.indim = indim
        self.floss = floss
        self.amd_floss=amd_floss
        self.tau = tau
        self.int_tsteps=int_tsteps
        self.z0 = torch.zeros((self.indim), requires_grad=True).to(self.device)
        self.ztau = None
        self.vtau = None
        self.phi = None
        self.tlist = torch.from_numpy(time_steps_func(self.tau,self.int_tsteps )).to(self.device)        
        self.inttlist = []
        self.zlist = []
        self.index = 0
        self.alpha=1
        self.z_t = lambda  t, z0  : odeint_amd(self.z_model, self.amdloss,
                                           z0, 
                                           torch.from_numpy(time_steps_func(t, self.int_tsteps )),
                                           rtol=rTOL, atol=aTOL, method='amd',
                                           options={'dtstep':dtstep,'dtmaxstep':dtmaxstep,'dtfactor':dtfactor,'dtiter':dtiter,
                                                    'conv_graddt':conv_graddt,'conv_dtstep':conv_dtstep,'conv_percentagetau':conv_percentagetau,'tau':tau,
                                                    'armijosigma':armijosigma})
        
        self.v_t = lambda t, v0: odeint(self.v_model, v0, 
                                        t, 
                                        rtol=rTOL, atol=aTOL, method='rk4') 
        
        # self.err_t = lambda  t, err0: odeint(self.err_model,  
        #                                       err0, 
        #                                       torch.tensor([0, t]).float(),  
        #                                       rtol=rTOL, atol=aTOL)[-1,:]      
        
        # self.err_t = lambda  t, err0: odeint(self.err_model,  
        #                                      err0, 
        #                                      torch.tensor([0, t]).float(),  
        #                                      rtol=rTOL, atol=aTOL, method='rk4')[-1,:]
         
        self.counter_z=0
        self.counter_error=0
        self.counter_adj=0    
    def set_z0(self, z0=None, batch_size=None): 
        if  z0 is None:      z0_ = torch.zeros((self.x_data.shape[0], self.indim), requires_grad=True).to(self.device)
        elif z0 == 'det':    z0_ = self.determine_zs(self.x_data, 10000)
        elif z0 == 'update': z0_ = self.z0.repeat(self.x_data.shape[0], 1)
        else: z0_ = z0
        return z0_


    def forward(self):
        self.zs = self.z_t(self.tau, torch.zeros((self.x_data.shape[0], self.indim), requires_grad=True).to(self.device) )
        self.ztau = self.zs[-1]
        self.inttlist = torch.Tensor(self.inttlist)
        x_model, log_px = self.net(self.ztau)
        self.losstau = self.floss( x_model , self.x_data)     
        self.vtau= jacobian(self.floss(x_model, self.x_data), self.ztau)
        self.index = self.inttlist.shape[0]
        vtlist = torch.flip(self.tlist,dims=[0])
        vs = self.v_t(vtlist, self.vtau) 
        self.zvt = torch.stack([self.zs, torch.flip(vs, dims=[0])], dim=1)
        self.dt = self.tlist[1]-self.tlist[0]

    def backward(self, z0=None): 
        DlDphi_list = []
        for i, self.phi in enumerate(self.net.parameters()):
            dldphi=jacobian(self.losstau, self.phi) 
            adjoint_model = self.err_model(self.tlist[1]-self.tlist[0], self.zvt).detach()
            full_derivative = -1.0*adjoint_model + dldphi
            DlDphi_list.append(full_derivative)
        # DlDphi_list.append(torch.sum(vs[-1], dim=0)) #ask ender 
        self.zero_lists()
        print('counter err={},adj={},z={}'.format( self.counter_error, self.counter_adj, self.counter_z))
        return DlDphi_list

    def backward_approx(self, z0=None): 
        DlDphi_list = []
        x_model, log_px = self.net(self.ztau)
        self.losstau = self.floss( x_model , self.x_data)     
        for i, self.phi in enumerate(self.net.parameters()):
            dldphi=jacobian(self.losstau, self.phi) 
            # adjoint_model = self.err_model(self.tlist[1]-self.tlist[0], self.zvt).detach()
            full_derivative = dldphi
            DlDphi_list.append(full_derivative)
        # DlDphi_list.append(torch.sum(vs[-1], dim=0)) #ask ender 
        self.zero_lists()
        return DlDphi_list
        
    def zero_lists(self): 
        self.inttlist = []
        self.zlist = []
        
    def determine_zs(self, z0=None): 
        z0_ = self.set_z0(z0=z0, batch_size=self.x_data.shape[0])
        self.ztau = self.z_t(self.tau, z0_)[-1]
        self.zero_lists()
        return self.ztau
 
    def z_model(self, t, z):
        self.counter_z+=1
        self.inttlist.append(t)
        self.zlist.append(z.clone())
        zt = z.clone()
        x_model, log_px = self.net(zt)    
        loss=self.floss(x_model ,self.x_data)
        dLdz = -1.0 * step_size_func(t,self.tau, self.alpha) * jacobian(loss,zt);    
        return dLdz
    
    def v_model(self, t, v):
        self.counter_adj+=1
        vt = v.clone()
        self.index = self.index-1
        zt = self.zlist[self.index]
        x_model, log_px = self.net(zt)    
        loss=self.floss(x_model,self.x_data)
        hess=hessian_NEW(loss*torch.ones(1,dtype=torch.float)[0], zt, vt)
        ddLdz = step_size_func(t,self.tau, self.alpha)*hess
        
        return ddLdz
    
    
    def err_model(self, dt, zvt):  
        dLdzdphi = torch.zeros_like(self.phi)
        self.counter_error+=1
        for j in range(0, zvt.shape[0], self.errdt):
            zt = zvt[j,0].clone()
            vt = zvt[j,1].clone()
            x_model, log_px = self.net(zt)    
            loss=self.floss(x_model ,self.x_data)
            ddLdzdphi = doublenabla(loss * torch.ones(1, dtype=torch.float)[0], 
                                       zt, 
                                       self.phi, 
                                       vt)
            dLdzdphi += step_size_func(self.tlist[j],self.tau, self.alpha) * ddLdzdphi * dt * self.errdt
        return dLdzdphi

    def amdloss(self, z): 
        zt = z.clone()
        x_model = self.net(zt)[0] 
        loss= self.amd_floss(x_model, self.x_data)
        return loss 

class GFlow_gen_adj:
    def __init__(self, net, indim, floss, tau=1.0, rTOL=1e-8, aTOL=1e-9, int_tsteps=200, device='cuda' ): 
        self.device=device
        self.net = net.to(self.device)
        self.indim = indim
        self.floss = floss
        self.x_data = None
        self.tau = tau
        self.int_tsteps=int_tsteps
        self.z0 = torch.zeros((self.indim), requires_grad=True).to(self.device)
        self.ztau = None
        self.vtau = None
        self.phi = None
        self.tlist = torch.from_numpy(time_steps_func(self.tau,self.int_tsteps )).to(self.device)       
        self.inttlist = []
        self.zlist = []
        self.index = 0
        self.z_t = lambda  t, z0  : odeint(self.z_model, 
                                           z0, 
                                           # torch.from_numpy(np.linspace(0, t, self.err_tsteps)),
                                           torch.from_numpy(time_steps_func(t, self.int_tsteps )), 
                                           rtol=rTOL, atol=aTOL, method='rk4')
        
        self.v_t = lambda t, v0: odeint(self.v_model, v0, 
                                        t, 
                                        rtol=rTOL, atol=aTOL, method='rk4')
        
        # self.err_t = lambda  t, err0: odeint(self.err_model,  
        #                                       err0, 
        #                                       torch.tensor([0, t]).float(),  
        #                                       rtol=rTOL, atol=aTOL)[-1,:]      
        
        # self.err_t = lambda  t, err0: odeint(self.err_model,  
        #                                      err0, 
        #                                      torch.tensor([0, t]).float(),  
        #                                      rtol=rTOL, atol=aTOL, method='rk4')[-1,:]
           
    def set_z0(self, z0=None, batch_size=None): 
        if  z0 is None:      z0_ = torch.zeros((self.x_data.shape[0], self.indim), requires_grad=True).to(self.device)
        elif z0 == 'det':    z0_ = self.determine_zs(self.x_data, 10000)
        elif z0 == 'update': z0_ = self.z0.repeat(self.x_data.shape[0], 1)
        else: z0_ = z0
        return z0_

    def backward(self, x_data, z0=None): 
        self.x_data = x_data
        z0_ = self.set_z0(z0=z0, batch_size=x_data.shape[0])
        self.zs = self.z_t(self.tau, z0_)
        self.ztau = self.zs[-1]
        self.inttlist = torch.Tensor(self.inttlist)
        x_model, log_px = self.net(self.ztau)    
        self.vtau= (jacobian(self.floss(x_model, self.x_data), self.ztau)
                    -jacobian(log_px.mean(axis=0), self.ztau) ) # - logpx because it needs to be maxed, + the whole because delzL gets a - 
        losstau = self.floss( x_model , self.x_data)     
        self.index = self.inttlist.shape[0]
        vtlist = torch.flip(self.tlist,dims=[0])
        vs = self.v_t(vtlist, self.vtau) 
        zvt = torch.stack([self.zs, torch.flip(vs, dims=[0])], dim=1)
        DlDphi_list = []
        print('>>> ran the error backwards...computing the derivatives for the model parameters')
        for i, self.phi in enumerate(self.net.parameters()):
            dldphi=jacobian(losstau, self.phi) 
            adjoint_model = self.err_model(self.tlist[1]-self.tlist[0], zvt).detach()
            full_derivative = -1.0*adjoint_model + dldphi
            # print('--adjoint model - normal derivatives / normal derivatives ratio mean--')
            # print(torch.mean(torch.abs(adjoint_model)), torch.mean(torch.abs(dldphi)))
            if torch.isnan(torch.mean(torch.abs(adjoint_model))): 
                sys.exit(0)
            DlDphi_list.append(full_derivative)
        print('>>> ...computing the derivative for the initial condition')
        DlDphi_list.append(torch.sum(vs[-1], dim=0)) #ask ender 
        self.zero_lists()
        return DlDphi_list

      
    def zero_lists(self): 
        self.inttlist = []
        self.zlist = []
        
    def determine_zs(self, x_data, tau=None, z0=None): 
        self.x_data = x_data
        z0_ = self.set_z0(z0=z0, batch_size=x_data.shape[0])
        if tau is None: 
            self.ztau = self.z_t(self.tau, z0_)[-1]
        else: 
            self.ztau = self.z_t(tau, z0_)[-1]
        self.zero_lists()
        return self.ztau
 
    def z_model(self, t, z): 
        self.inttlist.append(t)
        self.zlist.append(z.clone())
        zt = z.clone()
        x_model, log_px = self.net(zt)    
        loss=self.floss(x_model ,self.x_data)
        #dLdz = -1.0*self.step_size * jacobian(loss,zt);    
        dLdz = -1.0 * step_size_func(t,self.tau,1) * jacobian(loss,zt);    
        return dLdz
    
    def v_model(self, t, v):
        vt = v.clone() # that is lambda
        self.index = self.index-1
        #zt = self.zs[torch.where(self.inttlist == t)[0][0]]
        zt = self.zlist[self.index]
        x_model, log_px = self.net(zt)    
        loss=self.floss(x_model ,self.x_data)
        hess=hessian_NEW(loss*torch.ones(1,dtype=torch.float)[0], zt, vt)
        ddLdz = step_size_func(t,self.tau,1)*hess
        return ddLdz
    
    
    def err_model(self, dt, zvt):  
        dLdzdphi = torch.zeros_like(self.phi)
        for j in range(0, zvt.shape[0], 4):
            zt = zvt[j,0].clone()
            vt = zvt[j,1].clone()
            x_model, log_px = self.net(zt)    
            loss=self.floss(x_model ,self.x_data)
            ddLdzdphi = doublenabla(loss * torch.ones(1, dtype=torch.float)[0], 
                                       zt, 
                                       self.phi, 
                                       vt)
            dLdzdphi += step_size_func(self.tlist[j],self.tau,1) * ddLdzdphi * dt * 4
        return dLdzdphi


#================= adjust tau ========================

# def adjust_tau_iteration(iteration): 
#     batch_size = x_data.shape[0]
#     gflnet.zero_lists()
#     zs = odeint(gflnet.z_model, 
#         gflnet.z0.repeat(x_data.shape[0], 1), 
#         torch.from_numpy(time_steps_func(tau,tsteps)), method='rk4')
#     buffer= int(tsteps*0.1) # 10% tsteps buffer for finding stable point
#     errors = []
#     for i, zsv in enumerate(zs): 
#         errors.append((loss_function(gflnet.net(zsv)[0], gflnet.x_data)/batch_size).detach().numpy())
#     d=np.gradient(errors)
#     increase=True 
#     index=tsteps
#     for i in np.arange(len(d)): 
#         if abs(np.mean(d[i:i+buffer])) < threshold:
#             index=i+int(buffer/2) 
#             break
#         elif abs(np.mean(d[i:i+buffer])) < threshold_safe: 
#             increase=False
#         else: index=tsteps
#     if increase: index*=1.2
#     return index


# # IN main training ================= Adaptive tua ===============
# if  ADAPT_TAU:             
#     new_TSTEPS=int(adjust_tau(gflow, x_data, ODE_TAU, ODE_TSTEPS, ODE_ADTAU_MIN, ODE_ADTAU_MAX,loss_function)) 
#     if new_TSTEPS!=ODE_TSTEPS:
#         new_TAU=ODE_TAU*(new_TSTEPS/ODE_TSTEPS)
#         print('>>> Checked convergence ...changing tau {}->{} and tsteps {}->{}'.format(ODE_TAU,new_TAU,ODE_TSTEPS,new_TSTEPS))
#         ODE_TAU=new_TAU
#         ODE_TSTEPS=new_TSTEPS
#         gflow = GFlow_approx(net, INPUT_DIM, loss_function,tau=ODE_TAU, rTOL=ODE_RTOL, int_tsteps=ODE_TSTEPS, device=device)        
#     else : print('>>> Checked convergence kept tau {} and tsteps {}'.format(ODE_TAU,ODE_TSTEPS))     


# def adjust_tau(gflnet, x_data, tau, tsteps, threshold, threshold_safe, loss_function): 
#     batch_size = x_data.shape[0]
#     gflnet.zero_lists()
#     zs = odeint(gflnet.z_model, 
#         gflnet.z0.repeat(x_data.shape[0], 1), 
#         torch.from_numpy(time_steps_func(tau,tsteps)), method='rk4')
#     buffer= int(tsteps*0.1) # 10% tsteps buffer for finding stable point
#     errors = []
#     for i, zsv in enumerate(zs): 
#         errors.append((loss_function(gflnet.net(zsv)[0], gflnet.x_data)/batch_size).detach().numpy())
#     d=np.gradient(errors)
#     increase=True 
#     index=tsteps
#     for i in np.arange(len(d)): 
#         if abs(np.mean(d[i:i+buffer])) < threshold:
#             index=i+int(buffer/2) 
#             break
#         elif abs(np.mean(d[i:i+buffer])) < threshold_safe: 
#             increase=False
#         else: index=tsteps
#     if increase: index*=1.2
#     return index 
#     # plt.plot( time_steps_func(15000,30), d, '-x'), 
#     # plt.plot( time_steps_func(15000,30), dd, '-x'), 
#     # plt.plot( time_steps_func(15000,30), ddd, '-x'), 
#     # ind = np.argpartition(abs(d), -4)[-4:]
#     # print('indexes****',ind)
#     # plt.savefig(name[:-4] + '_d.png')
#     # plt.close()
    
    
# class GFlow_autograd(torch.autograd.Function):
#   """
#   We can implement our own custom autograd Functions by subclassing
#   torch.autograd.Function and implementing the forward and backward passes
#   which operate on Tensors.
#   """
#   @staticmethod
#   def forward(self, x_model):
#     """
#     In the forward pass we receive a context object and a Tensor containing the
#     input; we must return a Tensor containing the output, and we can use the
#     context object to cache objects for use in the backward pass.
#     """
#     # #Calculate the forward value
#     # z0_ = GFlow.set_z0(z0=z0)
#     zs = GFlow.z_t(GFlow.tau)#, #z0_)
#     output=x_model(zs[-1])
#     #Cache tensors for backward pass
#     self.save_for_backward(zs)
    
#     # return output

#     # self.save_for_backward(input)
#     # return input

#   @staticmethod
#   def backward(self, grad_output):
#     """
#     In the backward pass we receive the context object and a Tensor containing
#     the gradient of the loss with respect to the output produced during the
#     forward pass. We can retrieve cached data from the context object, and must
#     compute and return the gradient of the loss with respect to the input to the
#     forward function.
#     """
#     # zs = self.saved_tensors
#     grad_phi = None

#     x_model_tau = grad_output.clone()
#     grad_phi=x_model_tau    

#     return grad_phi

#     # input, = self.saved_tensors
#     # grad_input = grad_output.clone()
#     # return grad_input
