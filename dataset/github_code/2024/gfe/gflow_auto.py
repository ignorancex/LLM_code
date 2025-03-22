 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:33:27 2020

@author: kflouris
"""

from macros import jacobian, hessian_NEW, doublenabla
from functools import partial
import torch
from torch import nn 
import sys
sys.path.append("torchdiffeq-master-amd")

from torchdiffeq import odeint
from torchdiffeq import odeint_amd
# from macros import jacobian
import numpy as np
from gflow import time_steps_func, step_size_func

# =================================== hooks ========================================
def my_partial(func, *args, **keywords):
    '''https://stackoverflow.com/questions/20594193/dynamically-created-method-and-decorator-got-error-functools-partial-object-h'''
    def newfunc(*fargs, **fkeywords):
        newkeywords = keywords.copy()
        newkeywords.update(fkeywords)
        return func (*(args + fargs), **newkeywords)
    newfunc.func = func
    newfunc.args = args
    newfunc.keywords = keywords
    return newfunc


class GFlow_AMD_adj_hooks:
    def __init__(self, net, indim, floss, amd_floss, approx_factor=4, tau=1.0, alpha=1.0, rTOL=1e-8, aTOL=1e-9, int_tsteps=200, 
                  dtstep=1, dtmaxstep=10, dtfactor=0.75, dtiter=50, conv_graddt=1e-9, conv_dtstep=100,conv_percentagetau=0.9, armijosigma=1e-5,
                  method='amd', device='cuda' ): 
        # TODO 
        # - remove clutter
        # - rename things properly
        # - make the error less dt euler?
        self.device=device
        self.approx_factor=approx_factor
        self.net = net.to(self.device)
        self.x_data = None
        self.indim = indim
        self.floss = floss
        self.amd_floss=amd_floss
        self.tau = tau
        self.z_t = lambda  t, z0  : odeint_amd(self.z_model, self.amdloss,
                                            z0, 
                                            torch.from_numpy(time_steps_func(t, 2 )),
                                            rtol=rTOL, atol=aTOL, method='amd',
                                            options={'dtstep':dtstep,'dtmaxstep':dtmaxstep,'dtfactor':dtfactor,'dtiter':dtiter,
                                                    'conv_graddt':conv_graddt,'conv_dtstep':conv_dtstep,'conv_percentagetau':conv_percentagetau,'tau':tau,
                                                    'armijosigma':armijosigma})
        
        self.v_t = lambda t, v0: odeint(self.v_model, v0, 
                                        t, 
                                        rtol=rTOL, atol=aTOL, method='euler')        
        self.add_hooks()
        
    def reinitialise(self): 
        self.ind_v = 0
        self.zs = self.tlist = self.ztau = self.vtau = self.vs  = None
        
    def forward(self):
        self.reinitialise()
        self.zs, self.tlist = self.z_t(self.tau, torch.zeros((self.x_data.shape[0], self.indim), requires_grad=True).to(self.device) )
        self.ztau = self.zs[-1]
        x_model, log_px = self.net(self.ztau)    
        self.vtau= jacobian(self.floss(x_model, self.x_data), self.ztau)
        self.ind_v = self.tlist.shape[0]
        vtlist = torch.flip(self.tlist,dims=[0])
        self.vs = self.v_t(vtlist, self.vtau) 
           
    def determine_zs(self): 
        self.reinitialise()
        z0 = torch.zeros((self.x_data.shape[0], self.indim), requires_grad=True).to(self.device)    
        self.zt, _ = self.z_t(self.tau, z0)
        return self.zt[-1]

    def z_model(self, t, zt): 
        x_model = self.net(zt)[0]    
        loss=self.floss(x_model,self.x_data)
        dLdz = -1.0 * jacobian(loss,zt);
        return dLdz
        # del zt, loss, x_model, dLdz
    
    def v_model(self, t, vt):
        self.ind_v = self.ind_v-1
        zt = self.zs[self.ind_v]
        x_model = self.net(zt)[0]    
        loss=self.floss(x_model,self.x_data)
        hess=hessian_NEW(loss*torch.ones(1,dtype=torch.float)[0], zt, vt)
        ddLdz =1 * hess
        return ddLdz
        # del zt, loss, x_model, hess, ddLdz
    
    def adjoint_model_by_name(self, grad, paramname='dummy'):
        phi = [param for testname, param in self.net.named_parameters() if testname == paramname][0]
        # print(paramname)
        dLdzdphi = torch.zeros_like(phi)
        lastt=self.tlist[-1].data+(self.tlist[-1].data-self.tlist[-2].data)
        extented_tlist = torch.cat((self.tlist,torch.tensor([lastt]).view(1), )) # adds the save value to the end
        for j in range(0, self.tlist.shape[0], self.approx_factor):
            dt=extented_tlist[j+1]-extented_tlist[j]
            zt = self.zs[j]#.clone()
            vt = self.vs[j]#.clone()
            x_model, log_px = self.net(zt)    
            loss=self.floss(x_model, self.x_data)
            dldz= torch.autograd.grad(loss, zt, create_graph=True)[0] # retain_graph=True,
            # dldz= jacobian(loss, zt,  create_graph=True)
            vdldz=torch.einsum('ij, ij->i', vt, dldz) 
            vdldzdphi=jacobian(vdldz, phi, create_graph=True).sum(dim=0).detach()
            dLdzdphi += 1 * vdldzdphi * dt * self.approx_factor
                    
            dldz.detach()
            del dldz, vdldz, vdldzdphi, zt, vt, x_model, log_px, loss, dt
        
        output = grad - dLdzdphi
        return output
        del output, dLdzdphi, extented_tlist, phi

    def add_hooks(self):
        print('\n')
        with torch.no_grad():
            for name, param in self.net.named_parameters():
                print(' attaching hook on:', name)            
                param.register_hook(my_partial(self.adjoint_model_by_name, paramname=name))
                
    def amdloss(self, zt): 
        x_model = self.net(zt)[0] 
        loss= self.amd_floss(x_model, self.x_data)
        return loss 

# class GFlow_AMD_adj_hooks_all_euler_NOT_WORKING:
#     def __init__(self, net, indim, floss, amd_floss, reduction_dt_factor=4, tau=1.0, alpha=1.0, rTOL=1e-8, aTOL=1e-9, int_tsteps=200, 
#                  dtstep=1, dtmaxstep=10, dtfactor=0.75, dtiter=50, conv_graddt=1e-9, conv_dtstep=100,conv_percentagetau=0.9, armijosigma=1e-5,
#                  method='amd', device='cuda' ): 
#         # TODO 
#         # - remove clutter
#         # - rename things properly
#         # - make the error less dt euler?
#         self.device=device
#         self.reduction_dt_factor=reduction_dt_factor
#         self.net = net.to(self.device)
#         self.x_data = None
#         self.indim = indim
#         self.floss = floss
#         self.amd_floss=amd_floss
#         self.tau = tau
#         # self.tlist = torch.from_numpy(time_steps_func(self.tau,self.int_tsteps )).to(self.device)        
#         self.z_t = lambda  t, z0  : odeint_amd(self.z_model, self.amdloss,
#                                            z0, 
#                                            torch.from_numpy(time_steps_func(t, 2 )),
#                                            rtol=rTOL, atol=aTOL, method='amd',
#                                            options={'dtstep':dtstep,'dtmaxstep':dtmaxstep,'dtfactor':dtfactor,'dtiter':dtiter,
#                                                     'conv_graddt':conv_graddt,'conv_dtstep':conv_dtstep,'conv_percentagetau':conv_percentagetau,'tau':tau,
#                                                     'armijosigma':armijosigma})
        
#         self.v_t = lambda t, v0: odeint(self.v_model, v0, 
#                                         t, 
#                                         rtol=rTOL, atol=aTOL, method='euler')    
#         self.dLdzdphi_t = lambda t, dLdzdphi0: odeint(self.vdldzdphi_model, dLdzdphi0, 
#                                         t, 
#                                         rtol=rTOL, atol=aTOL, method='euler')    
#         self.add_hooks()
#         self.zero_lists()
        
#     def zero_lists(self): 
#         self.inttlist = []
#         self.zlist = []
#         self.ind_v = 0
#         self.ind_vdldzdphi= 0   
        
#     def forward(self):
#         self.zs = self.z_t(self.tau, torch.zeros((self.x_data.shape[0], self.indim), requires_grad=True).to(self.device) )
#         self.ztau = self.zs[-1]
#         # print(len(self.zlist))
#         #to get all the steps
#         self.inttlist = torch.Tensor(self.inttlist)
#         x_model, log_px = self.net(self.ztau)    
#         self.vtau= jacobian(self.floss(x_model, self.x_data), self.ztau)
#         self.ind_v = self.inttlist.shape[0]
#         # vtlist = torch.flip(self.tlist,dims=[0])
#         vtlist = torch.flip(self.inttlist,dims=[0])
#         vs = self.v_t(vtlist, self.vtau) 
#         zs=torch.stack(self.zlist)
#         self.zvt = torch.stack([zs, torch.flip(vs, dims=[0])], dim=1)
           

        
#     def determine_zs(self): 
#         z0 = torch.zeros((self.x_data.shape[0], self.indim), requires_grad=True).to(self.device)    
#         self.ztau = self.z_t(self.tau, z0)[-1]
#         self.zero_lists()
#         return self.ztau
 
#     def z_model(self, t, z): 
#         self.inttlist.append(t)
#         self.zlist.append(z.clone())
#         zt = z.clone()
#         x_model = self.net(zt)[0]    
#         loss=self.floss(x_model,self.x_data)
#         dLdz = -1.0 * jacobian(loss,zt);    
#         return dLdz
    
#     def v_model(self, t, v):
#         vt = v.clone() # that is lambda
#         self.ind_v -= 1
#         zt = self.zlist[self.ind_v]
#         x_model = self.net(zt)[0]    
#         loss=self.floss(x_model,self.x_data)
#         hess=hessian_NEW(loss*torch.ones(1,dtype=torch.float)[0], zt, vt)
#         ddLdz =1 * hess
        
#         return ddLdz

#     def vdldzdphi_model(self, t, vdldzdphi):
#             # we need an index here , we 
#             zt = self.zvt[self.ind_vdldzdphi,0].clone()
#             vt = self.zvt[self.ind_vdldzdphi,1].clone()
#             self.ind_vdldzdphi +=1 

#             x_model, log_px = self.net(zt)    
#             loss=self.floss(x_model, self.x_data)
#             # dldz= torch.autograd.grad(loss, zt, create_graph=True)[0] # retain_graph=True,
#             dldz= jacobian(loss, zt,  create_graph=True)
#             vdldz=torch.einsum('ij, ij->i', vt, dldz) 
#             vdldzdphi=jacobian(vdldz, self.phi, create_graph=True).sum(dim=0) #.detach()
#             return vdldzdphi                    
#             del dldz, vdldz, vdldzdphi, zt, vt, x_model, log_px, loss    

#     def adjoint_model_by_name(self, grad, paramname='dummy'):
#         self.phi = [param for testname, param in self.net.named_parameters() if testname == paramname][0]
#         # print(paramname)
#         dLdzdphi_0 = torch.zeros_like(self.phi)
#         # # print(self.zvt.shape[0])
#         # dt=1
#         # for j in range(0, self.zvt.shape[0], self.errdt):
#         #     #TODO ask about cloning
#         #     zt = self.zvt[j,0].clone()
#         #     vt = self.zvt[j,1].clone()
    
#         #     x_model, log_px = self.net(zt)    
#         #     loss=self.floss(x_model, self.x_data)
#         #     # dldz= torch.autograd.grad(loss, zt, create_graph=True)[0] # retain_graph=True,
#         #     dldz= jacobian(loss, zt,  create_graph=True)
#         #     vdldz=torch.einsum('ij, ij->i', vt, dldz) 
#         #     vdldzdphi=jacobian(vdldz, phi, create_graph=True).sum(dim=0) #.detach()
#         #     dLdzdphi += 1 * vdldzdphi * dt * self.errdt
#         dLdzdphi_s = self.dLdzdphi_t(self.inttlist, dLdzdphi_0) 
#         self.zero_lists()


            
#         output = grad - dLdzdphi_s
#         return output

#     def add_hooks(self):
#         for name, param in self.net.named_parameters():
#             print('defname', name)            
#             param.register_hook(my_partial(self.adjoint_model_by_name, paramname=name))
            
#     def amdloss(self, z): 
#         zt = z.clone()
#         x_model = self.net(zt)[0] 
#         loss= self.amd_floss(x_model, self.x_data)
#         return loss 

# class GFlow_AMD_adj_hooks:
#     def __init__(self, net, indim, floss, amd_floss, errdt=4, tau=1.0, alpha=1.0, rTOL=1e-8, aTOL=1e-9, int_tsteps=200, 
#                  dtstep=1, dtmaxstep=10, dtfactor=0.75, dtiter=50, conv_graddt=1e-9, conv_dtstep=100,conv_percentagetau=0.9, armijosigma=1e-5,
#                  method='amd', device='cuda' ): 
#         # TODO 
#         # - remove clutter
#         # - rename things properly
#         # - make the error less dt euler?
#         self.device=device
#         self.errdt=errdt
#         self.net = net.to(self.device)
#         self.x_data = None
#         self.indim = indim
#         self.floss = floss
#         self.amd_floss=amd_floss
#         self.tau = tau
#         self.int_tsteps=int_tsteps
#         self.z0 = torch.zeros((self.indim), requires_grad=True).to(self.device)
#         self.ztau = None
#         self.vtau = None
#         self.phi = None
#         # self.tlist = torch.from_numpy(time_steps_func(self.tau,self.int_tsteps )).to(self.device)        
#         self.inttlist = []
#         self.zlist = []
#         self.index = 0
#         self.z_t = lambda  t, z0  : odeint_amd(self.z_model, self.amdloss,
#                                            z0, 
#                                            torch.from_numpy(time_steps_func(t, 2 )),
#                                            rtol=rTOL, atol=aTOL, method='amd',
#                                            options={'dtstep':dtstep,'dtmaxstep':dtmaxstep,'dtfactor':dtfactor,'dtiter':dtiter,
#                                                     'conv_graddt':conv_graddt,'conv_dtstep':conv_dtstep,'conv_percentagetau':conv_percentagetau,'tau':tau,
#                                                     'armijosigma':armijosigma})
        
#         self.v_t = lambda t, v0: odeint(self.v_model, v0, 
#                                         t, 
#                                         rtol=rTOL, atol=aTOL, method='euler')        
#         self.add_hooks()
        
#     def forward(self):
#         self.zs = self.z_t(self.tau, torch.zeros((self.x_data.shape[0], self.indim), requires_grad=True).to(self.device) )
#         self.ztau = self.zs[-1]
#         # print(len(self.zlist))
#         #to get all the steps
#         self.inttlist = torch.Tensor(self.inttlist)
#         x_model, log_px = self.net(self.ztau)    
#         self.vtau= jacobian(self.floss(x_model, self.x_data), self.ztau)
#         self.index = self.inttlist.shape[0]
#         # vtlist = torch.flip(self.tlist,dims=[0])
#         vtlist = torch.flip(self.inttlist,dims=[0])
#         vs = self.v_t(vtlist, self.vtau) 
#         zs=torch.stack(self.zlist)
#         self.zvt = torch.stack([zs, torch.flip(vs, dims=[0])], dim=1)
           
#     def zero_lists(self): 
#         self.inttlist = []
#         self.zlist = []
        
#     def determine_zs(self): 
#         z0 = torch.zeros((self.x_data.shape[0], self.indim), requires_grad=True).to(self.device)    
#         self.ztau = self.z_t(self.tau, z0)[-1]
#         self.zero_lists()
#         return self.ztau
 
#     def z_model(self, t, z): 
#         self.inttlist.append(t)
#         self.zlist.append(z.clone())
#         #TODO
#         zt = z#.clone()
#         x_model = self.net(zt)[0]    
#         loss=self.floss(x_model,self.x_data)
#         dLdz = -1.0 * jacobian(loss,zt);    
#         return dLdz
    
#     def v_model(self, t, v):
#         #TODO
#         vt = v#.clone() # that is lambda
#         self.index = self.index-1
#         zt = self.zlist[self.index]
#         x_model = self.net(zt)[0]    
#         loss=self.floss(x_model,self.x_data)
#         hess=hessian_NEW(loss*torch.ones(1,dtype=torch.float)[0], zt, vt)
#         ddLdz =1 * hess
        
#         return ddLdz
# # **************************** can also make this euler with less timesteps?
#     def adjoint_model_by_name(self, grad, paramname='dummy'):
#         phi = [param for testname, param in self.net.named_parameters() if testname == paramname][0]
#         # print(paramname)
#         dLdzdphi = torch.zeros_like(phi)
#         # print(self.zvt.shape[0])
#         dt=1
#         for j in range(0, self.zvt.shape[0], self.errdt):
#             #TODO ask about cloning
#             zt = self.zvt[j,0].clone()
#             vt = self.zvt[j,1].clone()
    
#             x_model, log_px = self.net(zt)    
#             loss=self.floss(x_model, self.x_data)
#             # dldz= torch.autograd.grad(loss, zt, create_graph=True)[0] # retain_graph=True,
#             dldz= jacobian(loss, zt,  create_graph=True)
#             vdldz=torch.einsum('ij, ij->i', vt, dldz) 
#             vdldzdphi=jacobian(vdldz, phi, create_graph=True).sum(dim=0) #.detach()
#             dLdzdphi += 1 * vdldzdphi * dt * self.errdt
                    
#             del dldz, vdldz, vdldzdphi, zt, vt, x_model, log_px, loss
#             self.zero_lists()


            
#         output = grad - dLdzdphi
#         return output

#     def add_hooks(self):
#         for name, param in self.net.named_parameters():
#             print('defname', name)            
#             param.register_hook(my_partial(self.adjoint_model_by_name, paramname=name))
            
#     def amdloss(self, z): 
#         zt = z.clone()
#         x_model = self.net(zt)[0] 
#         loss= self.amd_floss(x_model, self.x_data)
#         return loss 

class GFlow_adj_hooks:
    def __init__(self,net, indim, floss, errdt=4, tau=1.0, rTOL=1e-8, aTOL=1e-9, int_tsteps=200, device='cuda' ): 
        self.device=device
        self.errdt=errdt
        self.net = net.to(self.device)
        self.x_data = None
        self.indim = indim
        self.floss = floss
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
        
        self.add_hooks()
        
    def forward(self):
        self.zs = self.z_t(self.tau, torch.zeros((self.x_data.shape[0], self.indim), requires_grad=True).to(self.device) )
        self.ztau = self.zs[-1]
        self.inttlist = torch.Tensor(self.inttlist)
        x_model, log_px = self.net(self.ztau)    
        self.vtau= jacobian(self.floss(x_model, self.x_data), self.ztau)
        self.index = self.inttlist.shape[0]
        vtlist = torch.flip(self.tlist,dims=[0])
        vs = self.v_t(vtlist, self.vtau) 
        self.zvt = torch.stack([self.zs, torch.flip(vs, dims=[0])], dim=1)
        self.dt = self.tlist[1]-self.tlist[0]
           
    def zero_lists(self): 
        self.inttlist = []
        self.zlist = []
        
    def determine_zs(self): 
        z0 = torch.zeros((self.x_data.shape[0], self.indim), requires_grad=True).to(self.device)    
        self.ztau = self.z_t(self.tau, z0)[-1]
        self.zero_lists()
        return self.ztau
 
    def z_model(self, t, z): 
        self.inttlist.append(t)
        self.zlist.append(z.clone())
        #TODO
        zt = z#.clone()
        x_model = self.net(zt)[0]    
        loss=self.floss(x_model,self.x_data)
        dLdz = -1.0 * step_size_func(t,self.tau,1) * jacobian(loss,zt);    
        return dLdz
    
    def v_model(self, t, v):
        #TODO
        vt = v#.clone() # that is lambda
        self.index = self.index-1
        zt = self.zlist[self.index]
        x_model = self.net(zt)[0]    
        loss=self.floss(x_model,self.x_data)
        hess=hessian_NEW(loss*torch.ones(1,dtype=torch.float)[0], zt, vt)
        ddLdz = step_size_func(t,self.tau,1)*hess
        
        return ddLdz
    
    # def backward(self, z0=None): 
    #     DlDphi_list = []
    #     for i, self.phi in enumerate(self.net.parameters()):
    #         dldphi=jacobian(self.losstau, self.phi) 
    #         adjoint_model = self.err_model(self.tlist[1]-self.tlist[0], self.zvt).detach()
    #         full_derivative = -1.0*adjoint_model + dldphi
    #         DlDphi_list.append(full_derivative)
    #     # DlDphi_list.append(torch.sum(vs[-1], dim=0)) #ask ender 
    #     self.zero_lists()
    #     print('counter err={},adj={},z={}'.format( self.counter_error, self.counter_adj, self.counter_z))
    #     return DlDphi_list
        


    def adjoint_model_by_name(self, grad, paramname='dummy'):
        phi = [param for testname, param in self.net.named_parameters() if testname == paramname][0]
        # print(paramname)
        dLdzdphi = torch.zeros_like(phi)
        # print(self.zvt.shape[0])
        for j in range(0, self.zvt.shape[0], self.errdt):
            #TODO ask about cloning
            zt = self.zvt[j,0].clone()
            vt = self.zvt[j,1].clone()
            x_model, log_px = self.net(zt)    
            loss=self.floss(x_model, self.x_data)
            dldz= torch.autograd.grad(loss, zt, create_graph=True)[0] # retain_graph=True,
            # dldz= jacobian(loss, zt,  create_graph=True)
            vdldz=torch.einsum('ij, ij->i', vt, dldz) 
            vdldzdphi=jacobian(vdldz, phi, create_graph=True).sum(dim=0).detach()
            dLdzdphi += step_size_func(self.tlist[j],self.tau,1) * vdldzdphi * self.dt * self.errdt
                    
            del dldz, vdldz, vdldzdphi, zt, vt, x_model, log_px, loss
            self.zero_lists()


            
        output = grad - dLdzdphi
        # phi.grad=None
        del dLdzdphi, grad, phi
        return output

    def add_hooks(self):
        for name, param in self.net.named_parameters():
            print('defname', name)            
            param.register_hook(my_partial(self.adjoint_model_by_name, paramname=name))



# =================================== Autograd ========================================
class Autograd_static(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, net, indim, floss, tau, rTOL, aTOL, int_tsteps, device, xdata):
        """
        In the forward pass we receive a context object and a Tensor containing the
        input; we must return a Tensor containing the output, and we can use the
        context object to cache objects for use in the backward pass.
        """

        def time_steps_func(tau, steps, base=6): 
            return (tau/base)*(np.logspace(0, 1,  num=steps, base=base)-1)
            # return np.linspace(0,tau, num=steps)  

        def step_size_func(t, tau): 
            return 1.0 * torch.exp(-2*t/tau)
            # return 1.0 
            
        ctx.device=device
        ctx.net = net.to(ctx.device)
        ctx.indim = indim
        ctx.floss = floss
        ctx.x_data = xdata
        ctx.tau = tau
        ctx.int_tsteps=int_tsteps
        ctx.z0 = torch.zeros((ctx.indim), requires_grad=True).to(ctx.device)
        ctx.ztau = None
        ctx.vtau = None
        ctx.phi = None
        ctx.tlist = torch.from_numpy(time_steps_func(ctx.tau,ctx.int_tsteps )).to(ctx.device)        
        ctx.inttlist = []
        ctx.zlist = []
        ctx.index = 0
        ctx.z_t = lambda  t, z0  : odeint(z_model, 
                                            z0, 
                                            # torch.from_numpy(np.linspace(0, t, ctx.err_tsteps)),
                                            torch.from_numpy(time_steps_func(t, ctx.int_tsteps )), 
                                            rtol=rTOL, atol=aTOL, method='rk4')
 
        # def z_t(t,z0): return     odeint(z_model, 
        #                                    z0, 
        #                                    # torch.from_numpy(np.linspace(0, t, ctx.err_tsteps)),
        #                                    torch.from_numpy(time_steps_func(t, ctx.int_tsteps )), 
        #                                    rtol=rTOL, atol=aTOL, method='rk4')        
        
        ctx.v_t = lambda t, v0: odeint(ctx.v_model, v0, 
                                        t, 
                                        rtol=rTOL, atol=aTOL, method='rk4')

            
        def set_z0( z0=None, batch_size=None): 
            if  z0 is None:      z0_ = torch.zeros((ctx.x_data.shape[0], ctx.indim), requires_grad=True).to(ctx.device)
            elif z0 == 'det':    z0_ = determine_zs(ctx.x_data, 10000)
            elif z0 == 'update': z0_ = ctx.z0.repeat(ctx.x_data.shape[0], 1)
            else: z0_ = z0
            return z0_
        
        def backward_G( x_data, z0=None): 
            ctx.x_data = x_data
            z0_ = ctx.set_z0(z0=z0, batch_size=x_data.shape[0])
            ctx.zs = ctx.z_t(ctx.tau, z0_)
            ctx.ztau = ctx.zs[-1]
            ctx.inttlist = torch.Tensor(ctx.inttlist)
            x_model, log_px = ctx.net(ctx.ztau)    
            ctx.vtau= jacobian(ctx.floss(x_model, ctx.x_data), ctx.ztau)
            losstau = ctx.floss( x_model , ctx.x_data)     
            ctx.index = ctx.inttlist.shape[0]
            vtlist = torch.flip(ctx.tlist,dims=[0])
            vs = ctx.v_t(vtlist, ctx.vtau) 
            zvt = torch.stack([ctx.zs, torch.flip(vs, dims=[0])], dim=1)
            DlDphi_list = []
            print('>>> ran the error backwards...computing the derivatives for the model parameters')
            for i, ctx.phi in enumerate(ctx.net.parameters()):
                dldphi=jacobian(losstau, ctx.phi) 
                adjoint_model = ctx.err_model(ctx.tlist[1]-ctx.tlist[0], zvt).detach()
                full_derivative = -1.0*adjoint_model + dldphi
                # print('--adjoint model - normal derivatives / normal derivatives ratio mean--')
                # print(torch.mean(torch.abs(adjoint_model)), torch.mean(torch.abs(dldphi)))
                if torch.isnan(torch.mean(torch.abs(adjoint_model))): 
                    sys.exit(0)
                DlDphi_list.append(full_derivative)
            print('>>> ...computing the derivative for the initial condition')
            DlDphi_list.append(torch.sum(vs[-1], dim=0)) #ask ender 
            ctx.zero_lists()
            
            # DlDphi= torch.cat(DlDphi_list, dim=1) 
            return DlDphi_list
        
        # def backward_approx(self, x_data, z0=None): 
        #     self.x_data = x_data
        #     z0_ = self.set_z0(z0=z0)
        #     self.zs = self.z_t(self.tau, z0_)
        #     self.ztau = self.zs[-1]
        #     self.inttlist = torch.Tensor(self.inttlist)
        #     self.vtau= 1.0*jacobian(self.floss(self.net(self.ztau)[0], self.x_data), self.ztau)
        #     losstau = self.floss( self.net(self.ztau)[0], self.x_data)     
        #     DlDphi_list = []
        #     # print('>>> ran the error backwards...ignoring the adjoint part')
        #     for i, self.phi in enumerate(self.net.parameters()):
        #         dldphi=jacobian(losstau, self.phi) 
        #         full_derivative = dldphi
        #         DlDphi_list.append(full_derivative)
        #     DlDphi_list.append(torch.sum(self.vtau, dim=0)) #ask ender 
        #     self.zero_lists()
        #     return DlDphi_list
           
        def zero_lists(ctx): 
            ctx.inttlist = []
            ctx.zlist = []
            
        def determine_zs( x_data, tau, z0=None): 
            ctx.x_data = x_data
            z0_ = set_z0(z0=z0, batch_size=x_data.shape[0])
            # ctx.ztau = ctx.z_t(tau, z0_)[-1]
            ctx.ztau =ctx.z_t(tau, z0_)[-1]
            zero_lists()
            return ctx.ztau
         
        def z_model( t, z): 
            ctx.inttlist.append(t)
            ctx.zlist.append(z)
            zt = z
            # zt.requires_grad=True
            x_model, log_px = ctx.net(z)    
            loss=floss(x_model ,ctx.x_data)
            # dLdz = -1.0 * step_size_func(t,ctx.tau) * jacobian(loss,zt);    
            dLdz = -1.0 * step_size_func(t,ctx.tau) * torch.autograd.grad(loss,zt, retain_graph=True, create_graph=True);    
            return dLdz
        
        def v_model( t, v):
            vt = v.clone() # that is lambda
            ctx.index = ctx.index-1
            #zt = ctx.zs[torch.where(ctx.inttlist == t)[0][0]]
            zt = ctx.zlist[ctx.index]
            x_model, log_px = ctx.net(zt)    
            loss=ctx.floss(x_model,ctx.x_data)
            hess=hessian_NEW(loss*torch.ones(1,dtype=torch.float)[0], zt, vt)
            ddLdz = step_size_func(t,ctx.tau)*hess
            
            return ddLdz
        
        
        def err_model( dt, zvt):  
            dLdzdphi = torch.zeros_like(ctx.phi)
            for j in range(0, zvt.shape[0], 4):
                zt = zvt[j,0].clone()
                vt = zvt[j,1].clone()
                x_model, log_px = ctx.net(zt)    
                loss=ctx.floss(x_model ,ctx.x_data)
                ddLdzdphi = doublensabla(loss * torch.ones(1, dtype=torch.float)[0], 
                                            zt, 
                                            ctx.phi, 
                                            vt)
                dLdzdphi += step_size_func(ctx.tlist[j],ctx.tau) * ddLdzdphi * dt * 4
            return dLdzdphi
        
        z0_ = torch.zeros((ctx.x_data.shape[0], ctx.indim), requires_grad=True).to(ctx.device)  
        z0_2=set_z0()    
        x,p=ctx.net(z0_2)             
        zs = determine_zs(ctx.x_data, ctx.tau, z0=None)  
        x_model=net(zs)     
        # dlist = gflow.backward(x_data, z0=None)
        ctx.save_for_backward(x_data)
        return x_model
    
    # @oncedifferentiable          
    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive the context object and a Tensor containing
        the gradient of the loss with respect to the output produced during the
        forward pass. We can retrieve cached data from the context object, and must
        compute and return the gradient of the loss with respect to the input to the
        forward function.
        """
        gradients = grad_output.clone() #this would be the gradients of x_model i.e. the return of forward
        zs, stupid,x_data = ctx.saved_tensors
        
        gradients=gflow.backward_G(x_data, z0=None)
        # grad_input = grad_output.clone()
        return gradients



# class GFlow_autograd:
#     """
#     We can implement our own custom autograd Functions by subclassing
#     torch.autograd.Function and implementing the forward and backward passes
#     which operate on Tensors.
#     """
#     def __init__(self, net, indim, floss, tau=1.0, rTOL=1e-8, aTOL=1e-9, int_tsteps=200, device='cuda' ): 
#         self.device=device
#         self.net = net.to(self.device)
#         self.indim = indim
#         self.floss = floss
#         self.x_data = None
#         self.tau = tau
#         self.int_tsteps=int_tsteps
#         self.z0 = torch.zeros((self.indim), requires_grad=True).to(self.device)
#         self.ztau = None
#         self.vtau = None
#         self.phi = None
#         self.tlist = torch.from_numpy(time_steps_func(self.tau,self.int_tsteps )).to(self.device)        
#         self.inttlist = []
#         self.zlist = []
#         self.index = 0
#         self.z_t = lambda  t, z0  : odeint(self.z_model, 
#                                             z0, 
#                                             # torch.from_numpy(np.linspace(0, t, self.err_tsteps)),
#                                             torch.from_numpy(time_steps_func(t, self.int_tsteps )), 
#                                             rtol=rTOL, atol=aTOL, method='rk4')
        
#         self.v_t = lambda t, v0: odeint(self.v_model, v0, 
#                                         t, 
#                                         rtol=rTOL, atol=aTOL, method='rk4')
    
#     def set_z0(self, z0=None, batch_size=None): 
#         if  z0 is None:      z0_ = torch.zeros((self.x_data.shape[0], self.indim), requires_grad=True).to(self.device)
#         elif z0 == 'det':    z0_ = self.determine_zs(self.x_data, 10000)
#         elif z0 == 'update': z0_ = self.z0.repeat(self.x_data.shape[0], 1)
#         else: z0_ = z0
#         return z0_
    
#     def backward_G(self, x_data, z0=None): 
#         self.x_data = x_data
#         z0_ = self.set_z0(z0=z0, batch_size=x_data.shape[0])
#         self.zs = self.z_t(self.tau, z0_)
#         self.ztau = self.zs[-1]
#         self.inttlist = torch.Tensor(self.inttlist)
#         x_model, log_px = self.net(self.ztau)    
#         self.vtau= jacobian(self.floss(x_model, self.x_data), self.ztau)
#         losstau = self.floss( x_model , self.x_data)     
#         self.index = self.inttlist.shape[0]
#         vtlist = torch.flip(self.tlist,dims=[0])
#         vs = self.v_t(vtlist, self.vtau) 
#         zvt = torch.stack([self.zs, torch.flip(vs, dims=[0])], dim=1)
#         DlDphi_list = []
#         print('>>> ran the error backwards...computing the derivatives for the model parameters')
#         for i, self.phi in enumerate(self.net.parameters()):
#             dldphi=jacobian(losstau, self.phi) 
#             adjoint_model = self.err_model(self.tlist[1]-self.tlist[0], zvt).detach()
#             full_derivative = -1.0*adjoint_model + dldphi
#             # print('--adjoint model - normal derivatives / normal derivatives ratio mean--')
#             # print(torch.mean(torch.abs(adjoint_model)), torch.mean(torch.abs(dldphi)))
#             if torch.isnan(torch.mean(torch.abs(adjoint_model))): 
#                 sys.exit(0)
#             DlDphi_list.append(full_derivative)
#         print('>>> ...computing the derivative for the initial condition')
#         DlDphi_list.append(torch.sum(vs[-1], dim=0)) #ask ender 
#         self.zero_lists()
        
#         # DlDphi= torch.cat(DlDphi_list, dim=1) 
#         return DlDphi_list
    
#     def backward_approx(self, x_data, z0=None): 
#         self.x_data = x_data
#         z0_ = self.set_z0(z0=z0)
#         self.zs = self.z_t(self.tau, z0_)
#         self.ztau = self.zs[-1]
#         self.inttlist = torch.Tensor(self.inttlist)
#         self.vtau= 1.0*jacobian(self.floss(self.net(self.ztau)[0], self.x_data), self.ztau)
#         losstau = self.floss( self.net(self.ztau)[0], self.x_data)     
#         DlDphi_list = []
#         # print('>>> ran the error backwards...ignoring the adjoint part')
#         for i, self.phi in enumerate(self.net.parameters()):
#             dldphi=jacobian(losstau, self.phi) 
#             full_derivative = dldphi
#             DlDphi_list.append(full_derivative)
#         DlDphi_list.append(torch.sum(self.vtau, dim=0)) #ask ender 
#         self.zero_lists()
#         return DlDphi_list
       
#     def zero_lists(self): 
#         self.inttlist = []
#         self.zlist = []
        
#     def determine_zs(self, x_data, tau=None, z0=None): 
#         self.x_data = x_data
#         z0_ = self.set_z0(z0=z0, batch_size=x_data.shape[0])
#         if tau is None: 
#             self.ztau = self.z_t(self.tau, z0_)[-1]
#         else: 
#             self.ztau = self.z_t(tau, z0_)[-1]
#         self.zero_lists()
#         return self.ztau
     
#     def z_model(self, t, z): 
#         self.inttlist.append(t)
#         self.zlist.append(z.clone())
#         zt = z.clone()
#         x_model, log_px = self.net(zt)    
#         loss=self.floss(x_model ,self.x_data)
#         dLdz = -1.0 * step_size_func(t,self.tau) * jacobian(loss,zt);    
#         return dLdz
    
#     def v_model(self, t, v):
#         vt = v.clone() # that is lambda
#         self.index = self.index-1
#         #zt = self.zs[torch.where(self.inttlist == t)[0][0]]
#         zt = self.zlist[self.index]
#         x_model, log_px = self.net(zt)    
#         loss=self.floss(x_model,self.x_data)
#         hess=hessian_NEW(loss*torch.ones(1,dtype=torch.float)[0], zt, vt)
#         ddLdz = step_size_func(t,self.tau)*hess
        
#         return ddLdz
    
    
#     def err_model(self, dt, zvt):  
#         dLdzdphi = torch.zeros_like(self.phi)
#         for j in range(0, zvt.shape[0], 4):
#             zt = zvt[j,0].clone()
#             vt = zvt[j,1].clone()
#             x_model, log_px = self.net(zt)    
#             loss=self.floss(x_model ,self.x_data)
#             ddLdzdphi = doublenabla(loss * torch.ones(1, dtype=torch.float)[0], 
#                                         zt, 
#                                         self.phi, 
#                                         vt)
#             dLdzdphi += step_size_func(self.tlist[j],self.tau) * ddLdzdphi * dt * 4
#         return dLdzdphi
    
class Autograd(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __init__(self, net, indim, floss, tau=1.0, rTOL=1e-8, aTOL=1e-9, int_tsteps=200, device='cuda'):
        super(Autograd,self).__init__()
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
    
    def set_z0(self, z0=None, batch_size=None): 
        if  z0 is None:      z0_ = torch.zeros((self.x_data.shape[0], self.indim), requires_grad=True).to(self.device)
        elif z0 == 'det':    z0_ = self.determine_zs(self.x_data, 10000)
        elif z0 == 'update': z0_ = self.z0.repeat(self.x_data.shape[0], 1)
        else: z0_ = z0
        return z0_
    
    def backward_G(self, x_data, z0=None): 
        self.x_data = x_data
        z0_ = self.set_z0(z0=z0, batch_size=x_data.shape[0])
        self.zs = self.z_t(self.tau, z0_)
        self.ztau = self.zs[-1]
        self.inttlist = torch.Tensor(self.inttlist)
        x_model, log_px = self.net(self.ztau)    
        self.vtau= jacobian(self.floss(x_model, self.x_data), self.ztau)
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
        
        # DlDphi= torch.cat(DlDphi_list, dim=1) 
        return DlDphi_list
    
    def backward_approx(self, x_data, z0=None): 
        self.x_data = x_data
        z0_ = self.set_z0(z0=z0)
        self.zs = self.z_t(self.tau, z0_)
        self.ztau = self.zs[-1]
        self.inttlist = torch.Tensor(self.inttlist)
        self.vtau= 1.0*jacobian(self.floss(self.net(self.ztau)[0], self.x_data), self.ztau)
        losstau = self.floss( self.net(self.ztau)[0], self.x_data)     
        DlDphi_list = []
        # print('>>> ran the error backwards...ignoring the adjoint part')
        for i, self.phi in enumerate(self.net.parameters()):
            dldphi=jacobian(losstau, self.phi) 
            full_derivative = dldphi
            DlDphi_list.append(full_derivative)
        DlDphi_list.append(torch.sum(self.vtau, dim=0)) #ask ender 
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
        self.inttlist.append(t)
        self.zlist.append(z.clone())
        zt = z.clone()
        x_model, log_px = self.net(zt)    
        loss=self.floss(x_model ,self.x_data)
        dLdz = -1.0 * step_size_func(t,self.tau) * jacobian(loss,zt);    
        return dLdz
    
    def v_model(self, t, v):
        vt = v.clone() # that is lambda
        self.index = self.index-1
        #zt = self.zs[torch.where(self.inttlist == t)[0][0]]
        zt = self.zlist[self.index]
        x_model, log_px = self.net(zt)    
        loss=self.floss(x_model,self.x_data)
        hess=hessian_NEW(loss*torch.ones(1,dtype=torch.float)[0], zt, vt)
        ddLdz = step_size_func(t,self.tau)*hess
        
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
            dLdzdphi += step_size_func(self.tlist[j],self.tau) * ddLdzdphi * dt * 4
        return dLdzdphi
           
    def forward(self, x_data, gflow):
        x=x_data        
        zs = gflow.determine_zs()  
        x_model=gflow.net(zs)  
        # dlist = gflow.backward(x_data, z0=None)
        return x_model
    
    def backward(self, grad_output):
        """
        In the backward pass we receive the context object and a Tensor containing
        the gradient of the loss with respect to the output produced during the
        forward pass. We can retrieve cached data from the context object, and must
        compute and return the gradient of the loss with respect to the input to the
        forward function.
        """
        gradients = grad_output.clone()#this would be the gradients of x_model i.e. the return of forward
        dlist = backward_G(self.x_data, z0=None)
        
        
        
        # zs, stupid,x_data = ctx.saved_tensors
        
        # gradients=gflow.backward_G(x_data, z0=None)
        # grad_input = grad_output.clone()
        return dlist 




class Autograd_simple(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, gflow, zs ):
        """
        In the forward pass we receive a context object and a Tensor containing the
        input; we must return a Tensor containing the output, and we can use the
        context object to cache objects for use in the backward pass.
        """
        x_model=gflow.net(zs)[0] 
        # x_model=torch.autograd.Variable(torch.zeros(1,1), requires_grad=True)
        ctx.save_for_backward(gflow, zs)
        return x_model    
    
    # @oncedifferentiable          
    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive the context object and a Tensor containing
        the gradient of the loss with respect to the output produced during the
        forward pass. We can retrieve cached data from the context object, and must
        compute and return the gradient of the loss with respect to the input to the
        forward function.
        """
        gradients = grad_output.clone() #this would be the gradients of x_model i.e. the return of forward
        gflow, zs = ctx.saved_tensors
        DlDphi_list = []
        for i, gflow.phi in enumerate(gflow.net.parameters()):
            dldphi=jacobian(gflow.losstau, gflow.phi) 
            adjoint_model = gflow.err_model(gflow.tlist[1]-gflow.tlist[0], gflow.zvt).detach()
            full_derivative = -1.0*adjoint_model + dldphi
            DlDphi_list.append(full_derivative)
        # DlDphi_list.append(torch.sum(vs[-1], dim=0)) #ask ender 
        gflow.zero_lists()
        
        return gradients
    
#Should i try the other method?
# (1)

# class Test(torch.autograd.Function):
#     def __init__(self):
#         super(Test,self).__init__()

#     def forward(self, x1, x2):
#         self.state = state(x1)
#         return torch.arange(8)

#     def backward(self, grad_out):
#         grad_input = grad_out.clone()
#         return torch.arange(10,18),torch.arange(20,28)

# # then use function = Test()
# or
# (2)

# class Test(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x1, x2):
#         ctx.state = state(x1)
#         return torch.arange(8)

#     @staticmethod
#     def backward(ctx, grad_out):
#         grad_input = grad_out.clone()
#         return torch.arange(10,18),torch.arange(20,28)

# # then use function = Test.apply

