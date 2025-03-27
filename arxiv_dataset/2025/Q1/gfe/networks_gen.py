import torch
import numpy as np
from networks_gen_grad import expm_func, expm_rot, asinh
#import expm
# import math
import sys
sys.path.append("pytorch_expm")
from pytorch_expm import expm_taylor

#================================================


class net_generative(torch.nn.Module):
    def __init__(self, init_dim, final_dim, flow_num, device, dimlist=None):
        """
        In the constructor we instantiate our parameters to be optimized.
        """
        super(net_generative, self).__init__()
        self.flow_num = flow_num
        # assert (final_dim - init_dim) % flow_num == 0, \
        #     "The dimension increase can not be divided into steps of equal size. " \
        #     "Please adjust flow number such that: (final_dim - init_dim) % flow_num == 0"        
        self.device = device


        self.linsin = lambda z:  1.2*z + torch.sin(z)
        self.det_linsin = lambda z:  torch.sum(torch.log(torch.cos(z) + 1.2), dim=0)

        self.rot_func = lambda z,L: torch.matmul(expm_taylor.expm_taylor(L - L.T), z) 
#         # self.rot_func= lambda z, L : torch.matmul(expm_taylor.expm(L - L.T),z)
        # self.rot_func = lambda z,L:   expm_rot.apply(L,z)
# # 
        # self.linear= lambda z, L ,b : torch.matmul(expm_taylor.expm(L),z) + b
#         # self.linear= lambda z, L ,b : expm_taylor.expm(L,z)+ b               
        self.linear= lambda z, L ,b : torch.matmul(expm_taylor.expm_taylor(L), z) + b
        # self.linear= lambda z, L ,b : expm_func.apply(L,z) + b 
        self.det_linear= lambda L: torch.trace(L)             
        self.f_x_density_affine = lambda z: -0.5*(torch.matmul(z.t(), z).diag() +  z.shape[0]*np.log(np.pi))
       
        self.arcsinh= lambda z: torch.log(z+(z**2+1)**0.5)
        self.det_arcsinh= lambda z: torch.sum(torch.log(1.0/torch.sqrt(1.0 + z**2)), dim=0)
      
     
        self.ELU= lambda z: torch.nn.functional.elu(z)
        self.det_ELU= lambda z: torch.sum(torch.log(1*(z>0) + torch.exp(z)*(z<=0)), dim=0)
             
      
        self.sinh= lambda z:  torch.sinh(z)
        self.det_sinh= lambda z: torch.sum(torch.log(torch.cosh(z)), dim=0)      
        

        
        if dimlist is None:
            self.dim_step = int((final_dim-init_dim)/flow_num)
            self.dimlist = list(np.arange(init_dim, final_dim+1, self.dim_step))
        else:
            assert dimlist[0] == init_dim
            assert dimlist[-1] == final_dim
            assert (np.diff(dimlist) >= 0).all()
            self.dimlist = dimlist
    
        self.Lw = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros((d, d), requires_grad=True, device=device)) for d in self.dimlist[:-1]])
        self.b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros((d, 1), requires_grad=True, device=device)) for d in self.dimlist[:-1]])
        self.Rw = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros((d, d), requires_grad=True, device=device)) for d in self.dimlist[1:]])
        # self.Lw = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros((init_dim, init_dim), requires_grad=True, device=device))
        #                                 for i in range(flow_num))
        # self.b = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros((init_dim, 1), requires_grad=True, device=device))
        #                                 for i in range(flow_num))
        # self.R = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros((init_dim, init_dim), requires_grad=True, device=device))
        #                                 for i in range(flow_num))

        
    def forward(self, z0):
         z0 = z0.t()
         list_z = [z0]
         log_det_factor = 0
         for i in range(len(self.dimlist)-1):
           #linear function
           z1 = self.linear(list_z[-1], self.Lw[i], self.b[i])
           log_det1 = self.det_linear(self.Lw[i])
           # z1, log_det1 =linear_func(list_z[-1], self.Lw[i], self.b[i])
           #non-linearity
           # print('linear', log_det1.data)
           if i % 2 == 0:
                    # z2 = self.sinh(z1)
                    # log_det2 = self.det_sinh(z1) 
                    z2 = self.arcsinh(z1)
                    log_det2 = self.det_arcsinh(z1)                    
                    # print('sinh' , i, self.dimlist)
                    # print('arcsinh', log_det2.data)
                  # z2, log_det2 =arcsinh_func(z1)                 
                    # z2 = self.ELU(z1)
                    # log_det2 = self.det_ELU(z1) 
                   # print( log_det1 , log_det2)          
                    # z2 = self.linear(z1)
                    # log_det2 = self.det_linsin(z1)                      
           else:
                    # z2 = self.linsin(z1)
                    # log_det2 = self.det_linsin(z1)   
                    z2 = self.sinh(z1)
                    log_det2 = self.det_sinh(z1)                     
                    # z2 = self.arcsinh(z1)
                    # log_det2 = self.det_arcsinh(z1)
                    # print('arcsinh', log_det2.data)

                   # z2, log_det2 =sinh_func(z1)
                    # z2 = self.ELU(z1)
                    # log_det2 = self.det_ELU(z1)                
                    # print( log_det1 , log_det2)          

            #rotation
           zero_fill = torch.zeros((self.dimlist[i+1]-self.dimlist[i], z0.shape[1]), device=self.device)
           z2_aug = torch.cat((z2, zero_fill), 0)
           z3 = self.rot_func(z2_aug, self.Rw[i])
           #save determinant and layer output
           log_det_factor += 2*log_det1 + 2*log_det2
           list_z.append(z3)

         # import sys
         # sys.exit(0)
         log_pz = self.f_x_density_affine(list_z[-1])
         log_px =  log_pz -0.5*log_det_factor #- torch.log(torch.sqrt(2.0))
         # print(log_det_factor.mean())
         # import sys
         # sys.exit(0)
         px = torch.exp(log_px)
         return list_z[-1].t(), log_px
        
        # z0 = z0.t()
        # list_z = [z0]
        # log_det_factor = 0        
        # for x in range(0, self.flow_num):
        #      z1, log_det1 =linear_func(list_z[-1], self.Lw[x], self.b[x])
        #      if x % 2 == 0:
        #          z2, log_det2 =arcsinh_func(z1)
        #      else:
        #          z2, log_det2 =sinh_func(z1)
        #      z3, log_det3 = linear_func(z2, self.R[x])
        #      log_det_factor = log_det_factor + log_det1 + log_det2 + log_det3
        #      list_z.append(z3)
        # 
        # log_pz = self.f_x_density_affine(list_z[-1])
        # log_px = log_pz + log_det_factor #- torch.log(torch.sqrt(2.0))
        # px = torch.exp(log_px)
        # return list_z[-1].t(), log_px

  
    # return list_z[-1], log_final_prob
#============================================================


import math

def sin(z):
    z_out =0.8*z + torch.sin(0.3*z)

    det = torch.sum(torch.log(torch.cos(z) + 1.2), dim=0)

    return z_out, det 
def linear_func(z, L, b=None):
    '''
    The inverse function that updates the z.
    z2 = exmp(L)*z1 + b
    The log determinant of the inverse function.
    log(det(g(x)')) = log(det(exmp(L))) = log(exp(trace(L))) = trace(L)
    '''
    det = torch.trace(L)

    if b is None:
        return expm_func.apply(L, z), det
    else:
        return expm_func.apply(L, z) + b, det

def sinh_func(z):
    '''
    z2 = sinh(z1)
    log(det(g(z)')) = log(det(cosh(z)))
    '''

    det = torch.sum(torch.log(torch.cosh(z)), dim=0)
    #det = torch.cosh(z[0, :])*torch.cosh(z[1, :])

    return torch.sinh(z), det

def arcsinh_func(z):
    '''
    z2 = arcsinh(z1)
    log(det(g(z)')) = log(det(1/(sqrt(1 + z^2))
    '''
    det = torch.sum(torch.log(1.0/torch.sqrt(1.0 + z**2)), dim=0)
    #det = (1.0/torch.sqrt(1.0 + z[0, :]**2))*(1.0/torch.sqrt(1.0 + z[1, :]**2))

    return asinh.apply(z), det

def tanh_func(z):
    '''
    z2 = tanh(z1)
    log(det(g(z)')) = log(det(1-tanh(z)^2))
    '''
    det = torch.sum(torch.log(1.0-torch.tanh(z)**2), dim=0)
    #det = (1-torch.tanh(z[0, :])**2)*(1-torch.tanh(z[1, :])**2)

    return torch.tanh(z), det


def f_x_density_affine(z):
    '''
    The log probability of the transformed coordinates in the "initial" fx space.
    Here the initial distribution is modelled as gaussian.
    Since we have the mean at 0 and the identity matrix as co-varaince this is what
    the log of a multivariate gaussian boils down to
    '''
    dim = z.shape[0]
    return -0.5*(torch.matmul(z.t(), z).diag() + dim*np.log(math.pi))


def inf_train_gen(batch_size, order=1.0):
    """
    Generates the coordinates (sample points in target space) to be transformed to
    the initial distribution coordinates. Order scales this distribution.
    """
    kernel_size = int(np.sqrt(batch_size))
    data = np.zeros((batch_size, 2))
    runner = 0
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            data[runner, 0] = i - int(kernel_size/2)
            data[runner, 1] = j - int(kernel_size/2)
            runner += 1

    return torch.from_numpy(data * order)

def rot_func(z, L):
    '''
    The inverse function that updates the z.
    z2 = W*z1
    The determinant of the inverse function.
    log(det(g(x)')) = log(exp(trace(L- L.t))) = trace(L - L.t)
    '''
    #det = torch.trace(L - L.t())

    return expm_rot.apply(L, z)

#================================================



class flow_2step(torch.nn.Module):
  def __init__(self, dim, flow_num, device):
    """
    In the constructor we instantiate our parameters to be optimized.
    """
    super(flow_2step, self).__init__()
    self.flow_num = flow_num
    self.device = device

    self.Lw = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros((dim, dim), requires_grad=True, device=device))
                                    for i in range(flow_num))
    self.b = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros((dim, 1), requires_grad=True, device=device))
                                    for i in range(flow_num))

  def forward(self, z0):
    """
    The forward function takes the z0 input coordinates and transform them to zN output coordinates.
    The zN output coordinates are then evaluated at the initial distribution
    To approximate the probability of the input coordinates, the normalizing flow theory is applied by
    multiplying the probability of the samples in the initial distribution with the det of the coordinate
    transformation function.
    To make everything numerically stable we use log probabilities and then convert them back in the end.

    The forward flow is done according to the number of specified flow layers.
    The output is the probability of each input point.
    """
    z0 = z0.t()    
    list_z = [z0]
    log_det_factor = 0
    for x in range(0, self.flow_num):
        z1, log_det1 = linear_func(list_z[-1], self.Lw[x], self.b[x])
        if x % 2 == 0:
            z2, log_det2 = sin(z1)
            #z2, log_det2 = sinh_func(z1)
        else:
            z2, log_det2 = sin(z1)
            #z2, log_det2 = arcsinh_func(z1)
        log_det_factor = log_det_factor + log_det1 + log_det2
        list_z.append(z2)

    log_z_prob = f_x_density_affine(list_z[-1])
    log_final_prob = log_z_prob + log_det_factor
    final_prob = torch.exp(log_final_prob)

    return list_z[-1].t(), log_final_prob


class flow(torch.nn.Module):
  def __init__(self, dim, flow_num, device):
    """
    In the constructor we instantiate our parameters to be optimized.
    """
    super(flow, self).__init__()
    self.flow_num = flow_num
    self.device = device

    self.Lw = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros((dim, dim), requires_grad=True, device=device))
                                    for i in range(flow_num))
    self.b = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros((dim, 1), requires_grad=True, device=device))
                                    for i in range(flow_num))
    self.R = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros((dim, dim), requires_grad=True, device=device))
                                    for i in range(flow_num))

  def forward(self, z0):
    """
    The forward function takes the z0 input coordinates and transform them to zN output coordinates.
    The zN output coordinates are then evaluated at the initial distribution
    To approximate the probability of the input coordinates, the normalizing flow theory is applied by
    multiplying the probability of the samples in the initial distribution with the det of the coordinate
    transformation function.
    To make everything numerically stable we use log probabilities and then convert them back in the end.

    The forward flow is done according to the number of specified flow layers.
    The output is the probability of each input point.
    """
    z0 = z0.t()

    list_z = [z0]
    log_det_factor = 0
    for x in range(0, self.flow_num):
        z1, log_det1 = linear_func(list_z[-1], self.Lw[x], self.b[x])
        if x % 2 == 0:
            z2, log_det2 = arcsinh_func(z1)
        else:
            z2, log_det2 = sinh_func(z1)
        z3, log_det3 = linear_func(z2, self.R[x])
        log_det_factor += log_det1 + log_det2 + log_det3
        list_z.append(z3)

    log_z_prob = f_x_density_affine(list_z[-1])
    log_final_prob = log_z_prob + log_det_factor
    final_prob = torch.exp(log_final_prob)

    return list_z[-1].t(), log_final_prob

class dim_flow(torch.nn.Module):
  def __init__(self, init_dim, final_dim, flow_num, device):
    """
    In the constructor we instantiate our parameters to be optimized.
    """
    super(dim_flow, self).__init__()
    self.flow_num = flow_num
    self.device = device
    assert (final_dim - init_dim) % flow_num == 0, \
        "The dimension increase can not be divided into steps of equal size. " \
        "Please adjust flow number such that: (final_dim - init_dim) % flow_num == 0"
    dim_step = int((final_dim-init_dim)/flow_num)
    self.dim_step = dim_step

    self.Lw = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros(((i*dim_step)+init_dim, (i*dim_step)+init_dim),
                            requires_grad=True, device=device)) for i in range(flow_num))
    self.b = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros(((i*dim_step)+init_dim, 1),
                            requires_grad=True, device=device)) for i in range(flow_num))
    self.Rw = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros((((i+1)*dim_step)+init_dim, ((i+1)*dim_step)+init_dim),
                            requires_grad=True, device=device)) for i in range(flow_num))

  def forward(self, z0):
    """
    The forward flow is done according to the number of specified flow layers.
    The output is the probability of each input point.
    """
    # z0 = z0.t()
    list_z = [z0]
    log_det_factor = 0
    zero_fill = torch.zeros((self.dim_step, z0.shape[1]), device=self.device)
    for x in range(0, self.flow_num):
        #linear function
        z1, log_det1 = linear_func(list_z[-1], self.Lw[x], self.b[x])
        #non-linearity
        if x % 2 == 0:
            z2, log_det2 = arcsinh_func(z1)
        else:
            z2, log_det2 = sinh_func(z1)
        #rotation
        z2_aug = torch.cat((z2, zero_fill), 0)
        z3 = rot_func(z2_aug, self.Rw[x])
        #save determinant and layer output
        log_det_factor = log_det_factor + 2*log_det1 + 2*log_det2
        list_z.append(z3)

    log_z_prob = f_x_density_affine(list_z[-1])
    log_final_prob = log_z_prob + log_det_factor
    final_prob = torch.exp(log_final_prob)

    return final_prob


# def tanh_func(z):
#     '''
#     z2 = tanh(z1)
#     log(det(g(z)')) = log(det(1-tanh(z)^2))
#     '''
#     det = torch.sum(torch.log(1.0-torch.tanh(z)**2), dim=0)
#     #det = (1-torch.tanh(z[0, :])**2)*(1-torch.tanh(z[1, :])**2)

#     return torch.tanh(z), det


# def inf_train_gen(batch_size, order=1.0):
#     """
#     Generates the coordinates (sample points in target space) to be transformed to
#     the initial distribution coordinates. Order scales this distribution.
#     """
#     kernel_size = int(np.sqrt(batch_size))
#     data = np.zeros((batch_size, 2))
#     runner = 0
#     for i in range(0, kernel_size):
#         for j in range(0, kernel_size):
#             data[runner, 0] = i - int(kernel_size/2)
#             data[runner, 1] = j - int(kernel_size/2)
#             runner += 1

#     return torch.from_numpy(data * order)








# class dim_flow_laffine(torch.nn.Module):
#   def __init__(self, init_dim, final_dim, flow_num, device):
#     """
#     In the constructor we instantiate our parameters to be optimized.
#     """
#     super(dim_flow_laffine, self).__init__()
#     self.flow_num = flow_num
#     self.device = device
#     assert (final_dim - init_dim) % flow_num == 0, \
#         "The dimension increase can not be divided into steps of equal size. " \
#         "Please adjust flow number such that: (final_dim - init_dim) % flow_num == 0"
#     dim_step = int((final_dim-init_dim)/flow_num)
#     self.dim_step = dim_step

#     self.Lw = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros(((i*dim_step)+init_dim, (i*dim_step)+init_dim),
#                             requires_grad=True, device=device)) for i in range(flow_num))
#     self.b = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros(((i*dim_step)+init_dim, 1),
#                             requires_grad=True, device=device)) for i in range(flow_num))
#     self.Rw = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros((((i+1)*dim_step)+init_dim, ((i+1)*dim_step)+init_dim),
#                             requires_grad=True, device=device)) for i in range(flow_num))
#     self.Lf = torch.nn.Parameter(torch.zeros((int(final_dim), int(final_dim)),
#                             requires_grad=True, device=device))
#     self.bf = torch.nn.Parameter(torch.zeros((int(final_dim), 1),
#                             requires_grad=True, device=device))

#   def forward(self, z0):
#     """
#     The forward flow is done according to the number of specified flow layers.
#     The output is the probability of each input point.
#     """
#     z0 = z0.t()
#     list_z = [z0]
#     log_det_factor = 0
#     zero_fill = torch.zeros((self.dim_step, z0.shape[1]), device=self.device)
#     for x in range(0, self.flow_num):
#         #linear function
#         z1, log_det1 = linear_func(list_z[-1], self.Lw[x], self.b[x])
#         #non-linearity
#         if x % 2 == 0:
#             z2, log_det2 = arcsinh_func(z1)
#         else:
#             z2, log_det2 = sinh_func(z1)
#         #rotation
#         z2_aug = torch.cat((z2, zero_fill), 0)
#         z3 = rot_func(z2_aug, self.Rw[x])
#         #save determinant and layer output
#         log_det_factor = log_det_factor + 2*log_det1 + 2*log_det2
#         list_z.append(z3)
#         if x == (self.flow_num-1):
#             z4, log_det4 = linear_func(list_z[-1], self.Lf, self.bf)
#             list_z.append(z4)
#             log_det_factor = log_det_factor + 2*log_det4

#     log_z_prob = f_x_density_affine(list_z[-1])
#     log_final_prob = log_z_prob - 0.5*log_det_factor
#     final_prob = torch.exp(log_final_prob)

#     return list_z[-1].t(), log_final_prob


# # class flow_2step(torch.nn.Module):
# #   def __init__(self, dim, flow_num, device):
# #     """
# #     In the constructor we instantiate our parameters to be optimized.
# #     """
# #     super(flow_2step, self).__init__()
# #     self.flow_num = flow_num
# #     self.device = device

# #     self.Lw = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros((dim, dim), requires_grad=True, device=device))
# #                                     for i in range(flow_num))
# #     self.b = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros((dim, 1), requires_grad=True, device=device))
# #                                     for i in range(flow_num))

# #   def forward(self, z0):
# #     """
# #     The forward function takes the z0 input coordinates and transform them to zN output coordinates.
# #     The zN output coordinates are then evaluated at the initial distribution
# #     To approximate the probability of the input coordinates, the normalizing flow theory is applied by
# #     multiplying the probability of the samples in the initial distribution with the det of the coordinate
# #     transformation function.
# #     To make everything numerically stable we use log probabilities and then convert them back in the end.

# #     The forward flow is done according to the number of specified flow layers.
# #     The output is the probability of each input point.
# #     """
# #     list_z = [z0]
# #     log_det_factor = 0
# #     for x in range(0, self.flow_num):
# #         z1, log_det1 = linear_func(list_z[-1], self.Lw[x], self.b[x])
# #         if x % 2 == 0:
# #             z2, log_det2 = arcsinh_func(z1)
# #             #z2, log_det2 = sinh_func(z1)
# #         else:
# #             z2, log_det2 = sinh_func(z1)
# #             #z2, log_det2 = arcsinh_func(z1)
# #         log_det_factor = log_det_factor + log_det1 + log_det2
# #         list_z.append(z2)

# #     log_z_prob = f_x_density_affine(list_z[-1])
# #     log_final_prob = log_z_prob + log_det_factor
# #     final_prob = torch.exp(log_final_prob)

#     return final_prob


# class flow(torch.nn.Module):
#   def __init__(self, dim, flow_num, device):
#     """
#     In the constructor we instantiate our parameters to be optimized.
#     """
#     super(flow, self).__init__()
#     self.flow_num = flow_num
#     self.device = device

#     self.Lw = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros((dim, dim), requires_grad=True, device=device))
#                                     for i in range(flow_num))
#     self.b = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros((dim, 1), requires_grad=True, device=device))
#                                     for i in range(flow_num))
#     self.R = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros((dim, dim), requires_grad=True, device=device))
#                                     for i in range(flow_num))

#   def forward(self, z0):
#     """
#     The forward function takes the z0 input coordinates and transform them to zN output coordinates.
#     The zN output coordinates are then evaluated at the initial distribution
#     To approximate the probability of the input coordinates, the normalizing flow theory is applied by
#     multiplying the probability of the samples in the initial distribution with the det of the coordinate
#     transformation function.
#     To make everything numerically stable we use log probabilities and then convert them back in the end.

#     The forward flow is done according to the number of specified flow layers.
#     The output is the probability of each input point.
#     """
#     list_z = [z0]
#     log_det_factor = 0
#     for x in range(0, self.flow_num):
#         z1, log_det1 = linear_func(list_z[-1], self.Lw[x], self.b[x])
#         if x % 2 == 0:
#             z2, log_det2 = arcsinh_func(z1)
#         else:
#             z2, log_det2 = sinh_func(z1)
#         z3, log_det3 = linear_func(z2, self.R[x])
#         log_det_factor = log_det_factor + log_det1 + log_det2 + log_det3
#         list_z.append(z3)

#     log_z_prob = f_x_density_affine(list_z[-1])
#     log_final_prob = log_z_prob + log_det_factor
#     final_prob = torch.exp(log_final_prob)

#     return final_prob

# def f_x_density_affine(z, center=0, var=[1, 1], device=torch.device("cpu")):
#     '''
#     The probability of the transformed coordinates in the "initial" fx space.
#     Here the initial distribution is modelled as gaussian.
#     '''
#     sigma = torch.tensor(var, device=device)
#     return torch.exp(-((((z[0, :] - center) ** 2) / (2 * sigma[0] ** 2)) +
#                        (((z[1, :] - center) ** 2) / (2 * sigma[1] ** 2))))

# class flow_3step(torch.nn.Module):
#   def __init__(self, dim, flow_num, device):
#     """
#     In the constructor we instantiate our parameters to be optimized.
#     """
#     super(flow_3step, self).__init__()
#     self.flow_num = flow_num
#     self.device = device
#
#     self.Lw = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros((dim, dim), requires_grad=True, device=device))
#                                     for i in range(flow_num))
#     self.b = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros((dim, 1), requires_grad=True, device=device))
#                                     for i in range(flow_num))
#
#   def forward(self, z0):
#     """
#     The forward function takes the z0 input coordinates and transform them to zN output coordinates.
#     The zN output coordinates are then evaluated at the initial distribution
#     To approximate the probability of the input coordinates, the normalizing flow theory is applied by
#     multiplying the probability of the samples in the initial distribution with the det of the coordinate
#     transformation function.
#     To make everything numerically stable we use log probabilities and then convert them back in the end.
#
#     The forward flow is done according to the number of specified flow layers.
#     The output is the probability of each input point.
#     """
#     list_z = [z0]
#     det_factor = 0
#     tracker = 0
#     for x in range(0, self.flow_num):
#         z1, log_det1 = linear_func(list_z[-1], self.Lw[x], self.b[x])
#         if tracker == 0:
#             #z2, log_det2 = arcsinh_func(z1)
#             z2, log_det2 = tanh_func(z1)
#             tracker = tracker + 1
#         if tracker == 1:
#             z2, log_det2 = sinh_func(z1)
#             tracker = tracker + 1
#         if tracker == 2:
#             #z2, log_det2 = tanh_func(z1)
#             z2, log_det2 = arcsinh_func(z1)
#             tracker = 0
#         log_det_factor = det_factor + log_det1 + log_det2
#         list_z.append(z2)
#
#     log_z_prob = f_x_density_affine(list_z[-1])
#     log_final_prob = log_z_prob + log_det_factor
#     final_prob = torch.exp(log_final_prob)
#
#     return final_prob
