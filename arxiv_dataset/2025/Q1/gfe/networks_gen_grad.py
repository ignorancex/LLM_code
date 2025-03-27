import torch
from scipy.linalg import expm
import numpy as np
import sys
sys.path.append("pytorch_expm")
from pytorch_expm.expm_taylor import expm_taylor

class expm_rot(torch.autograd.Function):
  """
  We can implement our own custom autograd Functions by subclassing
  torch.autograd.Function and implementing the forward and backward passes
  which operate on Tensors.
  """
  @staticmethod
  def forward(ctx, L, x0):
    """
    In the forward pass we receive a context object and a Tensor containing the
    input; we must return a Tensor containing the output, and we can use the
    context object to cache objects for use in the backward pass.
    """
    device = L.device
    #Calculate the forward value
    Ac = torch.from_numpy(expm(L.detach().cpu().numpy() - L.detach().cpu().numpy().T)).to(device)
    # Ac =expm_taylor(L-L.T)   
    output = torch.matmul(Ac, x0)

    #Cache tensors for backward pass
    ctx.save_for_backward(L, x0)

    return output

  @staticmethod
  def backward(ctx, grad_output):
    """
    In the backward pass we receive the context object and a Tensor containing
    the gradient of the loss with respect to the output produced during the
    forward pass. We can retrieve cached data from the context object, and must
    compute and return the gradient of the loss with respect to the input to the
    forward function.
    """
    L, x0 = ctx.saved_tensors
    grad_L = grad_x0 = None
    device = L.device

    if ctx.needs_input_grad[0]:
      grad_x = grad_output.clone()
      dim = L.shape[0]
      B = torch.matmul(grad_x, x0.t())
      K1 = torch.cat(((L - L.t()).T, B), 1)
      K2 = torch.cat((torch.zeros((dim, dim)).to(device), (L - L.t()).T), 1)
      K = torch.cat((K1, K2), 0)
      # grad_L1 = expm_taylor(K)[:dim, dim:]
     
      grad_L1 = torch.from_numpy(expm(K.detach().cpu().numpy())).to(device)[:dim, dim:]
      # print(grad_L1_old.shape)

      B = torch.matmul(x0, grad_x.t())
      K1 = torch.cat(((L - L.t()), B), 1)
      K2 = torch.cat((torch.zeros((dim, dim)).to(device), (L - L.t())), 1)
      K = torch.cat((K1, K2), 0)
      # grad_L2 = expm_taylor(K)[:dim, dim:]
      
      grad_L2 = torch.from_numpy(expm(K.detach().cpu().numpy())).to(device)[:dim, dim:]

      grad_L = grad_L1 - grad_L2

    if ctx.needs_input_grad[1]:
      # Ac = torch.from_numpy(expm(L.detach().cpu().numpy())).to(device)
      Ac =expm_taylor(L)
      
      grad_x = grad_output.clone()
      grad_x0 = torch.matmul(Ac, grad_x)

    return grad_L, grad_x0

class expm_func(torch.autograd.Function):
  """
  We can implement our own custom autograd Functions by subclassing
  torch.autograd.Function and implementing the forward and backward passes
  which operate on Tensors.
  """
  @staticmethod
  def forward(ctx, L, x0):
    """
    In the forward pass we receive a context object and a Tensor containing the
    input; we must return a Tensor containing the output, and we can use the
    context object to cache objects for use in the backward pass.
    """
    #Calculate the forward value
    device = L.device
    Ac = torch.from_numpy(expm(L.detach().cpu().numpy())).to(device)
    # Ac =expm_taylor(L)

    # Ac = torch.from_numpy(expm(L.detach().cpu().numpy())).to(device)
    
    output = torch.matmul(Ac, x0)

    #Cache tensors for backward pass
    ctx.save_for_backward(L, x0)

    return output

  @staticmethod
  def backward(ctx, grad_output):
    """
    In the backward pass we receive the context object and a Tensor containing
    the gradient of the loss with respect to the output produced during the
    forward pass. We can retrieve cached data from the context object, and must
    compute and return the gradient of the loss with respect to the input to the
    forward function.
    """
    L, x0 = ctx.saved_tensors
    grad_L = grad_x0 = None
    device = L.device

    if ctx.needs_input_grad[0]:
      grad_x = grad_output.clone()
      dim = L.shape[0]
      B = torch.matmul(grad_x, x0.t())
      K1 = torch.cat((L.T, B), 1)
      K2 = torch.cat((torch.zeros((dim, dim)).to(device), L.T), 1)
      K = torch.cat((K1, K2), 0)
      grad_L = torch.from_numpy(expm(K.detach().cpu().numpy())).to(device)[:dim, dim:]
      # grad_L = expm_taylor(K)[:dim, dim:]


    if ctx.needs_input_grad[1]:
       Ac = torch.from_numpy(expm(L.detach().cpu().numpy())).to(device)
      # Ac =expm_taylor(L)
       grad_x = grad_output.clone()
       grad_x0 = torch.matmul(Ac, grad_x)
     
    # print('>>full DL:', grad_L)
    return grad_L, grad_x0


class asinh(torch.autograd.Function):
  """
  We can implement our own custom autograd Functions by subclassing
  torch.autograd.Function and implementing the forward and backward passes
  which operate on Tensors.
  """
  @staticmethod
  def forward(ctx, x0):
    """
    In the forward pass we receive a context object and a Tensor containing the
    input; we must return a Tensor containing the output, and we can use the
    context object to cache objects for use in the backward pass.
    """
    #Calculate the forward value
    device = x0.device
    output = torch.from_numpy(np.arcsinh(x0.detach().cpu().numpy())).to(device)

    #Cache tensors for backward pass
    ctx.save_for_backward(x0)

    return output

  @staticmethod
  def backward(ctx, grad_output):
    """
    In the backward pass we receive the context object and a Tensor containing
    the gradient of the loss with respect to the output produced during the
    forward pass. We can retrieve cached data from the context object, and must
    compute and return the gradient of the loss with respect to the input to the
    forward function.
    """
    x0, = ctx.saved_tensors
    grad_x = grad_output.clone()
    grad_x0 = None

    if ctx.needs_input_grad[0]:
      grad_x0 = (1.0/torch.sqrt(1.0 +x0.pow(2)))*grad_x


    return grad_x0

# class expm_func(torch.autograd.Function):
#   """
#   We can implement our own custom autograd Functions by subclassing
#   torch.autograd.Function and implementing the forward and backward passes
#   which operate on Tensors.
#   """
#   @staticmethod
#   def forward(ctx, L, x0):
#     """
#     In the forward pass we receive a context object and a Tensor containing the
#     input; we must return a Tensor containing the output, and we can use the
#     context object to cache objects for use in the backward pass.
#     """
#     #Calculate the forward value
#     Ac = torch.from_numpy(expm(L.detach().numpy()))
#     output = torch.matmul(Ac, x0)
#
#     #Cache tensors for backward pass
#     ctx.save_for_backward(L, x0)
#
#     return output
#
#   @staticmethod
#   def backward(ctx, grad_output):
#     """
#     In the backward pass we receive the context object and a Tensor containing
#     the gradient of the loss with respect to the output produced during the
#     forward pass. We can retrieve cached data from the context object, and must
#     compute and return the gradient of the loss with respect to the input to the
#     forward function.
#     """
#     L, x0 = ctx.saved_tensors
#     grad_L = grad_x0 = None
#
#     if ctx.needs_input_grad[0]:
#       grad_x = grad_output.clone()
#       #B = -1.0 * torch.matmul(grad_x, x0.t())
#       #print("This is the grad_x matrix: ", grad_x)
#       #print("This is the x0 matrix: ", x0)
#       B = torch.matmul(grad_x, x0.t())
#       K1 = torch.cat((L.T, B), 1)
#       K2 = torch.cat((torch.zeros((2, 2)), L.T), 1)
#       K = torch.cat((K1, K2), 0)
#       #print("This is the K matrix: ", K)
#       grad_L = torch.from_numpy(expm(K.detach().numpy()))[:2, 2:]
#
#
#     if ctx.needs_input_grad[1]:
#       Ac = torch.from_numpy(expm(L.detach().numpy()))
#       grad_x = grad_output.clone()
#       grad_x0 = torch.matmul(Ac, grad_x)
#
#     return grad_L, grad_x0


# class expm_rot(torch.autograd.Function):
#   """
#   We can implement our own custom autograd Functions by subclassing
#   torch.autograd.Function and implementing the forward and backward passes
#   which operate on Tensors.
#   """
#   @staticmethod
#   def forward(ctx, L, x0):
#     """
#     In the forward pass we receive a context object and a Tensor containing the
#     input; we must return a Tensor containing the output, and we can use the
#     context object to cache objects for use in the backward pass.
#     """
#     #Calculate the forward value
#     Ac = torch.from_numpy(expm(L.detach().numpy() - L.detach().numpy().T))
#     output = torch.matmul(Ac, x0)
#
#     #Cache tensors for backward pass
#     ctx.save_for_backward(L, x0)
#
#     return output
#
#   @staticmethod
#   def backward(ctx, grad_output):
#     """
#     In the backward pass we receive the context object and a Tensor containing
#     the gradient of the loss with respect to the output produced during the
#     forward pass. We can retrieve cached data from the context object, and must
#     compute and return the gradient of the loss with respect to the input to the
#     forward function.
#     """
#     L, x0 = ctx.saved_tensors
#     grad_L = grad_x0 = None
#
#     if ctx.needs_input_grad[0]:
#       grad_x = grad_output.clone()
#       B = torch.matmul(grad_x, x0.t())
#       K1 = torch.cat(((L - L.t()).T, B), 1)
#       K2 = torch.cat((torch.zeros((2, 2)), (L - L.t()).T), 1)
#       K = torch.cat((K1, K2), 0)
#       grad_L1 = torch.from_numpy(expm(K.detach().numpy()))[:2, 2:]
#
#       B = torch.matmul(x0, grad_x.t())
#       K1 = torch.cat(((L - L.t()), B), 1)
#       K2 = torch.cat((torch.zeros((2, 2)), (L - L.t())), 1)
#       K = torch.cat((K1, K2), 0)
#       grad_L2 = torch.from_numpy(expm(K.detach().numpy()))[:2, 2:]
#
#       grad_L = grad_L1 - grad_L2
#
#
#     if ctx.needs_input_grad[1]:
#       Ac = torch.from_numpy(expm(L.detach().numpy()))
#       grad_x = grad_output.clone()
#       grad_x0 = torch.matmul(Ac, grad_x)
#
#     return grad_L, grad_x0


# class expm_func_old(torch.autograd.Function):
#   """
#   We can implement our own custom autograd Functions by subclassing
#   torch.autograd.Function and implementing the forward and backward passes
#   which operate on Tensors.
#   """
#   @staticmethod
#   def forward(ctx, L, x0):
#     """
#     In the forward pass we receive a context object and a Tensor containing the
#     input; we must return a Tensor containing the output, and we can use the
#     context object to cache objects for use in the backward pass.
#     """
#     #Calculate the forward value
#     Ac = torch.from_numpy(expm(L.detach().numpy()))
#     output = torch.matmul(Ac, x0)
#
#     #Cache tensors for backward pass
#     ctx.save_for_backward(L, x0)
#
#     return output
#
#   @staticmethod
#   def backward(ctx, grad_output):
#     """
#     In the backward pass we receive the context object and a Tensor containing
#     the gradient of the loss with respect to the output produced during the
#     forward pass. We can retrieve cached data from the context object, and must
#     compute and return the gradient of the loss with respect to the input to the
#     forward function.
#     """
#     L, x0 = ctx.saved_tensors
#     grad_L = grad_x0 = None
#
#     if ctx.needs_input_grad[0]:
#       grad_x = grad_output.clone()
#       B = torch.matmul(grad_x, x0.t())
#       deriv = torch.zeros(1)
#       dt = 1.0
#       for t in np.linspace(0,1-dt,np.int(1/dt)):
#         deriv = deriv + torch.matmul(torch.matmul(torch.from_numpy(expm(-L.detach().numpy().T * (1 - t- dt/2))), B),
#                                      torch.from_numpy(expm(L.detach().numpy().T * (t + dt / 2)) * dt))
#       grad_L = deriv
#
#     if ctx.needs_input_grad[1]:
#       Ac = torch.from_numpy(expm(L.detach().numpy()))
#       grad_x = grad_output.clone()
#       grad_x0 = torch.matmul(Ac, grad_x)
#
#     return grad_L, grad_x0
