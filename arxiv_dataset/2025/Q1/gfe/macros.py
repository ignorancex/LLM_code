#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:33:54 2020

@author: kflouris
"""
import torch

#---------------------------------------------------------------------------------------------------------------------------------
# Derivatives
#---------------------------------------------------------------------------------------------------------------------------------
# def jacobian(y: torch.Tensor, x: torch.Tensor, create_graph=False):
#     jac = []
#     flat_y = y.reshape(-1)
#     grad_y = torch.zeros_like(flat_y)
#     for i in range(len(flat_y)):
#         grad_y[i] = 1.
#         grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
#         jac.append(grad_x.reshape(x.shape))
#         grad_y[i] = 0.
#     return torch.stack(jac).reshape(y.shape + x.shape)

# # def hessian_old(y: torch.Tensor, x: torch.Tensor):
# #     return torch.einsum('ijik->ijk', jacobian(jacobian(y, x, create_graph=True), x))
# #     # return jacobian(jacobian(y, x, create_graph=True), x)
# # #   Hard coded contraction (works with [loss]==1) or byproduct of the jacobian fucntion?

# def hessian(y: torch.Tensor, x: torch.Tensor, v: torch.Tensor):
#     a = torch.einsum('ij,ij->i', v, jacobian(y, x, create_graph=True))
#     return torch.einsum('iij->ij', jacobian(a, x))    
     
# # def doublenabla_old(y: torch.Tensor, x: torch.Tensor , z: torch.Tensor):
# #     return jacobian(jacobian(y, x, create_graph=True), z)

# def doublenabla(y: torch.Tensor, x: torch.Tensor , z: torch.Tensor, v: torch.Tensor):
#     #print(jacobian(y, x, create_graph=True).shape)
#     #print(torch.einsum('ij, ij->i', v, jacobian(y, x, create_graph=True)).shape)
#     a = torch.einsum('ij, ij->i', v, jacobian(y, x, create_graph=True))
#     #print(jacobian(a, z).shape)
#     return torch.sum(jacobian(a, z), dim=0)
#     #return jacobian(jacobian(y, x, create_graph=True), z)
    
    
def jacobian(y: torch.Tensor, x: torch.Tensor, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)

def hessian(y: torch.Tensor, x: torch.Tensor):
    return torch.einsum('ijik->ijk', jacobian(jacobian(y, x, create_graph=True), x))
#   Hard coded contraction (works with [loss]==1) or byproduct of the jacobian fucntion?
    
def hessian_NEW(y: torch.Tensor, x: torch.Tensor, v: torch.Tensor):
    a = torch.einsum('ij,ij->i', v, jacobian(y, x, create_graph=True))
    return torch.einsum('iij->ij', jacobian(a, x))
     
def doublenabla_old(y: torch.Tensor, x: torch.Tensor , z: torch.Tensor):
    return jacobian(jacobian(y, x, create_graph=True), z)

def doublenabla(y: torch.Tensor, x: torch.Tensor , z: torch.Tensor, v: torch.Tensor):
    #print(jacobian(y, x, create_graph=True).shape)
    #print(torch.einsum('ij, ij->i', v, jacobian(y, x, create_graph=True)).shape)
    a = torch.einsum('ij, ij->i', v, jacobian(y, x, create_graph=True))
    #print(jacobian(a, z).shape)
    return torch.sum(jacobian(a, z), dim=0)
    #return jacobian(jacobian(y, x, create_graph=True), z)
