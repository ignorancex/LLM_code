#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 17:03:29 2019

@author: sotzee
"""

import scipy.optimize as opt
import numpy as np

def logic_no(sol_x,init_i,args,extra_args):
    return True

def solve_equations(equations,init_list,args,vary_list=np.linspace(1.,1.,1),tol=1e-20,logic_success_f=logic_no,equations_extra_args=[]):
    shape_init=np.array(init_list).shape
    init_vary_list=np.multiply(init_list,np.tile(np.array(np.meshgrid(*([vary_list]*shape_init[1]))), (shape_init[0],)+(1,)*(shape_init[1]+1)).transpose(list(range(2,shape_init[1]+2))+list(range(2)))).reshape((len(vary_list)**shape_init[1]*shape_init[0],shape_init[1]))
    for init_i in init_vary_list:
        sol = opt.root(equations,init_i,tol=tol,args=(args,equations_extra_args),method='hybr')
        if (sol.success and logic_success_f(sol.x,init_i,args,equations_extra_args)):
            return True,list(sol.x)
    return False,list(sol.x)