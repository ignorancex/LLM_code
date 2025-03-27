#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 18:58:15 2021

@author: danny
"""
import numpy as np
import math
from collections import defaultdict
from scipy import optimize
#from sdbox import sd_box 

class HALO():
    def __init__(self, f, bounds, max_feval, max_iter, beta, local_optimizer, verbose):
        self.f = f
        self.bounds = bounds
        self.max_feval = max_feval
        self.max_iter= max_iter
        self.feval = 0
        self.beta = beta
        self.local_optimizer = local_optimizer
        self.count_local = 0
        self.verbose = verbose
        self.G = []
        self.F = []
        self.v = []
        self.C = []
        self.S = []
        self.X = []
        self.Delta = []
        self.k = 0
        self.Y = []
        self.clustered = []
        self.X_local_hist = []
        self.tree = defaultdict(dict)
        self.neighbors = defaultdict(dict)
        self.levels = []
        self.indices_clustered = []
        self.F_hist = []
        self.X_hist = []
    
    def minimize(self):
        if self.beta > 1e-2 or self.beta < 1e-4:
            raise ValueError(
                "The parameter beta should be less than or equal to 1e-2 or greater than equal to 1e-4.")
        list_local_optimizers = ['L-BFGS-B', 'Nelder-Mead', 'TNC', 'Powell']
        if self.local_optimizer not in list_local_optimizers:
            raise ValueError(
                "Unrecognized local optimizer. Available local optimizers are: 'L-BFGS-B', 'Nelder-Mead', 'TNC' and 'Powell'.")
        global v0, n_rects, N
        if type(self.bounds) == list:
            self.bounds = np.asarray(self.bounds)
        x_L, x_U = self.bounds[:, 0], self.bounds[:, 1]  
        if (self.bounds[:, 0] > self.bounds[:, 1]).any():
            raise ValueError("One of the lower bounds is greater than an upper bound.")
        N = len(self.bounds)
        current_sampled = [[]]     
        n_rects = 1 # number of rectangles
        self.levels.append(0)
        self.C.append(np.ones(N) / 2.)
        x_c = x_L + self.C[0] * (x_U - x_L)
        self.feval += 1
        f_c = self.f(x_c)
        if self.verbose == 1:
            print('Iteration No.: ', self.k, '\n', 'Function Evaluation No.: ', self.feval, '\n',
                  'Performing Function Evaluation at: ', x_c, '\n', 
                  'Objective Function Value: ', f_c, '\n', sep="")
        self.X.append(x_c)
        self.neighbors[0][0] = [0]
        s0 = [0.5] * N
        I_star = [0]
        self.S.append(s0)
        v0 = np.sqrt(np.sum(np.array(s0) ** 2))
        self.v.append(v0)
        self.F.append(f_c)
        self.F_hist.append(f_c)
        self.X_hist.append(x_c)
        self.G.append([0.]*N)
        self.Delta.append([0.]*N)
        state, results = self.check()
        fathers = [0]
        self.k += 1
        if self.verbose == 1:
            print('Starting Iteration No.: ', self.k, 'Best Objective Function Value: ', min(self.F_hist))
        if state == 'continue':
            pass
        else:
            return results
        while self.k <= self.max_iter and self.feval <= self.max_feval:
            if self.k > 1:
                h = np.linalg.norm(self.G, axis=1)
                L_glob = max(h)
                l = self.compute_lipschitz(h, L_glob)
                I_star = self.selection(l, h)
            for i_star in I_star: 
                s_max = max(self.S[i_star]) 
                P = np.where(np.asarray(self.S[i_star]) == s_max)[0] 
                delta = 2. * s_max / 3.
                T = np.ones(N)*float("inf")
                G_new = []
                if len(P) == N: 
                    self.levels[i_star] += 1
                    level = self.levels[i_star]
                    self.neighbors[i_star][level]  = []
                    self.tree[i_star][level] = {}
                else:
                    level = self.levels[i_star]
                cn = []
                for i, p in enumerate(P): 
                    e_p = np.zeros(N)
                    e_p[p] = 1.
                    g_p1 = [0.]*N
                    C_p1 = self.C[i_star] + delta * e_p
                    x_p1 = x_L + C_p1 * (x_U - x_L)
                    self.feval += 1
                    f_p1 = self.f(x_p1)
                    if self.verbose == 1:
                        print('Iteration No: ', self.k, '\n', 'Function Evaluation No.: ', self.feval, '\n',
                              'Performing Function Evaluation at: ', x_p1, '\n', 
                              'Objective Function Value: ', f_p1, '\n', sep="")
                    state, results = self.check()
                    if state == 'continue':
                        pass
                    else:
                        return results
                    self.C.append(C_p1)
                    self.F.append(f_p1)
                    self.F_hist.append(f_p1)
                    self.X.append(x_p1)
                    self.X_hist.append(x_p1)
                    self.levels.append(level)
                    j = len(self.X)-1
                    cn.append(j)
                    current_sampled.append([i_star])
                    child_info = (p, '+', delta)
                    father_info = (p, '-', delta)
                    self.neighbors[i_star][level].append(j)
                    self.neighbors[j][level] = []
                    self.neighbors[j][level].append(i_star)
                    self.tree[i_star][level][j] = child_info
                    self.tree[j][level] = {i_star:father_info}
                    fathers.append(i_star)
                    g_p1[p] = abs(f_p1 - self.F[i_star]) / (delta + 1e-8)
                    G_new.append(g_p1)
                    g_p2 = [0.]*N
                    C_p2 = self.C[i_star] - delta * e_p
                    x_p2 = x_L + C_p2 * (x_U - x_L)
                    self.feval += 1
                    f_p2 = self.f(x_p2)
                    if self.verbose == 1:
                        print('Iteration No: ', self.k, '\n', 'Function Evaluation No.: ', self.feval, '\n',
                              'Performing Function Evaluation at: ', x_p2, '\n', 
                              'Objective Function Value: ', f_p2, '\n', sep="")
                    state, results = self.check()
                    if state == 'continue':
                        pass
                    else:
                        return results
                    self.levels.append(level)
                    self.C.append(C_p2)
                    self.F.append(f_p2)
                    self.F_hist.append(f_p2)
                    self.X.append(x_p2)
                    self.X_hist.append(x_p2)
                    j = len(self.X)-1
                    cn.append(j)
                    current_sampled.append([i_star])
                    child_info = (p, '-', delta)
                    father_info = (p, '+', delta)
                    self.neighbors[i_star][level].append(j)
                    self.neighbors[j][level] = []
                    self.neighbors[j][level].append(i_star)
                    self.tree[i_star][level][j] = child_info
                    self.tree[j][level] = {i_star:father_info}
                    fathers.append(i_star)
                    g_p2[p] = abs(self.F[i_star] - f_p2) / (delta + 1e-8)
                    G_new.append(g_p2)
                    self.G[i_star][p] = abs(f_p1 - f_p2) / (2 * delta + 1e-8)
                    T[i] = min(f_p1, f_p2)    
                current_sampled[i_star] = cn
                for g_new in G_new:
                    indices_zeros_g_new = np.where(np.array(g_new)==0.)[0]
                    for index_zero in indices_zeros_g_new:
                        g_new[index_zero] = self.G[i_star][index_zero]
                    self.G.append(g_new)
                U = np.argsort(T)[:len(P)]
                dict_dv = {}
                dict_l = {}
                for u in U:
                    m = P[u]
                    self.S[i_star][m] = delta / 2.
                    s1, s2 = list(self.S[i_star]), list(self.S[i_star])
                    dist_vert = math.sqrt(sum([elem**2 for elem in self.S[i_star]]))
                    self.v[i_star] = dist_vert
                    dict_l[u] = [s1]
                    dict_l[u].append(s2)
                    dict_dv[u] = [dist_vert]
                    dict_dv[u].append(dist_vert)    
                for i in range(len(P)):# reordering 
                    for ss in dict_l[i]:
                        self.S.append(ss)
                    for ddv in dict_dv[i]:
                        self.v.append(ddv)
                n_rects = n_rects + 2 * len(P)
                
            # In this part of the code, I made the tentative to compute the neighbors along the coordinate axis of each point 
            # sampled so far by HALO without computing the distances direclty which can be expensive. In this way, the approximation of the gradient through
            # finite difference is adaptevely updated not only with respect to the selected partitions (and the points sampled inside them)
            # but also to all the other points.
            # The code seems working but in case you wanna double check you can can uncomment those lines where is written 'for debugging'
            # you can also analyse the dictionary 'tree' and 'neighbors' to understand if the neighbors are located correctly.
            
            for ir in I_star:
                if ir == 0 and self.k == 1:
                    continue
                current_level_ir = self.levels[ir] #current level of ir
                currents_sampled_ir = current_sampled[ir] # indices of the partitons sampled inside ir
                father_ir = fathers[ir] #the father of the partition ir
                current_neighbors_ir = self.neighbors[ir][current_level_ir] # current neighbors of ir
                old_neighbors_ir = list(set(current_neighbors_ir)-set(currents_sampled_ir) ) # old neighbors of ir

                if father_ir in current_neighbors_ir: # ir is an hyperrectangle
                    
                    dim_ir, sign_ir, delta_ir = self.tree[father_ir][current_level_ir][ir]
                    
                    for old_neighbor_ir in old_neighbors_ir:
                        dim_old_neighbor_ir, sign_old_neighbor_ir, delta_old_neighbor_ir = self.tree[ir][current_level_ir][old_neighbor_ir]
                        neigbors_of_old_neighbor_current_level_ir = self.neighbors[old_neighbor_ir][current_level_ir]
                        
                        for neighbor_of_old_neighbor_ir in neigbors_of_old_neighbor_current_level_ir:
                            
                            dim_neighbor_of_old_neighbor_ir, sign_neighbor_of_old_neighbor_ir, delta_neighbor_of_old_neighbor_ir = self.tree[old_neighbor_ir][current_level_ir][neighbor_of_old_neighbor_ir]
    
                            for current_sampled_ir in currents_sampled_ir:
                                
                                dim_current_sampled_ir, sign_current_sampled_ir, delta_current_sampled_ir= self.tree[ir][current_level_ir][current_sampled_ir]
                                
                                if current_sampled_ir!= old_neighbor_ir and dim_current_sampled_ir == dim_neighbor_of_old_neighbor_ir and sign_neighbor_of_old_neighbor_ir == sign_current_sampled_ir:
                                    
                                    if old_neighbor_ir == father_ir:
                                        delta = delta_ir
                                        if sign_ir == '-':
                                            new_sign = '+'
                                        else:
                                            new_sign = '-'
                                        
                                        if neighbor_of_old_neighbor_ir not in self.neighbors[current_sampled_ir][current_level_ir]:
                                            
                                            #updating current_sampled_ir with neighbor_of_old_neighbor_ir
                                            self.neighbors[current_sampled_ir][current_level_ir].append(neighbor_of_old_neighbor_ir)
                                            self.tree[current_sampled_ir][current_level_ir][neighbor_of_old_neighbor_ir] = (dim_ir, new_sign, delta)
                                            # for debugging
                                            #new_delta_check = np.linalg.norm(self.C[current_sampled_ir] - self.C[neighbor_of_old_neighbor_ir])
                                            #if abs(new_delta_check - delta) > 1e-10:
                                            #    print('check!', new_delta_check, delta)
                                            #time.sleep(10)
                                            
                                            if new_sign == '-':
                                                self.G[current_sampled_ir][dim_ir] = abs(self.F[current_sampled_ir] - self.F[neighbor_of_old_neighbor_ir]) / (delta + 1e-8)
                                            else:
                                                self.G[current_sampled_ir][dim_ir] = abs(self.F[neighbor_of_old_neighbor_ir] - self.F[current_sampled_ir]) / (delta + 1e-8)
                                                
                                        
                                        if current_sampled_ir not in self.neighbors[neighbor_of_old_neighbor_ir][current_level_ir]:
                                            #updating neighbor_of_old_neighbor_ir with current_sampled_ir
                                            self.neighbors[neighbor_of_old_neighbor_ir][current_level_ir].append(current_sampled_ir)
                                            self.tree[neighbor_of_old_neighbor_ir][current_level_ir][current_sampled_ir] = (dim_ir, sign_ir, delta)
                                            # for debugging
                                            # new_delta_check = np.linalg.norm(self.C[current_sampled_ir] - self.C[neighbor_of_old_neighbor_ir])
                                            # if abs(new_delta_check - delta) > 1e-10:
                                            #     print('check!', new_delta_check, 2/3 * delta_old_neighbor_ir)
                                            #     time.sleep(10)
                                            if new_sign == '-':
                                                self.G[neighbor_of_old_neighbor_ir][dim_ir] = abs(self.F[current_sampled_ir] - self.F[neighbor_of_old_neighbor_ir]) / (delta + 1e-8)
                                            else:
                                                self.G[neighbor_of_old_neighbor_ir][dim_ir] = abs(self.F[neighbor_of_old_neighbor_ir] - self.F[current_sampled_ir]) / (delta + 1e-8)
                                    else:
                                        try:
                                            neighbors_next_level_of_neighbor = self.neighbors[neighbor_of_old_neighbor_ir][current_level_ir + 1]
                                            father_neighbor_of_old_neighbor_ir = fathers[neighbor_of_old_neighbor_ir]
                                            if len(neighbors_next_level_of_neighbor) == 2*N and father_neighbor_of_old_neighbor_ir not in neighbors_next_level_of_neighbor:
                                                #index neighbor_of_old_neighbor_ir has other neighbors along all the coordinates on the next level saved in neighbors_next_level_of_neighbor
                                                for neighbor_next_level_of_neighbor in neighbors_next_level_of_neighbor:
                                                    dim_next_level_neighbor_of_old_neighbor_ir, sign_next_level_neighbor_of_old_neighbor_ir, delta_next_level_neighbor_of_old_neighbor_ir = self.tree[neighbor_of_old_neighbor_ir][current_level_ir + 1][neighbor_next_level_of_neighbor]
                                                    if dim_old_neighbor_ir == dim_next_level_neighbor_of_old_neighbor_ir  and sign_old_neighbor_ir != sign_next_level_neighbor_of_old_neighbor_ir:
                                                     
                                                        #updating current_sampled_ir with neighbor_next_level_of_neighbor
                    
                                                        delta = delta_old_neighbor_ir - delta_next_level_neighbor_of_old_neighbor_ir
                                                        # for debugging
                                                        # new_delta_check = np.linalg.norm(self.C[neighbor_next_level_of_neighbor] - self.C[current_sampled_ir])
                                                        # if abs(new_delta_check - delta) > 1e-10:
                                                        #     print('check!', new_delta_check, delta)
                                                        #     time.sleep(10)
                                                        if sign_next_level_neighbor_of_old_neighbor_ir == '-':
                                                            new_sign = '+'
                                                        else:
                                                            new_sign = '-'
                                                        if current_level_ir + 1 in self.neighbors[current_sampled_ir].keys():
                                                            self.neighbors[current_sampled_ir][current_level_ir+1].append(neighbor_next_level_of_neighbor)
                                                            self.tree[current_sampled_ir][current_level_ir+1][neighbor_next_level_of_neighbor] = (dim_next_level_neighbor_of_old_neighbor_ir,new_sign, delta)
                                                            if new_sign == '-':
                                                                self.G[current_sampled_ir][dim_next_level_neighbor_of_old_neighbor_ir] = abs(self.F[current_sampled_ir] - self.F[neighbor_next_level_of_neighbor]) / (delta + 1e-8)
                                                            else:
                                                                self.G[current_sampled_ir][dim_next_level_neighbor_of_old_neighbor_ir] = abs(self.F[neighbor_next_level_of_neighbor] - self.F[current_sampled_ir]) / (delta + 1e-8)
                                                            break
                                                        else:
                                                            self.neighbors[current_sampled_ir][current_level_ir+1] = [neighbor_next_level_of_neighbor]
                                                            self.tree[current_sampled_ir][current_level_ir+1]={neighbor_next_level_of_neighbor : (dim_next_level_neighbor_of_old_neighbor_ir,new_sign, delta)}
                                                            if new_sign == '-':
                                                                self.G[current_sampled_ir][dim_next_level_neighbor_of_old_neighbor_ir] = abs(self.F[current_sampled_ir] - self.F[neighbor_next_level_of_neighbor]) / (delta + 1e-8)
                                                            else:
                                                                self.G[current_sampled_ir][dim_next_level_neighbor_of_old_neighbor_ir] = abs(self.F[neighbor_next_level_of_neighbor] - self.F[current_sampled_ir]) / (delta + 1e-8)
                                                            break
                                            else:
                                                delta = delta_old_neighbor_ir
                                                if sign_old_neighbor_ir == '-':
                                                    new_sign = '+'
                                                else:
                                                    new_sign = '-'
                                                # for debugging
                                                # new_delta_check = np.linalg.norm(self.C[current_sampled_ir] - self.C[neighbor_of_old_neighbor_ir])
                                                # if abs(new_delta_check - delta) > 1e-10:
                                                #     print('check!', new_delta_check, delta)
                                                #     time.sleep(10)
                                            
                                                if neighbor_of_old_neighbor_ir not in self.neighbors[current_sampled_ir][current_level_ir]:
                                                    #updating current_sampled_ir with neighbor_of_old_neighbor_ir
                                                    self.neighbors[current_sampled_ir][current_level_ir].append(neighbor_of_old_neighbor_ir)
                                                    self.tree[current_sampled_ir][current_level_ir][neighbor_of_old_neighbor_ir] = (dim_old_neighbor_ir, sign_old_neighbor_ir, delta)
                                                    if new_sign == '-':
                                                        
                                                        self.G[current_sampled_ir][dim_old_neighbor_ir] = -(self.F[neighbor_of_old_neighbor_ir] - self.F[current_sampled_ir]) / (delta + 1e-8)
                                                    else:
                                                        self.G[current_sampled_ir][dim_old_neighbor_ir] = -(self.F[current_sampled_ir] - self.F[neighbor_of_old_neighbor_ir]) / (delta + 1e-8)
                                                
                                                if current_sampled_ir not in self.neighbors[neighbor_of_old_neighbor_ir][ current_level_ir]:
                                                    #updating current_sampled_ir with neighbor_of_old_neighbor_ir
                                                    self.neighbors[neighbor_of_old_neighbor_ir][current_level_ir].append(current_sampled_ir)
                                                    self.tree[neighbor_of_old_neighbor_ir][current_level_ir][current_sampled_ir] = (dim_old_neighbor_ir, new_sign, delta)
                                                    if new_sign == '-':
                                                        
                                                        self.G[neighbor_of_old_neighbor_ir][dim_old_neighbor_ir] = abs(self.F[neighbor_of_old_neighbor_ir] - self.F[current_sampled_ir]) / (delta + 1e-8)
                                                    else:
                                                        self.G[neighbor_of_old_neighbor_ir][dim_old_neighbor_ir] = abs(self.F[current_sampled_ir] - self.F[neighbor_of_old_neighbor_ir]) / (delta + 1e-8)
                                                    
                                        except KeyError:
                                            delta = delta_old_neighbor_ir
                                            if sign_old_neighbor_ir == '-':
                                                new_sign = '+'
                                            else:
                                                new_sign = '-'
                                            # for debugging
                                            # new_delta_check = np.linalg.norm(self.C[current_sampled_ir] - self.C[neighbor_of_old_neighbor_ir])
                                            # if abs(new_delta_check - delta) > 1e-10:
                                            #     print('check!', new_delta_check, delta_old_neighbor_ir)
                                            #     time.sleep(10)
                                            
                                            if neighbor_of_old_neighbor_ir not in self.neighbors[current_sampled_ir][current_level_ir]:
                                                #updating neighbor_of_old_neighbor_ir with current_sampled_ir
                                                self.neighbors[current_sampled_ir][current_level_ir].append(neighbor_of_old_neighbor_ir)
                                                self.tree[current_sampled_ir][current_level_ir][neighbor_of_old_neighbor_ir] = (dim_old_neighbor_ir, sign_old_neighbor_ir, delta)
                                                if new_sign == '-':
                                                        
                                                    self.G[current_sampled_ir][dim_old_neighbor_ir] = abs(self.F[neighbor_of_old_neighbor_ir] - self.F[current_sampled_ir]) / (delta + 1e-8)
                                                else:
                                                    self.G[current_sampled_ir][dim_old_neighbor_ir] = abs(self.F[current_sampled_ir] - self.F[neighbor_of_old_neighbor_ir]) / (delta + 1e-8)
                                            if current_sampled_ir not in self.neighbors[neighbor_of_old_neighbor_ir][current_level_ir]:
                                                #updating current_sampled_ir with neighbor_of_old_neighbor_ir 
                                                self.neighbors[neighbor_of_old_neighbor_ir][current_level_ir].append(current_sampled_ir)
                                                self.tree[neighbor_of_old_neighbor_ir][current_level_ir][current_sampled_ir] = (dim_old_neighbor_ir, new_sign, delta_old_neighbor_ir)
                                                if new_sign == '-':
                                                        
                                                    self.G[neighbor_of_old_neighbor_ir][dim_old_neighbor_ir] = abs(self.F[neighbor_of_old_neighbor_ir] - self.F[current_sampled_ir]) / (delta + 1e-8)
                                                else:
                                                    self.G[neighbor_of_old_neighbor_ir][dim_old_neighbor_ir] = abs(self.F[current_sampled_ir] - self.F[neighbor_of_old_neighbor_ir]) / (delta + 1e-8)
                else:
                    old_neighbors_ir = self.neighbors[ir][current_level_ir-1]
                    for old_neighbor_ir in old_neighbors_ir:
                        
                        dim_old_neighbor_ir, sign_old_neighbor_ir, delta_old_neighbor_ir = self.tree[ir][current_level_ir-1][old_neighbor_ir]
                        
                        for current_neighbor_ir in current_neighbors_ir:

                            dim_current_neighbor_ir, sign_current_neighbor_ir, delta_current_neighbor_ir = self.tree[ir][current_level_ir][current_neighbor_ir]
                                
                            if dim_current_neighbor_ir == dim_old_neighbor_ir and sign_old_neighbor_ir == sign_current_neighbor_ir:
                                
                                try:
                                    neighbors_of_old_neighbor_ir = self.neighbors[old_neighbor_ir][current_level_ir]
                                    father_old_neighbor_ir = fathers[old_neighbor_ir]
                                    if len(neighbors_of_old_neighbor_ir) == 2*N and father_old_neighbor_ir not in neighbors_of_old_neighbor_ir:
                                        for neighbor_of_old_neighbor_ir in neighbors_of_old_neighbor_ir:
                                            
                                            dim_neighbor_of_old_neighbor_ir, sign_neighbor_of_old_neighbor_ir, delta_neighbor_of_old_neighbor_ir = self.tree[old_neighbor_ir][current_level_ir][neighbor_of_old_neighbor_ir]
                                            
                                            
                                            if dim_old_neighbor_ir == dim_neighbor_of_old_neighbor_ir and sign_old_neighbor_ir != sign_neighbor_of_old_neighbor_ir:
                                                # updating current_neighbor_ir with neighbor_of_old_neighbor_ir
                                            
                                                delta = delta_current_neighbor_ir
                                                # new_delta_check = np.linalg.norm(self.C[current_neighbor_ir] - self.C[neighbor_of_old_neighbor_ir])
                                                # for debugging
                                                # new_delta = new_delta_check
                                                # if abs(new_delta_check - delta) > 1e-10:
                                                #     print('check!', new_delta_check, delta_old_neighbor_ir)
                                                #      time.sleep(10)
                                                
                                                self.neighbors[current_neighbor_ir][current_level_ir].append(neighbor_of_old_neighbor_ir)
                                                self.neighbors[neighbor_of_old_neighbor_ir][current_level_ir].append(current_neighbor_ir)
                                                self.tree[current_neighbor_ir][current_level_ir][neighbor_of_old_neighbor_ir] = (dim_old_neighbor_ir, sign_old_neighbor_ir, delta)
                                                self.tree[neighbor_of_old_neighbor_ir][current_level_ir][current_neighbor_ir] = (dim_old_neighbor_ir, sign_neighbor_of_old_neighbor_ir, delta)
                                                if sign_old_neighbor_ir == '-':
                                                    self.G[current_neighbor_ir][dim_old_neighbor_ir] = abs(self.F[current_neighbor_ir] - self.F[neighbor_of_old_neighbor_ir]) / (delta + 1e-8)
                                                    self.G[neighbor_of_old_neighbor_ir][dim_old_neighbor_ir] = abs(self.F[current_neighbor_ir] - self.F[neighbor_of_old_neighbor_ir]) / (delta + 1e-8)
                                                else:
                                                    self.G[current_neighbor_ir][dim_old_neighbor_ir] = abs(self.F[neighbor_of_old_neighbor_ir] - self.F[current_neighbor_ir]) / (delta_current_neighbor_ir + 1e-8)
                                                    self.G[neighbor_of_old_neighbor_ir][dim_old_neighbor_ir] = abs(self.F[neighbor_of_old_neighbor_ir] - self.F[current_neighbor_ir]) / (delta_current_neighbor_ir + 1e-8)
                                                break
                                    else:
                                        #updating old_neighbor_ir with current_neighbor_ir
                                        delta = 2/3 * delta_old_neighbor_ir
                                        if sign_old_neighbor_ir == '-':
                                            new_sign = '+'
                                        else:
                                            new_sign = '-'
                                        # for debugging
                                        # new_delta_check = np.linalg.norm(self.C[old_neighbor_ir] - self.C[current_neighbor_ir])
                                        # if abs(new_delta_check - delta) > 1e-10:
                                        #     print('check!', new_delta_check, delta)
                                        #     time.sleep(10)
                                        if sign_old_neighbor_ir == '+':
                                            self.G[old_neighbor_ir][dim_old_neighbor_ir] = abs(self.F[old_neighbor_ir] - self.F[current_neighbor_ir]) / (delta + 1e-8)
                                        else:
                                            self.G[old_neighbor_ir][dim_old_neighbor_ir] = abs(self.F[current_neighbor_ir] - self.F[old_neighbor_ir]) / (delta + 1e-8)
                                            
                                        if current_level_ir not in self.neighbors[old_neighbor_ir].keys():
                                            self.neighbors[old_neighbor_ir][current_level_ir] = [current_neighbor_ir]
                                            self.tree[old_neighbor_ir][current_level_ir]= {current_neighbor_ir:(dim_old_neighbor_ir, new_sign, delta)}
                                            
                                        else:
                                            self.neighbors[old_neighbor_ir][current_level_ir].append(current_neighbor_ir)
                                            self.tree[old_neighbor_ir][current_level_ir][current_neighbor_ir] = (dim_old_neighbor_ir, new_sign, delta)
                                        
                                except KeyError:
                                    #updating old_neighbor_ir with current_neighbor_ir
                                    delta = 2/3 * delta_old_neighbor_ir
                                    if sign_old_neighbor_ir == '-':
                                        new_sign = '+'
                                    else:
                                        new_sign = '-'
                                    # for debugging
                                    #new_delta_check = np.linalg.norm(self.C[old_neighbor_ir] - self.C[current_neighbor_ir])
                                    # if abs(new_delta_check - delta) > 1e-10:
                                    #     print('check!', new_delta_check, delta)
                                    #     time.sleep(10)
                                    if sign_old_neighbor_ir == '+':
                                        self.G[old_neighbor_ir][dim_old_neighbor_ir] = abs(self.F[old_neighbor_ir] - self.F[current_neighbor_ir]) / (delta + 1e-8)
                                    else:
                                        self.G[old_neighbor_ir][dim_old_neighbor_ir] = abs(self.F[current_neighbor_ir] - self.F[old_neighbor_ir]) / (delta + 1e-8)
                                    if current_level_ir not in self.neighbors[old_neighbor_ir].keys():
                                        self.neighbors[old_neighbor_ir][current_level_ir] = [current_neighbor_ir]
                                        self.tree[old_neighbor_ir][current_level_ir]= {current_neighbor_ir:(dim_old_neighbor_ir, new_sign, delta)}
                                    else:
                                        self.neighbors[old_neighbor_ir][current_level_ir].append(current_neighbor_ir)
                                        self.tree[old_neighbor_ir][current_level_ir][current_neighbor_ir] = (dim_old_neighbor_ir, new_sign, delta)
                                        
            self.k += 1
            if self.verbose == 1:
                print('Starting Iteration No.: ', self.k,'\n', 
                      'Best Objective Function Value ', min(self.F_hist), sep="")
        state, results = self.check()
        return results
    
    def check(self):
        if self.feval >= self.max_feval or self.k >= self.max_iter:
            dict_results = {}
            dict_results['F_history'] = self.F_hist
            dict_results['X_history'] = self.X_hist
            dict_results['F_history_global'] = self.F
            dict_results['X_history_global'] = self.X
            dict_results['C_history_global'] = self.C
            dict_results['X_history_local'] = self.X_local_hist
            dict_results['count_local'] = self.count_local
            dict_results['gradients'] = self.G
            dict_results['sides'] = self.S
            dict_results['tree'] = self.tree
            dict_results['neighbors'] = self.neighbors
            best_feval = np.argmin(np.asarray(self.F_hist))
            dict_results['best_feval'] = best_feval
            best_f = self.F_hist[best_feval]
            dict_results['best_f'] = best_f
            best_x = self.X_hist[best_feval]
            dict_results['best_x'] = best_x
            print('Global Optimization Procedure Completed')
            return ('Terminate', dict_results)
        else:
            return ('continue', None)
    
    def compute_lipschitz(self, h, L_glob):
        alpha_vector = (np.array(self.v) / v0)
        l = alpha_vector * np.array([L_glob] * len(self.v)) + h * (1 - alpha_vector)
        return l
        
    def init_selection(self): 
        I_star = [] 
        v_max = max(self.v)
        for index in range(len(self.v)):
            if self.v[index] == v_max:
                I_star.append(index)
        return I_star
    
    def selection(self, l, h):
        I_star = []
        d_max = max(self.v)
        lower_bounds = np.array(self.F) - np.array(self.v) * l
        indices_max_rects = np.where(abs(np.array(self.v) - d_max) <= 1e-10)[0]
        best_lb_index = np.argmin(lower_bounds)
        best_f_index = np.argmin(self.F)
        best_lb_dmax_index = indices_max_rects[np.argmin(lower_bounds[indices_max_rects])]
        I_star.append(best_lb_dmax_index)
        if self.v[best_lb_index] > 1e-8 and best_lb_index not in self.indices_clustered:
            if self.v[best_lb_index] <= self.beta:
                self.local_search(best_lb_index)
            else:
                I_star.append(best_lb_index)
        if self.v[best_f_index] > 1e-8 and best_f_index not in self.indices_clustered:
            if self.v[best_f_index] <= self.beta:
                self.local_search(best_f_index)
            else:
                I_star.append(best_f_index)
        I_star = np.unique(I_star) 
        return I_star   
    
    def local_search(self, index_x0):
        if self.compute_neighbors(index_x0) == True:
            self.count_local += 1
            x0 = self.X[index_x0]
            if self.verbose == 1:
                print('Starting Local Optimization Routine', '\n', 
                      'Local Optimization Method: ', self.local_optimizer, '\n', 
                      'Starting Point x0: ', x0, sep="")
            max_local_feval = self.max_feval - self.feval
            # if self.local_optimizer == 'SDBOX':
            #     x0 = list(x0)
            #     maxiter = 20000
            #     num_funct = 0
            #     nf_max = max_local_feval
            #     iprint = 0
            #     alfa_stop   = 1e-6
            #     x_min_local, f_min_local, n_fev_local = sd_box(N,x0,self.f,list(self.bounds[:, 0]), list(self.bounds[:, 1])
            #                                                    ,alfa_stop,nf_max,maxiter,num_funct,iprint)
            #     self.feval = self.feval + n_fev_local
            #     self.F_hist.append(f_min_local)
            #     self.X_hist.append(x_min_local)

            res = optimize.minimize(self.f, x0, method=self.local_optimizer,
                      bounds=self.bounds, options={'maxfun': max_local_feval})
            f_min_local = res.fun
            n_fev_local = res.nfev
            x_min_local = res.x
            self.feval = self.feval + n_fev_local
            self.F_hist = self.F_hist + [float('+inf')] * (n_fev_local - 1) + [f_min_local]
            self.X_hist = self.X_hist + [float('+inf')] * (n_fev_local - 1) + [x_min_local]
            state, results = self.check()
            if state == 'continue':
                pass
            else:
                return results
            if self.verbose == 1:
                print('Local Optimization Routine Terminated', '\n', 
                      'Objective Function Value: ', f_min_local, '\n', 
                      'No. Functions Evaluations Performed by the local optimizer: ', 
                      n_fev_local, '\n',
                      sep="")
        else:
            return None
    
    def compute_neighbors(self, index_x0):
        radius = 1e-4
        if not self.indices_clustered:
            c0 = self.C[index_x0]
            list_nearest_c0 = [self.C[i] for i in range(len(self.C)) if np.linalg.norm(c0 - self.C[i]) < radius]
            list_indices_nearest_c0 = [i for i in range(len(self.C)) if np.linalg.norm(c0 - self.C[i]) < radius]
            for i, elem in enumerate(list_nearest_c0):
                self.clustered.append(elem)
                self.indices_clustered.append(list_indices_nearest_c0[i])
            return True
        c0 = self.C[index_x0]
        list_indices_clustered_nearest_c0 = [index for index in self.indices_clustered if np.linalg.norm(c0 - self.C[index]) < radius]
        if not list_indices_clustered_nearest_c0:
            list_nearest_c0 = [self.C[i] for i in range(len(self.C)) if np.linalg.norm(c0 - self.C[i]) < radius]
            list_indices_nearest_c0 = [i for i in range(len(self.C)) if np.linalg.norm(c0 - self.C[i]) < radius]
            for i, elem in enumerate(list_nearest_c0):
                self.clustered.append(elem)
                self.indices_clustered.append(list_indices_nearest_c0[i])
            return True
        else:
            self.clustered.append(self.C[index_x0])
            self.indices_clustered.append(index_x0)
            return False
