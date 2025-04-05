# encoding: utf-8
'''
**Utils**
=========
'''

from __future__ import absolute_import

import sys
import numpy as np

from scipy.linalg import det
from scipy.misc import factorial
from scipy.spatial.distance import cdist, pdist
from scipy.sparse import dok_matrix


def get_subsimplices(simplices):
    '''
    Returns one-dimension lower simplices embedded in the given set of simplices.
    '''
    simplices = np.sort(simplices.copy(), axis=1)
    subsimplices = set()
    for j in np.arange(simplices.shape[1]):
        idx = list(range(simplices.shape[1]))
        idx.pop(j)
        subsimplices = subsimplices.union(set(tuple(sub) for sub in simplices.take(idx, axis=1)))
    subsimplices = np.array([ sub for sub in subsimplices])
    return subsimplices

def orient_simplices(points, simplices):
    '''
    Orient simplices.
    '''
    if simplices.shape[1] == 4 and points.shape[1]==3:
        for i in range(simplices.shape[0]):
            A = np.zeros((4,4))
            A[:,3] = 1
            for j in range(simplices.shape[1]):
                A[j,:-1] = points[simplices[i,j],:]
            if det(A) < 0:
                simplices[i] = simplices[i,[1,0,2,3]]
    return simplices                

def simpvol(points, simplices):

    ''' SIMPVOL Simplex volume.
        V=SIMPVOL(points, simplices)
        Copyright (C) 2004-2006 Per-Olof Persson. See COPYRIGHT.TXT for details.
    '''
    point_dim  =  points.shape[1]
    simp_dim = simplices.shape[1]
    # 1-simplex, edge in 1 dimension
    if simp_dim == 1:
        volume = np.ones(simplices.shape[0], dtype=int) 
    elif point_dim  ==  1 and simp_dim == 2:
        d12 = points[simplices[:,1],:] - points[simplices[:,0],:]
        volume = np.abs(d12)
    # 1-simplex, edge in any dimension
    elif simp_dim == 2:
        d12 = points[simplices[:,1],:] - points[simplices[:,0],:]
        volume = np.sqrt(np.sum(d12**2, axis=1))
    # 2-simplex, triangle in 2 dimension
    elif point_dim  ==  2 and simp_dim == 3:
        d12 = points[simplices[:,1],:] - points[simplices[:,0],:]
        d13 = points[simplices[:,2],:] - points[simplices[:,0],:]
        volume = (np.multiply(d12[:,0], d13[:,1]) - np.multiply(d12[:,1], d13[:,0]))*1.0/2
    # 2-simplex, triangle in 3 dimension
    elif point_dim == 3 and simp_dim == 3:
        d12 = points[simplices[:,1],:] - points[simplices[:,0],:]
        d13 = points[simplices[:,2],:] - points[simplices[:,0],:]
        volume = np.sqrt(np.sum(np.cross(d12, d13, axis=1)**2, axis=1))*1.0/2
    # 3-simplex, tetrahedra in 3 dimension
    elif point_dim  ==  3 and simp_dim == 4: # 
        '''
        d12 = points[simplices[:,1],:] - points[simplices[:,0],:]
        d13 = points[simplices[:,2],:] - points[simplices[:,0],:]
        d14 = points[simplices[:,3],:] - points[simplices[:,0],:]
        volume = np.dot(np.cross(d12,d13,axis=2),d14,axis=2)*1.0/6
        '''
        volume = np.zeros((simplices.shape[0],))
        for i in range(simplices.shape[0]):
            A = np.zeros((4,4))
            A[:,3] = 1
            for j in range(simplices.shape[1]):
                A[j,:-1] = points[simplices[i,j],:]
            volume[i] = np.abs(det(A))
        volume = volume/factorial(points.shape[1])
    # n-simplex in n-dimension
    else:
        volume = np.zeros((simplices.shape[0],))
        for i in range(simplices.shape[0]):
            A = np.zeros(points.shape[1] + 1)
            A[:,0] = 1
            for j in range(points.shape[1]+1):
                A[j,1:] = points[simplices[i,j],:]
            volume[i] = det(A)
        volume = volume/factorial(points.shape[1])
    return volume

def boundary_matrix(simplices, subsimplices, is_oriented=True, is_sparse=True, format='coo'):
    '''
    Builds a boundary matrix of given simplices. The format of a boundary matrix is as follows.
    boundary_matrix = (number of subsimplices) x (number of simplices)
    '''
    simplex_dim  = simplices.shape[1] 
    #if simplex_dim - subsimplices.shape[1] != 1:
    #    sys.stderr.write("Unable to build a boundary matrix. Please enter (d+1)-simplices and  d-subsimplices\n")
     #   exit()
    n_simplices = simplices.shape[0]
    m_subsimplices = subsimplices.shape[0] 
    if is_sparse:
        boundary_matrix = dok_matrix((m_subsimplices, n_simplices), dtype=np.int8) 
    else:
        boundary_matrix = np.array((m_subsimplices, n_simplices), dtype=np.int8) 
    if is_oriented:
        simplices_sort_idx = np.argsort(simplices)
        subsimplices_sort_idx = np.argsort(subsimplices)
        simplices_parity = permutationparity(simplices_sort_idx, 2)
        subsimplices_parity = permutationparity(subsimplices_sort_idx, 2)
    val = 1
    simplices = np.sort(simplices, axis=1)
    subsimplices = np.sort(subsimplices, axis=1)
    for i, simplex in enumerate(simplices):
        for j in np.arange(simplex_dim):
            idx = list(range(simplex_dim))
            idx.pop(j)
            subsimplex = simplex.take(idx)
            # to check the membership of subsimplex in subsimplices
            subsimplex_idx = np.argwhere((subsimplices==subsimplex).all(axis=1) == True)
            if subsimplex_idx.size == 0:
                sys.stderr.write("Unable to find subsimplex! Make sure subsimplices contains all boundary subsimplices\n")
                exit()
            subsimplex_idx = subsimplex_idx[0][0]
            if is_oriented:
                val = (-1)**((j + 1) + 1 + simplices_parity[i] + subsimplices_parity[subsimplex_idx])
            boundary_matrix[subsimplex_idx, i] = val
    if is_sparse:
        return boundary_matrix.asformat(format)
    else:
        return boundary_matrix

def boundary(simplex, idx=None):
    '''
    Returns simplex boundary as faces.
    if an index given, returns n-1 simplex by removing the element at the index.
    '''
    boundary = list()
    if idx == None:
        n = len(simplex)
        for i in xrange(0,n):
            face = list(simplex)
            face.pop(i)
            if (-1)**(i+2) < 0: 
                face = face[::-1]
            boundary.append(face)
    else:
        face = list(simplex)
        face.pop(idx)
        if (-1)**(idx+2) < 0: 
            face = face[::-1]
        boundary = face
    return boundary


def permutationparity(P, dim=None):
    '''
    permutationparity
       p = permutationparity(P,Dim)

       Returns the parity of a permutation. A value of 0 indicates an even 
       permutation, while a value of 1 indicates an odd permutation.

       The parity of permutations with duplicate values cannot be determined
       with this routine.

       `Dim` specifies which dimension the permutations are located on. 
       If Dim == 1 each column of the matrix `P` is treated as a permutation,
       If Dim == 2 each row of the matrix `P` is treated as a permutation.
       If Dim is empty or unspecified, if defaults to treating each column as 
       a different permutation.
       
       Example
           P = [1 2 3 5 4];            % An odd permutation
           pP = permutationparity(P)   % Get parity
           pP = 1

           P = [1 2 3 5 4; 1 2 4 5 3];     
           pP = permutationparity(P,2)
           pP = 1 
                0

           P = [1 2 3; 3 1 2; 2 3 1];  % Rows are odd, columns are even
           pP = permutationparity(P,1)
           pP = 0 0 0

           pP = permutationparity(P,2)
           pP = 1
                1
                1

       See also
           permutationparity perms randperm


        Reference
       'Pretty Permutation Parity Proof'
       Alan Macdonald
       Department of Mathematics
       Luther College,
       Decorah, IA 52101, U.S.A.
       macdonal@luther.edu
       June 16, 2004
       
       Let p be a permutation of [1 2 ... n]. Let G(p) be the number of times 
       in p that a number is greater than a number to its right. For example,
       G([2 4 1 3]) = 3. Note that G[1 2 ... n] = 0. 
       A transposition of numbers in adjacent positions changes G by ±1. Every 
       transposition can be expressed as a product of an odd number of such 
       transpositions. Therefore every transposition changes the parity of G. 
       Thus the number of transpositions used to obtain p is always even or 
       always odd, according as G(p) is even or odd.


     Author Information
     ------------------
     Pierce Brady
     Smart Systems Integration Group - SSIG
     Cork Institute of Technology, Ireland.


     License
     -------
     Copyright (c) 2010, Pierce
     All rights reserved.
     
     Redistribution and use in source and binary forms, with or without 
     modification, are permitted provided that the following conditions are 
     met:
     
         * Redistributions of source code must retain the above copyright 
           notice, this list of conditions and the following disclaimer.
         * Redistributions in binary form must reproduce the above copyright 
           notice, this list of conditions and the following disclaimer in 
           the documentation and/or other materials provided with the distribution
           
     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
     AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
     IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
     ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
     LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
     CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
     SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
     INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
     CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
     ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
     POSSIBILITY OF SUCH DAMAGE.
        '''
    nRows, mCols = P.shape
    if nRows == 1 and dim is None:
        dim = 2
    elif mCols == 1 and dim is None:
        dim = 1
    p = [0]
    if dim == 1:
        for i in np.arange(nRows):
            p = np.sum(np.tile(P[i,:].reshape(-1,1), (nRows-(i+1), 1)) > P[i+1:,:], 0) + p
    elif dim == 2:
        for j in np.arange(mCols):
            p = np.sum(np.tile(P[:, j].reshape(-1,1), (1, mCols-(j+1))) > P[:, j+1:], 1) + p
    p = np.mod(p, 2)
    return p

def edge_sign(simplex, edge):
    '''
    Given a triangle and its edge, return the induced orientation
    of the edge.
    '''
    for j in np.arange(3):
        idx = list(range(3))
        idx.pop(j)
        e = simplex.take(idx)
        flag = (-1)**j
        if flag == -1:
            e = e[::-1]
        if np.all(e == edge):
            return 1
        elif np.all(e==edge[::-1]):
            return -1

def check_orientation(simplices,edges):
    '''
        1. Create boundary matrix without any orientation with 0,1,-1 entries
            and use it to filter adjacent triangles and edges of a triangle
        2. Add all triangles to all_tris list
        3. Start with 0th triangle
        4. Propograte orientation to adjacent triangles:
            for each edge, find adjacent triangle and compute induced orientation of the edge on it.
                if adjacent triangle is not in oriented_tris
                    if induced orientations on the shared edge doesn't cancel each other,
                        do one swap on adjacent triangle and 
                        put corresponding orientation entry current triangle 
                        and edge in the boundary matrix
                    Add current and adjacent triangles to oriented_tris set

                if adjacent triangle is in oriented_tris 
                    check if the induced orientations on the shared edge are opposit,
                    then put the orientation entry in the boundary matrix position [edge, adj_traingle]
                    if not, return False
        5. Remove current triangle from all_tris
        6. Repeat 4 till all_tris list is empty(so all the triangles are checked)
        7. Return True
    '''
    triangles = simplices.copy()
    boundary_2 = boundary_matrix(triangles, edges, format="dok")
    all_tris = range(len(triangles))
    oriented_tris = set()
    tris = list()
    tris.append(0)
    #k = 0
    while len(all_tris) != 0:
        t_idx = tris.pop(0)
        all_tris.remove(t_idx)
        oriented_tris.add(t_idx)
        triangle = triangles[t_idx]
        e_indices =  [ int(idx) for idx in boundary_2[:,t_idx].nonzero()[0]]
        for edge_idx in e_indices:
            edge = edges[edge_idx]
            flag1 = edge_sign(triangle, edge)
            boundary_2[edge_idx, t_idx] = flag1
            adj_t =  [ int(idx) for idx in boundary_2[edge_idx,:].nonzero()[1] if idx!= t_idx]
            if len(adj_t) !=0:
                adj_t = adj_t[0]
                flag2 = edge_sign(triangles[adj_t], edge)
                if adj_t not in oriented_tris:
                    if flag2 == flag1:
                        #k += 1               
                        triangles[adj_t] = triangles[adj_t, [1,0,2]]
                        boundary_2[edge_idx,adj_t] = -flag1
                    oriented_tris.add(adj_t)
                    tris.append(adj_t)
                else:
                    if flag2 == flag1:
                        return False
                    else:
                        boundary_2[edge_idx,adj_t] = flag2
    #print "swappped ", k
    return True, triangles

if __name__ == '__main__':
    points = np.array([[0, 0, 0],[0,1,1],[0,2,0], [2,2,0]])
    simplices = np.array([[0, 1, 2], [1,2,3], [2, 4, 3]])
    simplices = np.array([[2, 1, 0], [3,1,2], [3, 2, 4]])
    permutationparity(simplices)
    simpvol1(points, simplices)

