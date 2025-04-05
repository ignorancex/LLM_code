# encoding: utf-8
'''
----------------------
Median surface demo 3D
−−−−−−−−−−−−−−−−--−−−−
'''
from __future__ import absolute_import
import os
import importlib
import time
from shutil import copyfile

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from medianshape.core import median
from medianshape import utils, inout

def load_tetgen_mesh(fname):
    '''
    Loads .node, .face, .ele files. Usually their indexing starts from 1.
    '''
    # point1, point2, point3
    points = np.loadtxt("%s.node"%fname, dtype=float, skiprows=1)[:,1:4]
    # point1, point2, point3, label
    triangles = np.loadtxt("%s.face"%fname, dtype=int, skiprows=1)[:,1:5]
    # point1, point2, point3, point4
    tetras = np.loadtxt("%s.ele"%fname, dtype=int, skiprows=1)[:,1:5]
    return points, triangles, tetras

def save_tetgen_mesh(t, q, r, fname):
    points = np.loadtxt("%s.node"%fname, dtype=float, skiprows=1)[:,1:4]
    triangles = np.loadtxt("%s.face"%fname, dtype=int, skiprows=1)
    tetrahedras= np.loadtxt("%s.ele"%fname, dtype=int, skiprows=1)
    t_idx = np.argwhere(t!=0).reshape(-1,)
    t_flag = -3
    q_flag = -41
    r_flag = -51
    print triangles.shape
    overlaps = []
    if len(t_idx)!=0:
        tris = triangles.copy()
        for t in t_idx:
            if tris[t,4] == -1 or tris[t,4]==-2:
                overlaps.append(tris[t][1:4]-1)
                print "overlapping"
        tris[t_idx, 4] = t_flag
        t_tris = tris[t_idx][:,1:4] -1
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=t_tris)
        ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=overlaps)
       # plt.title("T")
        plt.show()
        copyfile("%s.node"%fname, "%s.1.node"%fname)
        copyfile("%s.ele"%fname, "%s.1.ele"%fname)
        np.savetxt("%s.1.face"%fname, tris, fmt="%d", delimiter = "\t", header="%d %d"%(len(tris), 1), comments="")
    for i in range(len(q)):
        copyfile("%s.node"%fname, "%s.d%d.node"%(fname,i+1))
        qi = q[i]
        tris = triangles.copy()
        qi_idx = np.argwhere(qi!=0).reshape(-1,)
        if len(qi_idx)!=0:
            tris[qi_idx, 4] = q_flag - i
            qi_tris = tris[qi_idx][:,1:4] -1
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=qi_tris)
            plt.title("Q%d"%(i+1))
            plt.show()
        np.savetxt("%s.d%d.face"%(fname, i+1), tris, fmt="%d", delimiter = "\t", header="%d %d"%(len(tris), 1), comments="")


        ri = r[i]
        tetras = np.hstack((tetrahedras.copy(), np.zeros((tetrahedras.shape[0],1))))
        ri_idx = np.argwhere(ri!=0).reshape(-1,)
        if len(ri_idx)!=0:
            tetras[ri_idx, 5] = r_flag - i
            ri_tetras = tetras[ri_idx][:,1:5] -1
            '''
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_tetrasurf(points[:,0], points[:,1], points[:,2], triangles=ri_tetras)
            plt.show()
            '''
        np.savetxt("%s.d%d.ele"%(fname, i+1), tetras, fmt="%d", delimiter = "\t", header="%d %d %d"%(len(tetras), 4, 1), comments="")

    '''
    with open(os.environ['HOME'] + '/mediansurf.1.1.face', 'w') as f:
        f.write("%d\t%d"%(len(triangles),1))
        for tri in triangles:
            f.write("%d\t%d\t%d\t%d"%tuple(tri.astype(int)))
    '''
    print np.unique(t)
    print np.unique(q)
    print np.unique(r)

def surfaces3d(fname):
    points, faces, tetras = load_tetgen_mesh(fname)
    #Orient tetrahedras to have positive volumes
    tetras = tetras - 1
    tetras = utils.orient_simplices(points, tetras)
    surf1 = np.argwhere(faces[:,3] == -1).reshape(-1,)
    surf2 = np.argwhere(faces[:,3] == -2).reshape(-1,)
    triangles = np.copy(faces[:,:-1] - 1)
    surf1_tris = triangles[surf1].copy().reshape(-1,3)
    surf1_edges = utils.get_subsimplices(surf1_tris).reshape(-1,2)
    surf2_tris = triangles[surf2].copy().reshape(-1,3)
    surf2_tris[0] = surf2_tris[0,[1,0,2]]
    surf2_edges = utils.get_subsimplices(surf2_tris).reshape(-1,2)

    is_oriented1, surf1_tris = utils.check_orientation(surf1_tris, surf1_edges) 
    is_oriented2, surf2_tris = utils.check_orientation(surf2_tris, surf2_edges) 

    triangles = np.sort(triangles, axis=1)
    current1 = np.zeros(triangles.shape[0])
    current1[surf1] = 1
    current2 = np.zeros(triangles.shape[0])
    current2[surf2] = 1
    simplices_sort_idx1 = np.argsort(surf1_tris, axis=1)
    simplices_parity1 = utils.permutationparity(simplices_sort_idx1, 2)
    simplices_sort_idx2 = np.argsort(surf2_tris, axis=1)
    simplices_parity2 = utils.permutationparity(simplices_sort_idx2, 2)
    for i, t_idx in enumerate(surf1):
        if simplices_parity1[i] == 1:
            current1[t_idx] = -1
    for i, t_idx in enumerate(surf2):
        if simplices_parity2[i] == 1:
            current2[t_idx] = -1
    inputcurrents = np.vstack((current1, current2))
    lambda_ = 0
    mu = 0.999
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(points[:,0], points[:,1], points[:,2], color="r", triangles=triangles[surf1])
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(points[:,0], points[:,1], points[:,2], color="g", triangles=triangles[surf2])
    plt.show()

    return points, tetras, triangles, inputcurrents, lambda_, mu

def mediansurfdemo3d(outdir='data', save=True):
    '''
    Median shape demo for median surface.
    '''
    start = time.time()
    fname = os.environ['HOME'] +"/mediansurf.1"
    points, simplices, subsimplices, input_currents, _lambda, mu = surfaces3d(fname)
    w, v, b_matrix, cons = median.get_lp_inputs(points, simplices, subsimplices, len(input_currents))
    inout.save_data(input_currents=input_currents, b_matrix=b_matrix, w=w, v=v)
    t, q, r, norm = median.median(points, simplices, subsimplices, \
    input_currents, _lambda, mu, w, v, cons)
    elapsed = time.time() - start
    print 'Elapsed time %f mins.' % (elapsed/60)
    save_tetgen_mesh(t, q, r, fname)
    with open(os.environ['HOME'] + "/README.txt", "w") as f:
        f.write("Experiment Statistics\n")
        f.write("Number of points: %d\n"%len(points))
        f.write("Number of tetrahedras: %d\n"%simplices.shape[0])
        f.write("Number of triangles: %d\n"%subsimplices.shape[0])
        f.write("Number of triangles in Surface1: %d\n"%len(input_currents[0].nonzero()[0]))
        f.write("Number of triangles in Surface2: %d\n"%len(input_currents[1].nonzero()[0]))
        f.write("Lambda: %.5f\n"%_lambda)
        f.write("Mu: %.5f\n"%mu)


if __name__ == '__main__':
    mediansurfdemo3d(save=True)
