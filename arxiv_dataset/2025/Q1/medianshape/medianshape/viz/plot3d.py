# encoding: utf-8
'''
3D plotting
===========

'''

from __future__ import absolute_import

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from distmesh.plotting import axes_simpplot3d

def get_colors(n):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
        RGB color.'''
    if n < 4:
        if n == 1:
            return 'r'
        elif n == 2:
            return 'gr'
        elif n == 3:
            return 'gry'
        elif n == 4:
            return 'grcy' 
    else:
        return cm.rainbow(np.linspace(0, 1, n))

def plotmesh3d(mesh, title='', figname=None, file_doc=None, save=True):
    '''
    Plots a tetrahedralized mesh.
    '''
    dim = mesh.points.shape[1]
    if dim == 2:
        from distmesh.plotting import SimplexCollection
        ax = plt.gca()
        c = SimplexCollection()
        ax.add_collection(c)
        m1 = np.amin(mesh.bbox[0:3])
        m2 = np.amax(mesh.bbox[3:])
        ax.set_xlim([m1, m2])
        ax.set_ylim([m1, m2])
        ax.set_zlim([m1, m2])
        ax.set_aspect('equal')
        ax.set_axis_off()
        c.set_simplices((mesh.points, mesh.simplices))
    elif dim == 3:
        ax = plt.gca(projection='3d')
        fig = ax.figure
        m1 = np.amin(mesh.bbox[0:3])
        m2 = np.amax(mesh.bbox[3:])
        ax.set_xlim([m1, m2])
        ax.set_ylim([m1, m2])
        ax.set_zlim([m1, m2])
        ax.set_aspect('equal')
        #axes_simpplot3d(ax, mesh.points, mesh.simplices, mesh.points[:,1] > 0)
        axes_simpplot3d(ax, mesh.points, mesh.simplices)
        if title is not None:
            ax.set_title(title, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
        fig.tight_layout()
        if save and figname:
            plt.savefig('%s.png'%figname, pad_inches=-1, box_inches='tight')
        if save and file_doc:
            file_doc.savefig(fig)
    else:
        print "Plotting only supported in dimensions 2 and 3." 

def plot_point3d(mesh, vertex_vector, title=None, color="black", marker='.', s=200, label=""):
    '''
    Plots points in 3D.
    '''
    ax = plt.gca(projection='3d')
    if type(vertex_vector) == list:
        vertex_vector = np.array(vertex_vector)
    if vertex_vector.dtype != int:
        vertex_vector = vertex_vector.astype(int)
    point = mesh.points[vertex_vector]
    ax.scatter(point[0], point[1], point[2], c=color, marker=marker, s=s, label=label)
    if title:
        ax.set_title(title, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)

def plot_curve3d(mesh, func_path, title=None, color="black", marker=None, linewidth=3, ls='-', label="", set_lim = False):
    '''
    Plots a curve.
    '''
    ax = plt.gca(projection='3d')
    if set_lim:
        m1 = np.amin(mesh.bbox[0:3])
        m2 = np.amax(mesh.bbox[3:])
        ax.set_xlim([m1, m2])
        ax.set_ylim([m1, m2])
        ax.set_zlim([m1, m2])
        ax.set_aspect('equal')
    if type(func_path) == list:
        func_path = np.array(func_path)
    if func_path.dtype != int:
        func_path = func_path.astype(int)
    nonzero_edges = func_path.nonzero()[0]
    for i, edge_idx in enumerate(nonzero_edges):
        edge = mesh.edges[edge_idx]
        points = mesh.points[edge]
        ax.plot(points[:,0], points[:,1], points[:,2], c=color, linewidth=linewidth, marker=marker, ls=ls, label=label if i==0 else "")
    if title is not None:
        ax.set_title(title, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)

# Plot simplices
def plot_simplices3d(mesh, simplices, title=None, figname=None, file_doc=None, save=True, color='y'):
    '''
    Plots 3-simplices (tetrahedras).
    '''
    ax = plt.gca(projection='3d')
    fig = ax.figure
    m1 = np.amin(mesh.bbox[0:3])
    m2 = np.amax(mesh.bbox[3:])
    ax.set_xlim([m1, m2])
    ax.set_ylim([m1, m2])
    ax.set_zlim([m1, m2])
    ax.set_aspect('equal')
    #ccw_symbol = u'\u2941'
    #cw_symbol = u'\u21BB'
    triangles = mesh.triangles[simplices.nonzero()[0]].reshape(-1,3)
    ax.plot_trisurf(mesh.points[:,0], mesh.points[:,1], mesh.points[:,2], triangles=triangles,  color=(0.8, 0.9, 1), shade=False, linewidth=0.2, edgecolor='gray')

    '''
    #Alternative method for the previous two lines to plot the simplices
    for i in simplices.nonzero()[0]:
        #axes_simpplot3d(ax, mesh.points, mesh.triangles[i].reshape(1,-1), mesh.points[:,1] > 0)
        axes_simpplot3d(ax, mesh.points, mesh.triangles[i].reshape(1,-1))
    '''
    fig.tight_layout()
    if title is not None:
        ax.set_title(title, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
    if save and figname:
        plt.savefig('%s.png'%figname, pad_inches=-1, box_inches='tight')
    if save and file_doc:
        file_doc.savefig(fig)

def plot_curves_approx3d(mesh, points, vertices, paths, figname=None, file_doc=None, save=True, title=None):
    '''
    Plots curves described as points along with corresponding interpolated curves in a tetrahedralized mesh.
    '''
    colors = get_colors(len(points))
    ax = plt.gca(projection='3d')
    fig = ax.figure
    m1 = np.amin(mesh.bbox[0:3])
    m2 = np.amax(mesh.bbox[3:])
    ax.set_xlim([m1, m2])
    ax.set_ylim([m1, m2])
    ax.set_zlim([m1, m2])
    ax.set_aspect('equal')
    if title is not None:
        ax.set_title(title, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
    for i, path in enumerate(paths):
        plot_curve_approx3d(mesh, points[i], vertices[i], path, color=colors[i], label=r"$T_{%d}$"%(i+1))
    fig.tight_layout()
    if save and figname:
        plt.savefig('%s.png'%figname, box_inches='tight')
    if save and file_doc:
        file_doc.savefig(fig)

def plot_curve_approx3d(mesh, input_points, closest_vertices, path, title=None, color="red", linewidth=3, label=""):
    '''
    Plots a curve described as points along with a corresponding interpolated curve in a tetrahedralized mesh.
    '''
    ax = plt.gca(projection='3d')
    ax.plot(input_points[:,0], input_points[:,1], input_points[:,2], c=color, ls="--", label='Input points')
    if title is not None:
        ax.set_title(title, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
    for i, edge in enumerate(path):
        points = mesh.points[edge]
        if len(path) != 1:
            ax.plot(points[:,0], points[:,1], points[:,2], c=color, linewidth=linewidth, label=label if i==0 else "")
    ax.scatter(mesh.points[closest_vertices][:,0], mesh.points[closest_vertices][:,1], mesh.points[closest_vertices][:,2], s=100, c=color, label="Closest vertices")
    ax.scatter(input_points[:,0], input_points[:,1], input_points[:,2], c=color)
    #plt.legend(loc='lower right')

def plot_median3d(mesh, input_currents, t, title='', figname="", file_doc=None, save=True):
    '''
    Plots a median curve in a tetrahedralized mesh.
    Find the following line in the code and uncomment if you want to plot median on mesh.
    
    ax.plot_trisurf(mesh.points[:,0], mesh.points[:,1], mesh.points[:,2], triangles=mesh.triangles,  color=(0.8, 0.9, 1), shade=False)
    '''
    colors= get_colors(len(input_currents))
    ax = plt.gca(projection='3d')
    fig = ax.figure
    m1 = np.amin(mesh.bbox[0:3])
    m2 = np.amax(mesh.bbox[3:])
    ax.set_xlim([m1, m2])
    ax.set_ylim([m1, m2])
    ax.set_zlim([m1, m2])
    ax.set_aspect('equal')
    # Uncomment it if you want to plot median on mesh
    ax.plot_trisurf(mesh.points[:,0], mesh.points[:,1], mesh.points[:,2], triangles=mesh.triangles,  color=(0.8, 0.9, 1), shade=False, linewidth=0.2, edgecolor='gray')
    for i, c in enumerate(input_currents):
        plot_curve3d(mesh, c, color=colors[i], label=r'$T_{%d}$'%(i+1), linewidth=5)
    plot_curve3d(mesh, t, label=r"$Median$")
    fig.tight_layout()
    #plt.legend(loc='lower right')
    #plt.title(title, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
    if save and figname:
        plt.savefig("%s.png"%figname, pad_inches=-1, box_inches='tight')
    if save and file_doc:
        file_doc.savefig(fig)

def plot_curve_and_median3d(mesh, input_currents, t, title=None, figname=None, file_doc=None, save=True, lim=5):
    '''
    Plots input curves and their median curve.
    '''
    colors= get_colors(len(input_currents))
    ax = plt.gca(projection='3d')
    fig = ax.figure
    ax.set_aspect('equal')
    for i, c in enumerate(input_currents):
        lim = float(mesh.zmax)/10
        ax.set_xlim(mesh.xmax + lim)
        ax.set_ylim(mesh.ymax + lim)
        ax.set_zlim(mesh.zmax + lim)
        plt.clf()                    
        plot_curve3d(mesh, c, color=colors[i], linewidth=5, \
        label=r'$T%d$'%(i+1))
        plot_curve3d(mesh, t, title, label='Median')
        plt.legend(loc='lower right')
        fig.tight_layout()
        if save and figname:
            plt.savefig("%s-%d.png" % (figname, i), pad_inches=-1, box_inches='tight')
        if save and file_doc:
            file_doc.savefig(fig)

def plot_decomposition3d(mesh, input_currents, t, q, r, title='', figname=None, file_doc=None, save=True):
    '''
    Plots flat norm decomposition.
    '''
    colors = get_colors(len(input_currents))
    fig = plt.figure(figsize=(8,8))
    ax = plt.gca(projection='3d')
    fig = ax.figure
    m1 = np.amin(mesh.bbox[0:3])
    m2 = np.amax(mesh.bbox[3:])
    ax.set_xlim([m1, m2])
    ax.set_ylim([m1, m2])
    ax.set_zlim([m1, m2])
    ax.set_aspect('equal')
    for i, r_i in enumerate(r):
        set_lim = False
        color = colors[i]
        plt.clf()
        ax = plt.gca(projection='3d')
        m1 = np.amin(mesh.bbox[0:3])
        m2 = np.amax(mesh.bbox[3:])
        ax.set_xlim([m1, m2])
        ax.set_ylim([m1, m2])
        ax.set_zlim([m1, m2])
        ax.set_aspect('equal')
        plot_simplices3d(mesh, r_i, color=color)
        if np.count_nonzero(r_i) == 0:
            set_lim=False
        if q is not None:
            if title is not None:
                title = title + r", $Q_{%d}&R_{%d}"%(i+1, i+1)
            plot_curve3d(mesh, q[i], title, color='m', marker='*', linewidth=6, label='$Q_{%d}$'%(i+1), set_lim=set_lim)
        if t is not None:
            plot_curve3d(mesh, t, linewidth=4, label=r"$Median$", set_lim=set_lim)
                
        plot_curve3d(mesh, input_currents[i], color='r', ls='--', \
                label=r"$T_{%d}$"%(i+1), set_lim=set_lim)

        #plt.legend(loc='lower right')
        if save and figname:
            plt.savefig("%s-%d.png" % (figname, i), pad_inches=-1, box_inches='tight')
        if save and file_doc:
            file_doc.savefig(fig)
        fig.tight_layout()
        plt.axis("off")
        fig = plt.figure(figsize=(8,8))
