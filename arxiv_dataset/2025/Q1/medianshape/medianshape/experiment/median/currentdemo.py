from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt

from medianshape.simplicial.mesh import Mesh2D
from medianshape import utils 
from medianshape.viz import plot2d

def currentdemo():
    '''
    Hi
    '''
    mesh = Mesh2D()
    mesh.bbox = (0,0,1,1)
    mesh.set_boundary_points()
    mesh.set_diagonal()
    mesh.set_boundary_values()
    mesh.points = np.array([[0,0], [0, 1],[1,1],[1,0]])
    mesh.simplices = np.array([[0,3,2],
                                [0,2,1]])
    mesh.edges = utils.get_subsimplices(mesh.simplices)
    mesh.orient_simplices_2D()
    w = utils.simpvol(mesh.points, mesh.edges)
    v = utils.simpvol(mesh.points, mesh.simplices)
    b_matrix = utils.boundary_matrix(mesh.simplices, mesh.edges)
    print mesh.get_info()
    print "Points:\n", mesh.points
    print "Triangles:\n", mesh.simplices
    print "Edges:\n", mesh.edges
    fig = plt.figure(figsize=(8,8))
    ax = plt.gca()
    current1 = np.zeros(5)
    current1[2] = 1
    current1[3] = 1
    plot2d.plotmesh2d(mesh)
    plt.scatter(0,0, s=100 )
    plt.scatter(0,1, s=100)
    plt.scatter(1,1, s=100)
    plt.scatter(1,0, s=100)
    plt.annotate('p1', xy=(0,0), xytext=(3, 1.5))
    plot2d.plot_curve2d(mesh, current1, color='r', label="T1", linewidth=8)
    current2 = np.zeros(5)
    current2[1] = -1
    current2[0] = -1
    plot2d.plot_curve2d(mesh, current2, color='g', label="T2", linewidth=8)
    #plt.legend(loc='upper right')
    print "T1:", current1
    print "T2:", current2
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_frame_on(False)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    currentdemo()
