# encoding: utf-8
'''
++++++++++++
MSFN demo 3D
++++++++++++
'''

from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from medianshape.simplicial.meshgen import distmesh3d, scipy_mesh3d
from medianshape.simplicial.mesh import Mesh3D
from medianshape import utils 
from medianshape.simplicial import pointgen3d, currentgen
from medianshape.viz import plot3d
from medianshape.core.msfn import msfn

import distmesh as dm

def msfndemo3d():
    '''
    MSFN demo 3D 
    '''
    fig = plt.figure(figsize=(19,8))
    mesh = Mesh3D()
    # l - initial length of triangle sides. Change it to vary traingle size
    mesh.bbox = (0,0,0, 1, 1, 1)
    mesh.set_boundary_points()
    mesh.set_diagonal()
    mesh.set_boundary_values()
    mesh.points, mesh.simplices = scipy_mesh3d(mesh.bbox, mesh.fixed_points, l=0.2)
    mesh.triangles = utils.get_simplices(mesh.simplices)
    mesh.edges = utils.get_simplices(mesh.triangles)
    plot3d.plotmesh3d(mesh)
    print mesh.get_info()
    curves = [pointgen3d.curve2(mesh.bbox)]
    points = np.array(curves)
    vertices, paths, input_currents = currentgen.push_curves_on_mesh(mesh, points)
    title = mesh.get_info()
    plot3d.plot_curves_approx(mesh, points, vertices, paths, title)
    plt.show()

    lambdas = [0.0001]
    for input_current in input_currents:
        for l in lambdas:
            title = "lambda=%.04f"%l
            x, s, norm = msfn(mesh.points, mesh.triangles, mesh.edges, input_current, l)
            plot3d.plot_decomposition(mesh, input_currents, x.T, None, s, title)
            plt.show()

if __name__ == "__main__":
    msfndemo3d()
