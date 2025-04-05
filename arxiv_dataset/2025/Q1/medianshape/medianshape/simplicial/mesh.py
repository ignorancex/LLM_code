# encoding: utf-8
'''
Mesh
====

'''
from __future__ import absolute_import

import numpy as np
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from medianshape import utils

class Mesh2D():
    '''
    Mesh in 2D
    '''
    def __init__(self, *args, **kwargs):
        self.points = kwargs.get('points', None)
        self.simplices = kwargs.get('simplices', None)
        self.edges = kwargs.get('edges', None)
        self.bbox = kwargs.get('bbox', None)
        self.fixed_points = kwargs.get('fixed_points', None)
        self.diagonal = kwargs.get('diagonal', None)
        self.boundary_points = None
        self.diagonal = None
        self.ymin = None
        self.xmin = None
        self.ymax = None
        self.xmax = None
        if self.bbox is not None:
            self.set_boundary_points()
            self.set_boundary_values()
            if self.diagonal is None:
                self.set_diagonal()

    def set_boundary_values(self):
        if self.bbox is not None and len(self.bbox) == 4:
            self.xmin = self.bbox[0]
            self.ymin = self.bbox[1]
            self.xmax = self.bbox[2]
            self.ymax = self.bbox[3]
    def set_edges(self):
        edges = set()
        for simplex in self.simplices:
            for i in range(len(simplex)):
                edges.add(tuple(sorted([simplex[i], simplex[(i+1)%len(simplex)]])))
        self.edges = np.array(list(edges), dtype=int)

    def set_boundary_points(self):
        if self.bbox is not None and len(self.bbox) == 4:
                self.boundary_points = np.array([[self.bbox[0], self.bbox[1]],\
                                        [self.bbox[0], self.bbox[3]],\
                                        [self.bbox[2], self.bbox[1]],\
                                        [self.bbox[2], self.bbox[3]]])
    def set_diagonal(self):
        self.diagonal = np.sqrt(self.bbox[2]**2 + self.bbox[3]**2)

    # get simplices based on simplices vector
    def get_simplices(self, simplices_vector, opts=None):
        if simplices_vector.dtype != int:
            simplices_vector = simplices_vector.astype(int)
        # All simplices despite their orientations
        if not opts:
            simplices = self.simplices[simplices_vector.nonzero()[0]]
        # Counter clockwise simplices
        elif opts == 'ccw':
            simplices = self.simplices[np.where(simplices_vector == 1)[0]] 
        # Clockwise simplices
        elif opts == 'cw':
            simplices = self.simplices[np.where(simplices_vector == -1)[0]]
        return simplices 

    def orient_simplices_2D(self):
        for i, simplex in enumerate(self.simplices):
            if self.right_hand_rule(simplex) < 0: 
                self.simplices[i] = self.simplices[i,::-1]

    def right_hand_rule(self, simplex):
        edges = list()
        edges.append(utils.boundary(simplex, 0))
        edges.append(utils.boundary(simplex, 1))
        edge_points = self.points[np.array(edges)]
        v1 = edge_points[0][1] - edge_points[0][0]
        v2 = edge_points[1][1] - edge_points[1][0]
        direction = np.cross(v1,v2)
        return direction


    def get_info(self):
        return r"$%d$ $points$, $%d$ $triangles$ $and$ $%d$ $edges$"% \
        (len(self.points), len(self.simplices), len(self.edges))

    def print_detail(self):
        print self.get_info()
        print "Mesh Points:"
        for i, p in enumerate(self.points):
                print i, p
        print "Point numbers in triangle:"
        for i, t in enumerate(self.simplices):
                print i, t
        print "Edges in mesh:"
        for i, edge in enumerate(self.edges):
            print i, edge

class Mesh3D():
    '''
    Mesh in 3D
    '''
    def __init__(self, *args, **kwargs):
        self.points = kwargs.get('points', [])
        self.simplices = kwargs.get('simplices', [])
        self.edges = kwargs.get('edges', [])
        self.triangles = kwargs.get('triangles', [])
        self.bbox = kwargs.get('bbox', [])
        self.fixed_points = kwargs.get('fixed_points', [])
        self.diagonal = kwargs.get('diagonal', 1e10)
        self.surf_points = None
        if self.bbox is not None or len(self.bbox) != 0:
            self.set_boundary_values()
            self.set_boundary_points()
            if self.diagonal is None:
                self.set_diagonal()

    def set_boundary_values(self):
        if self.bbox is not None and len(self.bbox) == 6:
            self.xmin = self.bbox[0]
            self.ymin = self.bbox[1]
            self.zmin = self.bbox[2]
            self.xmax = self.bbox[3]
            self.ymax = self.bbox[4]
            self.zmax = self.bbox[5]

    def set_boundary_points(self):
        if self.bbox is not None and len(self.bbox) == 6:
            self.boundary_points = np.array([[self.bbox[0], self.bbox[1], self.bbox[2]],\
                                    [self.bbox[0], self.bbox[1], self.bbox[5]],\
                                    [self.bbox[0], self.bbox[4], self.bbox[2]],\
                                    [self.bbox[0], self.bbox[4], self.bbox[5]],\
                                    [self.bbox[3], self.bbox[1], self.bbox[2]],\
                                    [self.bbox[3], self.bbox[4], self.bbox[2]],\
                                    [self.bbox[3], self.bbox[1], self.bbox[5]],\
                                    [self.bbox[3], self.bbox[4], self.bbox[5]]])
    def set_diagonal(self):
        self.diagonal = pdist([[self.bbox[0],self.bbox[1], self.bbox[2]],\
                                [self.bbox[3], self.bbox[4], self.bbox[5]]])

    # get simplices based on simplices vector
    def get_simplices(self, simplices_vector, opts=None):
        if simplices_vector.dtype != int:
            simplices_vector = simplices_vector.astype(int)
        # All simplices despite their orientations
        if not opts:
            simplices = self.simplices[simplices_vector.nonzero()[0]]
        # Counter clockwise simplices
        elif opts == 'ccw':
            simplices = self.simplices[np.where(simplices_vector == 1)[0]] 
        # Clockwise simplices
        elif opts == 'cw':
            simplices = self.simplices[np.where(simplices_vector == -1)[0]]
        return simplices 

    def get_info(self):
        if self.surf_points is None:
            return r"$%d$ $points$, $%d$ $tetrahedras$, $%d$ $triangles$ $and$ $%d$ $edges$"% \
            (len(self.points), len(self.simplices), len(self.triangles), len(self.edges))
        else:
            return r"$%d$ $points$, $%d$ $triangles$ $and$ $%d$ $edges$"% \
            (len(self.surf_points), len(self.triangles), len(self.edges))



    def print_detail(self):
        print self.get_info()
        print "Mesh Points:"
        for i, p in enumerate(self.points):
                print i, p
        print "Point numbers in tetrahedras:"
        for i, t in enumerate(self.simplices):
                print i, t
        print "Point numbers in triangles:"
        for i, t in enumerate(self.triangles):
                print i, t
        print "Edges in mesh:"
        for i, edge in enumerate(self.edges):
            print i, edge
