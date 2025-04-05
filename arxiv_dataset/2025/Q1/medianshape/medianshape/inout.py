'''
**Input/output methods**
-------------------------
'''

import os.path

import numpy as np
from scipy import sparse

from medianshape.simplicial.mesh import Mesh2D, Mesh3D
import medianshape.utils as utils 

def load_mesh2d(dirname='data/dumps'):
    '''
    Loads previously computed  mesh in 2D, boundary matrix, input currents, w and v
    from a directory given as dirname.
    '''
    mesh = Mesh2D()
    if os.path.exists(dirname):
        if os.path.exists("%s/points.txt"%dirname):
            mesh.points = np.loadtxt("%s/points.txt"%dirname, dtype=np.float) 
            mesh.bbox = (np.amin(mesh.points[:,0]),\
                            np.amin(mesh.points[:,1]),\
                            np.amax(mesh.points[:,0]),\
                            np.amax(mesh.points[:,1]))

            mesh.set_boundary_points()
            mesh.set_diagonal()
            mesh.set_boundary_values()
        else:
            print "Can't load points. <points.txt> file doesn't exist"
        if os.path.exists("%s/edges.txt"%dirname):
            mesh.edges= np.loadtxt("%s/edges.txt"%dirname, dtype=np.int) 
        else:
            print "Can't load edges. <edges.txt> file doesn't exist"
        if os.path.exists("%s/simplices.txt"%dirname):
            mesh.simplices = np.loadtxt("%s/simplices.txt"%dirname, dtype=np.int) 
        else:
            print "Can't load simplices. <simplices.txt> file doesn't exist"
    else:
        print "%s directory doesn't exist"%dirname
    return mesh

def load_mesh3d(dirname='data/dumps'):
    '''
    Loads previously computed  mesh in 3D, boundary matrix, input currents, w and v
    from a directory given as dirname.
    '''
    mesh = Mesh3D()
    if os.path.exists(dirname):
        if os.path.exists("%s/points.txt"%dirname):
            mesh.points = np.loadtxt("%s/points.txt"%dirname, dtype=np.float) 
            mesh.bbox = (np.amin(mesh.points[:,0]),\
                            np.amin(mesh.points[:,1]),\
                            np.amin(mesh.points[:,2]),\
                            np.amax(mesh.points[:,0]),\
                            np.amax(mesh.points[:,1]),\
                            np.amax(mesh.points[:,2]))

            mesh.set_boundary_points()
            mesh.set_diagonal()
            mesh.set_boundary_values()
        else:
            print "Can't load points. <points.txt> file doesn't exist"
        if os.path.exists("%s/edges.txt"%dirname):
            mesh.edges= np.loadtxt("%s/edges.txt"%dirname, dtype=np.int) 
        else:
            print "Can't load edges. <edges.txt> file doesn't exist"
        if os.path.exists("%s/simplices.txt"%dirname):
            mesh.simplices = np.loadtxt("%s/simplices.txt"%dirname, dtype=np.int) 
        else:
            print "Can't load simplices. <simplices.txt> file doesn't exist"
        if os.path.exists("%s/triangles.txt"%dirname):
            mesh.triangles = np.loadtxt("%s/triangles.txt"%dirname, dtype=np.int)
        else:
            mesh.triangles = utils.get_subsimplices(mesh.simplices)
            print "Can't load triangles. <triangles.txt> file doesn't exist"
        if os.path.exists("%s/surf_points.txt"%dirname):
            mesh.surf_points = np.loadtxt("%s/surf_points.txt"%dirname, dtype=np.int) 
        else:
            print "Can't load surface points. <surf_points.txt> file doesn't exist"
    else:
        print "%s directory doesn't exist"%dirname
    return mesh

def load_weights_and_boundary(n_simplices, m_subsimplices, dirname='data/dumps'):
    '''
    Loads saved w and v from a directory given as dirname.
    '''
    w = np.zeros(shape=(m_subsimplices, 1))
    v = np.zeros(shape=(n_simplices, 1))
    b_matrix = sparse.dok_matrix((m_subsimplices, n_simplices), dtype=np.int8)
    if os.path.exists("%s/w.txt"%dirname):
        w = np.loadtxt("%s/w.txt"%dirname, dtype=np.float) 
    else:
        print "Can't load the weight vector, w. <%s/w.txt> file doesn't exist"%dirname
    if os.path.exists("%s/v.txt"%dirname):
        v = np.loadtxt("%s/v.txt"%dirname, dtype=np.float) 
    else:
        print "Can't load the weight vector, v. <%s/v.txt> file doesn't exist"%dirname
    if os.path.exists("%s/b_matrix.txt"%dirname):
        with open("%s/b_matrix.txt"%dirname, 'r') as f:
            for line in f.readlines():
                data = line.split()
                b_matrix[int(data[0]), int(data[1])] = np.int8(data[2])
    else:
        print "Can't load boundary matrix. <%s/b_matrix.txt> file doesn't exist"%dirname
    return w, v, b_matrix

def load_input_currents(m_subsimplices, k, dirname='data/dumps'):
    '''
    Loads saved input currents from a directory given as dirname.
    '''
    input_currents = np.zeros(shape=(k, m_subsimplices), dtype=np.int) 
    for i in range(k):
        if os.path.isfile('%s/input_current%d.txt'%(dirname, i)):
            with open("%s/input_current%d.txt"%(dirname, i), 'r') as f:
                for line in f.readlines():
                    data = line.split()
                    input_currents[i, int(data[1])] = np.int(data[2])
        else:
            print "Can't load input current %d. <input_current%d.txt> doesn't exist"%(i, i)
    return input_currents

def load_solutions(n_simplices, m_subsimplices, k, dirname='data/dumps'):
    '''
    Loads saved median shape solution from a directory given as dirname.
    '''
    x = np.zeros(shape=(2*m_subsimplices + 2*k*m_subsimplices + 2*k*n_simplices,1), dtype=np.int)
    t = np.zeros((m_subsimplices, 1), dtype=np.int) 
    q = np.zeros((k, m_subsimplices), dtype=int)
    r = np.zeros((k, n_simplices), dtype=int)
    if os.path.exists('%s/x.txt'%dirname):
        with open("%s/x.txt"%dirname, 'r') as f:
            for line in f.readlines():
                data = line.split()
                x[int(data[0])] = np.int(data[1])
        t = x[0:m_subsimplices] - x[m_subsimplices:2*m_subsimplices]
        qi_start = 2*m_subsimplices
        for i in range(k):
            qi_end = qi_start + 2*m_subsimplices
            q[i] = (x[qi_start: qi_start+m_subsimplices] - x[qi_start+m_subsimplices: qi_end]).reshape(m_subsimplices,)
            ri_start = qi_end
            ri_end = ri_start + 2*n_simplices
            r[i] = (x[ri_start: ri_start+n_simplices] - x[ri_start+n_simplices: ri_end]).reshape(n_simplices, )
            qi_start = ri_end
    else:
        print "Can't load the solution. <x.txt> file doesn't exist"
    return t, q, r

def load_lp(dirname='data/dumps'):
    '''
    Loads a saved settings of Median shape LP such as c, A, b in sparse dok_matrix format from a directory given as dirname.
    '''

    A = None
    b = None
    c = None
    if os.path.exists("%s/A.txt"%dirname):
        with open("%s/A.txt"%dirname, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    data = line.split()
                    A = sparse.dok_matrix((int(data[0]), int(data[1])), dtype=np.int8)
                else:
                    data = line.split()
                    A[int(data[0]), int(data[1])] = np.int8(data[2])
    else:
        print "Can't load <A>. <A.txt> doesn't exist"
    if os.path.isfile('%s/b.txt'%dirname):
        with open("%s/b.txt"%dirname, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    data = line.split()
                    b = sparse.dok_matrix((int(data[0]), int(data[1])), dtype=np.int8)
                else:
                    data = line.split()
                    b[int(data[0])] = np.int(data[2])
    else:
        print "Can't load <b>. <b.txt> doesn't exist"
    if os.path.isfile('%s/c.txt'%dirname):
        with open("%s/c.txt"%dirname, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    data = line.split()
                    c = sparse.dok_matrix((int(data[0]), int(data[1])), dtype=np.float64)
                else:
                    data = line.split()
                    c[int(data[0])] = data[2]
    else:
        print "Can't load <c>. <c.txt> doesn't exist"
    return A, b, c

def dump_lp(A=None, b=None, c=None, dirname=os.path.abspath('data/dumps'), **kwargs):
    ''' 
    Saves LP settings.
    '''
    if A is not None:
        sparse_savetxt('%s/A.txt' % dirname, A, include_dim=True)
    if b is not None:
        sparse_savetxt('%s/b.txt' % dirname, b.reshape(-1,1), include_dim=True)
    if c is not None:
        sparse_savetxt('%s/c.txt' % dirname, c.reshape(-1,1), fmt='%0.9f', include_dim=True)

def save_data(mesh=None, input_currents=None, b_matrix=None, w=None, v=None, t=None, q=None, r=None, dirname=os.path.abspath('data/dumps'), **kwargs):
    '''
    Saves mesh, input currents, boundary matrix, w and v.
    '''
    if mesh is not None:
        np.savetxt('%s/points.txt' % dirname, mesh.points, fmt='%.6f', delimiter=' ')
        np.savetxt('%s/edges.txt' % dirname, mesh.edges, fmt='%d', delimiter=' ')
        np.savetxt('%s/simplices.txt'% dirname, mesh.simplices, fmt='%d', delimiter=' ')
        if mesh.points.shape[1] == 3:
            np.savetxt('%s/triangles.txt'% dirname, mesh.triangles, fmt='%d', delimiter=' ')
            if mesh.surf_points is not None:
                np.savetxt('%s/surf_points.txt' % dirname, mesh.surf_points, fmt='%d', delimiter=' ')
    if input_currents is not None:
        for i, c in enumerate(input_currents):
            sparse_savetxt('%s/input_current%d.txt' % (dirname,i), c)
    if b_matrix is not None:
        sparse_savetxt('%s/b_matrix.txt' % dirname, b_matrix)
    if w is not None:
        np.savetxt('%s/w.txt' % dirname, w, delimiter=' ')
    if v is not None:
        np.savetxt('%s/v.txt' % dirname, v, delimiter=' ')
    if t is not None:
        if 'mu' in kwargs and 'lambda_' in kwargs:
            sparse_savetxt("%s/t-lambda-%s-mu-%s.txt"%(dirname, kwargs['lambda_'], kwargs['mu']), t)
        else:
            sparse_savetxt("%s/t.txt"%dirname, t)
    if q is not None:
        for i, qi in enumerate(q):
            sparse_savetxt('%s/q%d-lamdba=%s-mu-%s.txt' % (dirname,i+1, kwargs['lambda_'], kwargs['mu']), qi)
    if r is not None:
        for i, ri in enumerate(r):
            sparse_savetxt('%s/r%d-lamdba=%s-mu-%s.txt' % (dirname,i+1, kwargs['lambda_'], kwargs['mu']), ri)
def sparse_savetxt(fname, matrix, fmt='%d', include_dim=False):
    '''
    Saves sparse matrix as text. if the input is not sparse, set is_sparse argument to False.
    '''
    if sparse.issparse(matrix):
        if matrix.getformat() !='coo':
            matrix = matrix.asformat('coo')
    else:
        matrix = sparse.coo_matrix(matrix, dtype=matrix.dtype)
    with open(fname, 'w') as f:
        if include_dim:
            f.write("%d %d \n" % (matrix.shape[0], matrix.shape[1]))
        for i in range(len(matrix.row)):
            fmt_str = "%d %d " + fmt + "\n"
            f.write(fmt_str % (matrix.row[i], matrix.col[i], matrix.data[i]))
