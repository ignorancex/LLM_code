# encoding: utf-8
'''
Deformation demo 3D
+++++++++++++++++++
'''

from __future__ import absolute_import

import time

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from medianshape.simplicial import currentgen
from medianshape.experiment.deform import rundeforms, cases3d

from medianshape.viz import plot3d

def deformdemo3d(outdir='data', save=True):
    '''
    Deformation demo in 3D.
    '''
    lp_times = list()
    start = time.time()
    pdf_file = None
    if save:
        pdf_file = PdfPages('%s/deform3d.pdf'%outdir)
    fig = plt.figure(figsize=(12,8))
    figcount = 1
    mesh, simplices, subsimplices, points, lambdas, mus, alphas, is_closed \
    = cases3d.longitudes3ds_finer()
    print mesh.get_info()
    if mesh.surf_points is None:
        title = mesh.get_info() 
        title = None
        plot3d.plotmesh3d(mesh, title, '%s/figures/%d'%(outdir, figcount), pdf_file, save)
    else:
        title = None
        plot3d.plot_simplices3d(mesh, np.ones(len(mesh.triangles)), title, '%s/figures/%d'%(outdir, figcount), pdf_file, save)

    figcount += 1
    fig.tight_layout()
    fig = plt.figure(figsize=(8,8))
    vertices, paths, input_currents = currentgen.push_curves_on_mesh(mesh.points, mesh.edges, points, is_closed, mesh.surf_points)

    figname = '%s/figures/%d'%(outdir, figcount)
    plot3d.plot_curves_approx3d(mesh, points, vertices, paths, figname, pdf_file, save)
    figcount += 1
    fig.tight_layout()
   # plt.show()
    t = rundeforms.rundeforms3d(mesh, simplices, subsimplices, input_currents, lambdas, mus, alphas, file_doc=pdf_file, save=save, figcount=figcount)
    if save:
        pdf_file.close()
    elapsed = time.time() - start
    print 'Elapsed time %f mins.' % (elapsed/60)
    
if __name__ == '__main__':
    deformdemo3d()
