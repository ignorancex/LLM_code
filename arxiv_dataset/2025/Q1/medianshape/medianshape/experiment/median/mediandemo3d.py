#encoding: utf-8

'''
Median shape demo 3D
--------------------
'''

from __future__ import absolute_import

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from medianshape.simplicial import currentgen 
from medianshape.experiment.median import runmedians as run, cases3d

from medianshape.viz import plot3d 

def show_median3d():
    '''
    Shows the results of a saved experiment case.
    '''
    mesh, input_currents, t, q, r = cases3d.get_saved_case(dirname='data/curves_on_sphere')
    fig = plt.figure(figsize=(8,8))
    plot3d.plotmesh3d(mesh, mesh.get_info())
    fig.tight_layout()
    plt.show()
    plt.figure(figsize=(8,8))
    title = r"$\lambda=0.0010$, $\mu=0.000010$"
    plot3d.plot_median3d(mesh, input_currents, t, title=title) 
    plt.show()
    plot3d.plot_decomposition3d(mesh, input_currents, t, q, r, title=title)


def mediandemo3d(outdir='data', save=True):
    '''
    Median shape demo in 3D. The experiment case is chosen from 'medianshape.cases3d'.
    Given the experiment case, it gets input currents from the input curves by pushing
    the underlying simplicial complex. Then input the simplicial setting to Median LP which
    solves median current. The experiment result is saved in a given output directory(data).
    '''
    lp_times = list()
    start = time.time()
    pdf_file = None
    if save:
        pdf_file = PdfPages('%s/mediandemo3d.pdf'%outdir)

    fig = plt.figure(figsize=(8,8))
    figcount = 1
    mesh, simplices, subsimplices, points, lambdas, mus, is_closed \
    = cases3d.equally_spaced_longitudes3d()
    #= cases3d.tunnel_loops_on_torus_surface3d()
    #= cases3d.handle_loops_on_torus_surface3d()
    #= cases3d.torus_surface3d()
    #= cases3d.equally_spaced_longitudes3d()
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
    title = r'$Curve$ $approximation$'
    plot3d.plot_curves_approx3d(mesh, points, vertices, paths, figname, pdf_file, save, title=None)
    figcount += 1
    fig.tight_layout()
    run.runmedians3d(mesh, simplices, subsimplices, input_currents, lambdas, mus, file_doc=pdf_file, save=save, figcount=figcount)
    if save:
        pdf_file.close()
    elapsed = time.time() - start
    print 'Elapsed time %f mins.' % (elapsed/60)
    plt.show()

if __name__ == '__main__':
    mediandemo3d(save=True)
    #show_median3d()
