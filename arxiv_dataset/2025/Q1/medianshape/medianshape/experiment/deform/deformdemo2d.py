# encoding: utf-8

'''
Deformation demo 2D
+++++++++++++++++++
'''
from __future__ import absolute_import

import importlib
import time


import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


from matplotlib.backends.backend_pdf import PdfPages

from medianshape.simplicial import currentgen
from medianshape.viz import plot2d

from medianshape.experiment.deform import rundeforms as run, cases2d


def deform2d(outdir='data', save=True):
    '''
    Deformation demo in 2D
    '''
    lp_times = list()
    start = time.time()
    figcount = 1
    pdf_file = None
    if save:
        pdf_file = PdfPages('%s/deform2d.pdf'%outdir)
    fig = plt.figure(figsize=(12,4))
    mesh, simplices, subsimplices, points, lambdas, mus, alphas, is_closed \
    = cases2d.curvedeform2d() 
    print mesh.get_info()
    print alphas

    vertices, paths, input_currents = currentgen.push_curves_on_mesh(mesh.points, mesh.edges, points, is_closed=is_closed)
    figname = '%s/figures/%d'%(outdir, figcount)
    title = mesh.get_info()
    plot2d.plot_curves_approx2d(mesh, points, vertices, paths, title, figname, pdf_file, save=save)
    plt.tight_layout()
    figcount += 1

    run.rundeforms2d(mesh, simplices, subsimplices, input_currents, lambdas, mus, alphas, file_doc=pdf_file, save=save)
    if save:
        pdf_file.close()
    elapsed = time.time() - start
    print 'Elapsed time %f mins.' % (elapsed/60)
    
if __name__ == '__main__':
    deform2d()
