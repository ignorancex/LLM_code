# encoding: utf-8
'''
Median shape demo 2D
--------------------

'''
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from medianshape.simplicial import currentgen
from medianshape.experiment.median import runmedians as run, cases2d

from medianshape.viz import plot2d 

def mediandemo2d(outdir='data', save=True):
    '''
    Median shape demo in 2D. The experiment case is chosen from 'medianshape.cases2d'.
    Given the experiment case, it gets input currents from the input curves by pushing
    the underlying simplicial complex. Then input the simplicial setting to Median LP which
    solves median current. The experiment result is saved in an output directory(data)'.

    '''
    lp_times = list()
    start = time.time()
    pdf_file = None
    if save:
        pdf_file = PdfPages('%s/mediandemo2d.pdf'%outdir)
    fig = plt.figure(figsize=(8,8))
    figcount = 1

    mesh, simplices, subsimplices, points, lambdas, mus, is_closed = cases2d.multicurves2d() 

    vertices, paths, input_currents = currentgen.push_curves_on_mesh(mesh.points, mesh.edges, points, is_closed=is_closed)
    figname = '%s/figures/%d'%(outdir, figcount)
    title = mesh.get_info() + r' - $Curve$ $approximation$'
    plot2d.plot_curves_approx2d(mesh, points, vertices, paths, title, figname, pdf_file, save)
    figcount += 1
    
    run.runmedians2d(mesh, simplices, subsimplices, input_currents, lambdas, mus, file_doc=pdf_file, save=save)
    if pdf_file is not None:
        pdf_file.close()
    elapsed = time.time() - start
    print 'Elapsed time %f mins.' % (elapsed/60)
    
if __name__ == '__main__':
    mediandemo2d()
