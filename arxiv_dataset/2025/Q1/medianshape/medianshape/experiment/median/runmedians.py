'''
Running MRSMS for several :math:`\lambda`'s and :math:`\mu`'s
-------------------------------------------------------------
'''
import numpy as np
import matplotlib.pyplot as plt

import medianshape.core.median as median
from medianshape import utils, inout
from medianshape.viz import plot2d, plot3d

def runmedians2d(mesh, simplices, subsimplices, input_currents, lambdas, mus, w=None, v=None, b_matrix=None, file_doc=None, save=True, outdir='data', figcount=2):
    '''
    Accepts simplicial settings, input currents, a list of multiscale factors(:math:`\lambda`) 
    and a list of mass regularizing factors(:math:`\mu`). 
    Calls Median shape LP and gets median shape and flat norm decomposition for each combinations of lambda and mu. It calls plotting function from 'medianshape.viz' which can save them as pdf and/or separate figures.Let K be an underlying simplicial complex of dimension q.

    :param float mesh: an instance of Mesh3D class in 'medianshape.simplicial/mesh'.
    :param int simplices: (p+1)-simplices in K, an array of dimension (nx(p+1)) 
        where :math:`p+1 \leq q` and n is the number of p+1-simplices in K. 
    :param int subsimplices: p-simplices in K, an array of dimension (mxp) 
        where :math:`p \leq q` and m is the number of p-simplice in K. 
    :param int input_currents: input currents, an array of dimension kxm 
        where k is the number of input currents.
    :param float lambdas: a list or an array of multiscale factors.  
    :param float mu: Mass regularizing factor (no mass regularization when set to 0).
    :param float w: a vector of subsimplices volumes.
    :param float v: a vector of simplices volumes.
    :param int b_matrix: a boundary matrix representing the boundary operator :math:`\partial_{p+1}` of K.
    :param object file_doc: a file object to which plotted figures are saved. An instance of 'matplotlib.backends.backend_pdf.PdfPages'.
    :param bool save: a boolean flag to indicate whether to save the experiment results.
    :param str outdir: The name of a directory to which the experiment result is saved.
    :param int figcount: The starting index for figures.
    :returns: None 
    '''
    if w is None:
        w = utils.simpvol(mesh.points, subsimplices)
    if v is None:
        v = utils.simpvol(mesh.points, simplices)
    if b_matrix is None:
        b_matrix = utils.boundary_matrix(simplices, subsimplices)
    k_currents = len(input_currents)
    w, v, b_matrix, cons = median.get_lp_inputs(mesh.points, simplices, subsimplices,  k_currents, w, v, b_matrix)
    if save:
        inout.save_data(mesh, input_currents, b_matrix, w, v)
    for l in lambdas:
        for mu in mus:
            t, q, r, norm = median.median(mesh.points, mesh.simplices, mesh.edges, \
            input_currents, l, mu=mu, w=w, v=v, cons=cons)
            if save:
                inout.save_data(t=t, lambda_=l, mu=mu)
            title = ''
            figname = '%s/figures/%d'%(outdir, figcount)
            plot2d.plot_median2d(mesh, input_currents, t, title, figname, file_doc, save=save)
            #plt.show()
            figcount += 1

            figname = '%s/figures/%d'%(outdir, figcount)
            plot2d.plot_decomposition2d(mesh, input_currents, t, q, r, title, \
            figname, file_doc, save)
            figcount += input_currents.shape[0]

def runmedians3d(mesh, simplices, subsimplices, input_currents, lambdas, mus, w=None, v=None, b_matrix=None, file_doc=None, save=True, outdir='data', figcount=1):
    '''
    Accepts simplicial settings, input currents, a list of multiscale factors(:math:`\lambda`) 
    and a list of mass regularizing factors(:math:`\mu`). 
    Calls Median shape LP and gets median shape and flat norm decomposition for each combinations of lambda and mu. It calls plotting function from 'medianshape.viz' which can save them as pdf and/or separate figures.Let K be an underlying simplicial complex of dimension q.

    :param float mesh: an instance of Mesh3D class in 'medianshape.simplicial/mesh'.
    :param int simplices: (p+1)-simplices in K, an array of dimension (nx(p+1)) 
        where :math:`p+1 \leq q` and n is the number of p+1-simplices in K. 
    :param int subsimplices: p-simplices in K, an array of dimension (mxp) 
        where :math:`p \leq q` and m is the number of p-simplice in K. 
    :param int input_currents: input currents, an array of dimension kxm 
        where k is the number of input currents.
    :param float lambdas: a list or an array of multiscale factors.  
    :param float mu: Mass regularizing factor (no mass regularization when set to 0).
    :param float w: a vector of subsimplices volumes.
    :param float v: a vector of simplices volumes.
    :param int b_matrix: a boundary matrix representing the boundary operator :math:`\partial_{p+1}` of K.
    :param object file_doc: a file object to which plotted figures are saved. An instance of 'matplotlib.backends.backend_pdf.PdfPages'.
    :param bool save: a boolean flag to indicate whether to save the experiment results.
    :param str outdir: The name of a directory to which the experiment result is saved.
    :param int figcount: The starting index for figures.
    :returns: None 
    '''
    if w is None:
        w = utils.simpvol(mesh.points, subsimplices)
    if v is None:
        v = utils.simpvol(mesh.points, simplices)
    if b_matrix is None:
        b_matrix = utils.boundary_matrix(simplices, subsimplices)
    if save:
        inout.save_data(mesh, input_currents, b_matrix, w, v)
    k_currents = len(input_currents)
    w, v, b_matrix, cons = median.get_lp_inputs(mesh.points, simplices, subsimplices,  k_currents, w, v, b_matrix)
    #np.savetxt('%s/dumps/cons-%s.txt'%cons, fmt='%d', delimiter=' ')
    for l in lambdas:
        for mu in mus:
            t, q, r, norm = median.median(mesh.points, simplices, subsimplices, input_currents, l, mu, w, v, cons)
            if save:
                inout.save_data(t=t, q=q, r=r, lambda_=l, mu=mu)
            title = r"$\lambda=%.06f$, $\mu=%.06f$" % (l, mu)
            title = None
            figname = '%s/figures/%d'%(outdir, figcount)
            fig = plt.figure(figsize=(8,8))
            plot3d.plot_median3d(mesh, input_currents, t, title, figname, file_doc, save)
            plt.tight_layout()
            #plt.show()
            fig = plt.figure(figsize=(8,8))
            figcount += 1

            figname = '%s/figures/%d'%(outdir, figcount)
            plot3d.plot_decomposition3d(mesh, input_currents, t, q, r, title, \
            figname, file_doc, save)
            figcount += input_currents.shape[0]
