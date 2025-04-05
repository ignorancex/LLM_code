'''
++++++++++++++++++++
Running median shape
++++++++++++++++++++

'''
from __future__ import absolute_import 

import numpy as np
import matplotlib.pyplot as plt

from medianshape.core import median
from medianshape import utils, inout
from medianshape.viz import plot2d, plot3d

def rundeforms2d(mesh, simplices, subsimplices, input_currents, lambdas, mus, alphas, w=None, v=None, b_matrix=None, file_doc=None, save=True, outdir='data', figcount=2):
    '''
    Accepts simplicial settings, input currents, a list of multiscale factors(:math:`\lambda`), a list of mass regularizing factors(:math:`\mu`), and a list of weights for deformation. 
    Calls Median shape LP and gets a deformed current and flat norm decomposition for each combinations of lambda, mu and weights. It calls plotting function from 'medianshape.viz' which can save them as pdf and/or separate figures.Let K be an underlying simplicial complex of dimension q.

    :param float mesh: an instance of Mesh3D class in 'medianshape.simplicial/mesh'.
    :param int simplices: (p+1)-simplices in K, an array of dimension (nx(p+1)) 
        where :math:`p+1 \leq q` and n is the number of p+1-simplices in K. 
    :param int subsimplices: p-simplices in K, an array of dimension (mxp) 
        where :math:`p \leq q` and m is the number of p-simplice in K. 
    :param int input_currents: input currents, an array of dimension kxm 
        where k is the number of input currents.
    :param float lambdas: a list or an array of multiscale factors.  
    :param float mu: Mass regularizing factor (no mass regularization when set to 0).
    :param float alphas: a set of weights for deformation.
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
    #np.savetxt('%s/dumps/cons.txt', cons, fmt='%d', delimiter=' ')
    for l in lambdas:
        for mu in mus:
            for alpha in alphas:
                t, q, r, norm = median.median(mesh.points, simplices, subsimplices, \
                input_currents, l, mu, w, v, cons, alphas=alpha)
                if save:
                    inout.save_data(t=t, lambda_=l, mu=mu)
                title = r'$MRSMS$, $\lambda=%.06f$, $\mu=%.06f$, $\alpha=[%.06f$ $%.06f]$'%(l, mu, alpha[0], alpha[1])
                figname = '%s/figures/%d'%(outdir, figcount)
                plot2d.plot_median2d(mesh, input_currents, t, title, figname, file_doc, save=save)
                figcount += 1

def rundeforms3d(mesh, simplices, subsimplices, input_currents, lambdas, mus, alphas, w=None, v=None, b_matrix=None, file_doc=None, save=True, outdir='data', figcount=1):
    '''
    Accepts simplicial settings, input currents, a list of multiscale factors(:math:`\lambda`), a list of mass regularizing factors(:math:`\mu`), and a list of weights for deformation. 
    Calls Median shape LP and gets a deformed current and flat norm decomposition for each combinations of lambda, mu and weights. It calls plotting function from 'medianshape.viz' which can save them as pdf and/or separate figures.Let K be an underlying simplicial complex of dimension q.

    :param float mesh: an instance of Mesh3D class in 'medianshape.simplicial/mesh'.
    :param int simplices: (p+1)-simplices in K, an array of dimension (nx(p+1)) 
        where :math:`p+1 \leq q` and n is the number of p+1-simplices in K. 
    :param int subsimplices: p-simplices in K, an array of dimension (mxp) 
        where :math:`p \leq q` and m is the number of p-simplice in K. 
    :param int input_currents: input currents, an array of dimension kxm 
        where k is the number of input currents.
    :param float lambdas: a list or an array of multiscale factors.  
    :param float mu: Mass regularizing factor (no mass regularization when set to 0).
    :param float alphas: a set of weights for deformation.
    :param float w: a vector of subsimplices volumes.
    :param float v: a vector of simplices volumes.
    :param int b_matrix: a boundary matrix representing the boundary operator :math:`\partial_{p+1}` of K.
    :param object file_doc: a file object to which plotted figures are saved. An instance of 'matplotlib.backends.backend_pdf.PdfPages'.
    :param bool save: a boolean flag to indicate whether to save the experiment results.
    :param str outdir: The name of a directory to which the experiment result is saved.
    :param int figcount: The starting index for figures.
    :returns: None 
    '''
    figcount = figcount
    if w is None:
        w = utils.simpvol(mesh.points, mesh.edges)
    if v is None:
        v = utils.simpvol(mesh.points, mesh.triangles)
    if b_matrix is None:
        b_matrix = utils.boundary_matrix(mesh.triangles, mesh.edges)
    k_currents = len(input_currents)
    w, v, b_matrix, cons = median.get_lp_inputs(mesh.points, simplices, subsimplices,  k_currents, w, v, b_matrix)
    #np.savetxt('/home/altaa/dumps1/cons-%s.txt'%opt, cons, fmt='%d', delimiter=' ')
    for l in lambdas:
        for mu in mus:
            for alpha in alphas:
                t, q, r, norm = median.median(mesh.points, mesh.triangles, mesh.edges, input_currents, l, mu, w, v, cons,alphas=alpha)
                if save:
                    inout.save_data(t=t, lambda_=l, mu=mu)
                title = r'$MRSMS$, $\lambda=%.06f$, $\mu=%.06f$, $\alpha=[%.06f$ $%.06f]$'%(l, mu, alpha[0], alpha[1])
                figname = '%s/figures/%d'%(outdir, figcount)
                if save and file_doc is not None:
                    fig = plt.figure(figsize=(8,8))
                    plot3d.plot_median3d(mesh, input_currents, t, title, figname, file_doc, save=save)
                    figcount += 1
