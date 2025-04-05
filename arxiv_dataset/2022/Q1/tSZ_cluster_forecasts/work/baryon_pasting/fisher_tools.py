import numpy as np
from pylab import *

def get_knox_errors(els, cl11, fsky, cl22 = None, cl12 = None):

    """
    get Knox bandpower errors.

    els: multipoles
    cl11: signal + noise in the first map.
    cl22: signal + noise in the second map.
    cl12: signal + noise cross spectrum of the two maps.
    """

    delta_el = np.diff(els)[0]

    if cl22 is not None and cl12 is not None:
        cl_total = np.sqrt( (cl12**2. + cl11 * cl22)/2. )
    else:
        cl_total = cl11

    cl_knox_err = np.sqrt(2./ (2.*els + 1.) / fsky / delta_el ) * (cl_total)
    cl_knox_err[np.isnan(cl_knox_err)] = 0.
    cl_knox_err[np.isinf(cl_knox_err)] = 0.

    return cl_knox_err

def get_ymap_power_spectrum_fisher(els, cl_deriv_dic, delta_cl, params, lmin = 0, lmax = 12000):

    npar = len(params)
    F = np.zeros([npar,npar])

    for lcntr, l in enumerate( els ):

        if l<lmin or l>lmax: continue

        #get the covariance matrix
        cov_mat_l = np.mat( delta_cl[lcntr]**2. )
        ##cov_mat_inv_l = sc.linalg.pinv2(cov_mat_l)
        cov_mat_inv_l = np.linalg.pinv(cov_mat_l)

        param_combinations = []
        for pcnt,p in enumerate(params):
            for pcnt2,p2 in enumerate(params):
                param_combinations.append([p,p2, pcnt, pcnt2])

        for (p,p2, pcnt, pcnt2) in param_combinations:

            fprime1_l_vec = np.asarray( [cl_deriv_dic[p][lcntr]] )
            fprime2_l_vec = np.asarray( [cl_deriv_dic[p2][lcntr]] )

            #print(fprime1_l_vec, fprime2_l_vec, cov_mat_l, cov_mat_inv_l); sys.exit()

            curr_val = np.dot(fprime1_l_vec, np.dot( cov_mat_inv_l, fprime2_l_vec ))

            F[pcnt2,pcnt] += curr_val

    return F

def get_ellipse_specs(COV, howmanysigma = 1):
    """
    Refer https://arxiv.org/pdf/0906.4123.pdf
    """
    assert COV.shape == (2,2)
    confsigma_dic = {1:2.3, 2:6.17, 3: 11.8}

    sig_x2, sig_y2 = COV[0,0], COV[1,1]
    sig_xy = COV[0,1]
    
    t1 = (sig_x2 + sig_y2)/2.
    t2 = np.sqrt( (sig_x2 - sig_y2)**2. /4. + sig_xy**2. )
    
    a2 = t1 + t2
    b2 = t1 - t2

    a = np.sqrt(abs(a2))
    b = np.sqrt(abs(b2))

    t1 = 2 * sig_xy
    t2 = sig_x2 - sig_y2
    theta = np.arctan2(t1,t2) / 2.
    
    alpha = np.sqrt(confsigma_dic[howmanysigma])
    
    #return (a*alpha, b*alpha, theta)
    return (a*alpha, b*alpha, theta, alpha*(sig_x2**0.5), alpha*(sig_y2**0.5))

def get_Gaussian(mean, sigma, minx, maxx, delx = None):

    if delx is None: delx = (maxx - minx)/100000.

    x = np.arange(minx, maxx, delx)

    #return x, 1./(2*np.pi*sigma)**0.5 * np.exp( -(x - mean)**2. / (2 * sigma**2.)  )
    return x, np.exp( -(x - mean)**2. / (2 * sigma**2.)  )

def get_latex_param_str(param, use_H = False):
    params_str_dic= {'eps_f': r'$\epsilon_{f}$ [$10^{6}$]', 'A_nt': r'$A_{\rm nt}$', \
                     'sigma_8': r'$\sigma_{8}$', 'Omega_m': r'$\Omega_{m}$'}

    return params_str_dic[param]

def make_triangle_plot(F_dic, tr, tc, param_names, param_values_dict, desired_params_to_plot, one_or_two_sigma = 1, fix_axis_range_to_xxsigma = 5., fsval = 12, noofticks = 4, color_dic = None, ls_dic = None, lwval = 1., show_one_sigma_lab = True, use_percent = False, bias_dic = None):

    """
    F_dic: Fisher matrix dictionary with experiment names as keys.
    tr: total rows.
    tc: total rows.
    param_values_dict: dictionary containing Fiducial values of cosmological parameters.
    desired_params_to_plot: parameters to be plotted.
    one_or_two_sigma: one or two or three or XX sigma region to be shown. Default is 1\sigma.
    fix_axis_range_to_xxsigma: x and y limits of axis will be fixed to xx\sigma. Default is 5\sigma.
    fsval: fontsize.
    noofticks: noofticks on axis.
    color_dic: Colours to be used for different experiments. If None, choose it automatically.
    ls_dic: line style for experiments. If None, we will use "-".
    lwval: Line width.
    show_one_sigma_lab: If True, parameter errors will be reported on 1d posteriors.
    use_percent: If True, then parameter errors will be reported as per cent on 1d posteriors.
    bias_dic: Bias on parameters. Not used for 3G forecasting paper. Defualt is None.
    """

    import matplotlib.patches as patches
    import warnings, matplotlib.cbook
    warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)

    ################################################
    ################################################
    #pick colours
    if color_dic is None:
        color_arr = ['navy', 'darkgreen', 'goldenrod', 'orangered', 'darkred']
        color_dic = {}
        for expcntr, expname in enumerate( F_dic ):
            color_dic[expname] = color_arr[expcntr]

    #linestyles
    if ls_dic is None:
        ls_dic = {}
        for expname in F_dic:
            ls_dic[expname] = '-'


    param_names_to_plot = []
    cosmo_param_pl_chars_dict = {}
    pcntr_for_plotting = 1
    ##for pp in sorted( desired_params_to_plot ):
    for pp in desired_params_to_plot:
        param_names_to_plot.append(pp)
        cosmo_param_pl_chars_dict[pp] = pcntr_for_plotting
        pcntr_for_plotting += 1

    totparamstoplot = len(param_names_to_plot)
    diag_matrix = np.arange( totparamstoplot**2 ).reshape((totparamstoplot, totparamstoplot)) + 1
    ##print(diag_matrix); sys.exit()


    sbpl_locs_dic = {}
    for p1 in param_names_to_plot:
        for p2 in param_names_to_plot:
            sbpl_locs_dic[(p1,p2)] = cosmo_param_pl_chars_dict[p1] + ((cosmo_param_pl_chars_dict[p2]-1) * totparamstoplot)

    ##print(sbpl_locs_dic); sys.exit()

    widthvalues_for_axis_limits = {} #used later to fix axis ranges
    for pcntr1, p1 in enumerate( param_names ):
        widthvalues_for_axis_limits[p1] = 0.
        for pcntr2, p2 in enumerate( param_names ):        

            if p1 not in desired_params_to_plot or p2 not in desired_params_to_plot: continue
            '''
            if p1 in fix_params or p2 in fix_params: continue
            '''

            sbpl = sbpl_locs_dic[(p1,p2)]
            if sbpl not in np.tril(diag_matrix): continue

            cov_inds_to_extract = [(pcntr1, pcntr1), (pcntr1, pcntr2), (pcntr2, pcntr1), (pcntr2, pcntr2)]

            #fiducial values
            x = param_values_dict[p1]
            y = param_values_dict[p2]

            #x and y extents; \eplison_x and \epsilon_y for 1d Posteriors.
            deltax, deltay = 5*x, 5*y #some large range
            ##epsilon_x, epsilon_y = abs(x/10000.), abs(y/10000.) #for Gaussian 1d curve.
            x1, x2 = x - deltax, x + deltax
            y1, y2 = y - deltay, y + deltay

            if fix_axis_range_to_xxsigma is not None:
                x1, x2 = x - deltax*fix_axis_range_to_xxsigma, x + deltax*fix_axis_range_to_xxsigma
                y1, y2 = y - deltay*fix_axis_range_to_xxsigma, y + deltay*fix_axis_range_to_xxsigma
            else:
                x1, x2 = x - deltax, x + deltax
                y1, y2 = y - deltay, y + deltay

            '''
            if p1 =='eps_f': 
                x*=1e6
                x1*=1e6
                x2*=1e6
                deltax*=1e6
            if p2 =='eps_f': 
                y*=1e6
                y1*=1e6
                y2*=1e6
                deltay*=1e6
            '''
            
            #latex parameter labels
            p1str = get_latex_param_str(p1)
            p2str = get_latex_param_str(p2)

            #create subplot first
            ax = subplot(tr, tc, sbpl)#, aspect = 'equal')

            if sbpl<=(tr*(tc-1)):
                setp(ax.get_xticklabels(), visible=False)
            else:
                xlabel(p1str, fontsize = fsval);

            if ((sbpl-1)%tc == 0) and totparamstoplot>1 and sbpl!= 1:
                ylabel(p2str, fontsize = fsval);
            else:
                setp(ax.get_yticklabels(), visible=False)

            #print(p1, p2, sbpl)
            for expcntr, exp in enumerate( F_dic ):

                F_mat = F_dic[exp]
                #exp_COV = sc.linalg.pinv(F_mat)
                exp_COV = np.linalg.inv(F_mat)

                #cov_extract = np.asarray( [exp_COV[ii] for ii in cov_inds_to_extract] ).reshape((2,2))
                cov_extract = []
                for ii in cov_inds_to_extract:
                    cov_extract.append(exp_COV[ii])
                cov_extract = np.asarray( cov_extract ).reshape((2,2))

                #if np.sum(cov_extract)<=1e-20: print(p1,p2, cov_extract); continue
                #print(p1, p2, cov_extract)

                colorval = color_dic[exp]
                lsval = ls_dic[exp]
                for ss in range(one_or_two_sigma):

                    if p1 == p2:

                        widthval = cov_extract[0,0]**0.5##/2.35
                        #print(p1, widthval)
                        hor, ver = get_Gaussian(x, widthval, x1, x2)#, epsilon_x)
                        #labval = r'%.4f' %(widthval)
                        labval = None
                        if show_one_sigma_lab:
                            if bias_dic is not None:
                                labval = r'%.2f$\sigma$' %(bias_dic[p1]/widthval)
                                #print(p1, labval, bias_dic[p1], widthval)
                            else:
                                if abs(x)>0. and use_percent:
                                    labval = r'%.3f\%%' %(100. * abs(widthval/x))
                                else:
                                    labval = r'%.3g' %(widthval)
                        #print(labval)
                        if totparamstoplot==1 and exp_dic is not None:
                            labval = r'%s: %s' %(exp_dic[exp][0], labval)
                        plot(hor, ver, color = colorval, lw = lwval, label = labval, ls = lsval)
                        legend(loc = 4, framealpha = 1, fontsize = fsval-4, ncol = 1, edgecolor = 'None', handletextpad=0.8, handlelength = 0.8, numpoints = 1, columnspacing = 1)#, handlelength = 2.)

                        xlim(x1, x2)
                        ylim(0., 1.)
                        setp(ax.get_yticklabels(), visible=False); tick_params(axis='y',left='off')
                        title(p1str, fontsize = fsval+2);

                        if p1 in widthvalues_for_axis_limits:
                            widthvalues_for_axis_limits[p1] = max(widthvalues_for_axis_limits[p1], widthval)

                    else:

                        Ep = get_ellipse_specs(cov_extract, howmanysigma = ss + 1)
                        widthval, heightval = Ep[0], Ep[1]
                        #if widthval<=1e-10 or heightval<=1e-10: continue
                        #print(widthval, heightval, p1, p2)
                        ellipse = patches.Ellipse(xy=[x,y], width=2.*widthval, height=2.*heightval, angle=np.degrees(Ep[2]))

                        ax.add_artist(ellipse)
                        ellipse.set_clip_box(ax.bbox)
                        ellipse.set_facecolor('None')#colorarr[ss])
                        ellipse.set_edgecolor(colorval)
                        ellipse.set_linewidth(lwval)
                        ellipse.set_linestyle(lsval)
                        #ellipse.set_alpha(alphaarr[ss])

                        xlim(x1, x2)
                        ylim(y1, y2)

                        if exp.find('bias') == -1:
                            axhline(y, lw = 0.1);axvline(x, lw = 0.1)

            if noofticks is not None:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=noofticks))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=noofticks))

            for label in ax.get_xticklabels(): label.set_fontsize(fsval-3.5)
            for label in ax.get_yticklabels(): label.set_fontsize(fsval-3.5)

            if (0):
                grid(True, which='major', axis = 'x', lw = 0.5, alpha = 0.1)
                grid(True, which='major', axis = 'y', lw = 0.5, alpha = 0.1)
    
    #set axis limits now based on widths obtained
    ##print(widthvalues_for_axis_limits.keys()); sys.exit()
    if fix_axis_range_to_xxsigma is not None:
        for pcntr1, p1 in enumerate( param_names ):
            for pcntr2, p2 in enumerate( param_names ):
                if p1 not in desired_params_to_plot or p2 not in desired_params_to_plot: continue
                #if p1 in fix_params or p2 in fix_params: continue
                ##if (not show_diagonal) and p1 == p2: continue
                sbpl = sbpl_locs_dic[(p1,p2)]                            
                if sbpl not in np.tril(diag_matrix): continue
                deltax, deltay = widthvalues_for_axis_limits[p1], widthvalues_for_axis_limits[p2]
                if deltax == 0. and deltay == 0.: continue
                '''
                x, deltax = param_dict[p1]
                y, deltay = param_dict[p2]
                '''
                x = param_values_dict[p1]
                y = param_values_dict[p2]
                x1, x2 = x - deltax*fix_axis_range_to_xxsigma, x + deltax*fix_axis_range_to_xxsigma
                y1, y2 = y - deltay*fix_axis_range_to_xxsigma, y + deltay*fix_axis_range_to_xxsigma
                ##if p1 == p2: print(widthvalues_for_axis_limits[p1], x, x1, x2)
                ax = subplot(tr, tc, sbpl)#, aspect = 'equal')
                #print(deltax, deltay, p1, p2)

                xlim(x1, x2)
                if p1 != p2:
                    ylim(y1, y2)

    return color_dic, ls_dic
