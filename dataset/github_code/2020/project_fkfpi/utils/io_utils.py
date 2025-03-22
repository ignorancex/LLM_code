#!/usr/bin/env python3

import tables as h5
import numpy as np
import gvar as gv
import sys

'''
This file reads the data from an hdf5 file and prepares it into a correlated
gvar data set.

It also handles reading/writing yaml files containing prior width studies
'''

def sort_ens(ensembles):
    # sort ensembles in a useful order
    sorted = list(ensembles)
    sorted.sort(reverse=True)
    return sorted

def format_h5_data(data_path, switches):
    data = h5.open_file(data_path,'r')
    x = dict()
    y = dict()
    p = dict()
    if switches['bs_bias']:
        print('Shifting BS data to boot0')
    else:
        print('Treating mean as one of the bootstrap samples')
    if switches['print_lattice']:
        lattice_fits = []
        mixed_fits   = []

    print('%9s FK/Fpi' %'ensemble')
    print('-----------------------------------------------------------------')
    for ens in sort_ens(switches['ensembles']):
        x[ens] = dict()
        data_dict = dict()
        for m in ['mpi','mk','mss','mju','mjs','mrs','mru']:
            data_dict[m] = data.get_node('/'+ens+'/'+m).read()
        for f in ['FK','Fpi']:
            data_dict[f] = data.get_node('/'+ens+'/'+f).read()
        for m in ['mres_l','mres_s']:
            data_dict[m] = data.get_node('/'+ens+'/'+m).read()

        if switches['bs_bias']:
            data_bs = dict()
            for d in data_dict:
                data_bs[d] = data_dict[d][1:]
            gvdata = gv.dataset.avg_data(data_bs,bstrap=True)
            if switches['debug_bs']:
                print('data set | full mean | bs bias corrected mean')
                print('---------|-----------|-----------------------')
                gvdata_copy = dict(gvdata)
            for d in data_bs:
                gvdata[d] = gvdata[d] + (data_dict[d][0] - gvdata[d].mean)
                if switches['debug_bs']:
                    print(d,gvdata[d].mean,gvdata_copy[d].mean)
        else:
            gvdata = gv.dataset.avg_data(data_dict,bstrap=True)

        y[ens] = gvdata['FK']/gvdata['Fpi']
        print("%9s %s" %(ens,y[ens]))

        # mL
        L_ens = data.get_node('/'+ens+'/L').read()
        x[ens]['mpiL'] = gvdata['mpi'].mean * L_ens
        x[ens]['mkL']  = gvdata['mk'].mean  * L_ens
        x[ens]['meL']  = np.sqrt(4./3*x[ens]['mkL']**2 - 1./3*x[ens]['mpiL']**2)
        x[ens]['mssL'] = gvdata['mss'].mean * L_ens
        x[ens]['mjuL'] = gvdata['mju'].mean * L_ens
        x[ens]['mjsL'] = gvdata['mjs'].mean * L_ens
        x[ens]['mruL'] = gvdata['mru'].mean * L_ens
        x[ens]['mrsL'] = gvdata['mrs'].mean * L_ens

        # MASSES
        p[(ens,'mpi')] = gvdata['mpi']
        p[(ens,'mk')]  = gvdata['mk']
        p[(ens,'mss')] = gvdata['mss']
        p[(ens,'mju')] = gvdata['mju']
        p[(ens,'mjs')] = gvdata['mjs']
        p[(ens,'mru')] = gvdata['mru']
        p[(ens,'mrs')] = gvdata['mrs']
        p[(ens,'Lchi_PP')] = 4 * np.pi * gvdata['Fpi']
        p[(ens,'Lchi_PK')] = 4 * np.pi * np.sqrt(gvdata['FK'] * gvdata['Fpi'])
        p[(ens,'Lchi_KK')] = 4 * np.pi * gvdata['FK']

        # HISQ params
        aw0 = data.get_node('/'+ens+'/aw0').read()
        p[(ens,'aw0')]   = gv.gvar(aw0[0],aw0[1])
        a2di = data.get_node('/'+ens+'/a2DI').read()
        p[(ens,'a2DI')]  = gv.gvar(a2di[0],a2di[1])
        x[ens]['alphaS'] = data.get_node('/'+ens+'/alpha_s').read()
        x[ens]['mxL']    = np.sqrt(4./3 *(gvdata['mk'].mean)**2 -1./3 *(gvdata['mpi'].mean)**2 + a2di[0]) * L_ens

        if switches['print_lattice']:
            lattice_fits.append('%9s& %s& %s& %s& %s& %.2f& %s& %s& %s& %s& %s\\\\' \
                %(ens, gvdata['mpi'], gvdata['mk'], \
                    (gvdata['mpi']/4/np.pi/gvdata['Fpi'])**2,\
                    (gvdata['mk']/4/np.pi/gvdata['Fpi'])**2,\
                    x[ens]['mpiL'], (p[(ens,'aw0')] / 2)**2, x[ens]['alphaS'],\
                    #p[(ens,'aw0')]**2 / 4 / np.pi, x[ens]['alphaS'],\
                    gvdata['Fpi'],gvdata['FK'], gvdata['FK']/gvdata['Fpi']))
            dju = p[(ens,'aw0')]**(-2) * (gvdata['mju']**2 - gvdata['mpi']**2)
            djs = p[(ens,'aw0')]**(-2) * (gvdata['mjs']**2 - gvdata['mk']**2)
            dru = p[(ens,'aw0')]**(-2) * (gvdata['mru']**2 - gvdata['mk']**2)
            drs = p[(ens,'aw0')]**(-2) * (gvdata['mrs']**2 - gvdata['mss']**2)
            dpq = p[(ens,'aw0')]**(-2) * p[(ens,'a2DI')]
            mixed_fits.append('%9s& %s& %s& %s& %s& %s& %s& %s& %s& %s& %s\\\\' \
                %(ens,gvdata['mju'],gvdata['mjs'],gvdata['mru'],gvdata['mrs'],gvdata['mss'],dju,djs,dru,drs,dpq))
    if switches['print_lattice']:
        print(r'ensemble& $am_\pi$& $am_K$& $\e_\pi^2$& $\e_K^2$& $m_\pi L$& $\e_a^2$& $\a_S$& $aF_\pi$& $aF_K$&  $F_K / F_\pi$\\')
        print(r'\hline')
        for l in lattice_fits:
            print(l)
            if any(ens in l for ens in ['a15m135XL','a12m130','a09m135']):
                print("\\hline")
        print('')
        print(r'ensemble& $am_{ju}$& $am_{js}$& $am_{ru}$& $am_{rs}$& $am_{ss}$& $w_0^2 \D_{\rm Mix, ju}^2$& $w_0^2\D_{\rm Mix, js}^2$& $w_0^2\D_{\rm Mix, ru}^2$& $w_0^2\D_{\rm Mix, rs}^2$& $w_0^2 a^2 \D_{\rm I}$\\')
        print(r'\hline')
        for l in mixed_fits:
            print(l)
            if any(ens in l for ens in ['a15m135XL','a12m130','a09m135']):
                print("\\hline")
        data.close()
        sys.exit()

    data.close()
    return {'x':x, 'y':y, 'p':p}
