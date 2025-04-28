import numpy as np
import pandas as pd
from astropy.io import fits
import os
import sys
from skimage import morphology
from astropy.wcs import WCS
sys.path.append('../')
from setup import *
from config import *
import glob
from spectral_cube import SpectralCube

'''
In this routine, we first eliminate small islands of 1 component or 2 component fits
from the original nested sampling results. We save the mle results only where each
solution corresponds and name them _filtered.

Note that for the 3 Gaussian component, we did not evaluate it everywhere, so many parts of 
the K3-2 map are nan, and that affects our filtering

Then, we do a Quality Assessment of the fitted parameters, where we take out the 
components where the central velocity has an uncertainty larger than the channel width 
of the cube, and call them _QA. Note that in the case of HC3N, this will leave out the outflows.
The components that are correct along the same spectra are preserved.

Finally, we create a cube where every plane consists of one component, ordered from most blueshifted 
to most redshifted. 
'''

def header_flatten(head):
    """ Flattens 3D header into 2D. Surely it exists somewhere already... """
    flathead = head.copy()
    for key in flathead.keys():
        if key.endswith('3'):
            flathead.pop(key)
    flathead['NAXIS'] = 2
    flathead['WCSAXES'] = 2

    return flathead

overwrite_cubes = True

ncomps = 2
molecule = 'N2Hp' # for HC3N, add a third component through ALL the code 
cubefile = n2hp_1_0_cube+'.fits'
ngaussmapfile = 'nested-sampling/{}/npeaks_cut5.fits'.format(molecule)
ngaussmapfilefiltered = 'nested-sampling/{}/npeaks_cut5_noislands.fits'.format(molecule)
npeaksfileQA = 'nested-sampling/{}/npeaks_cut5_noislands_QA.fits'.format(molecule)
velcubetotalfile_noQA = 'nested-sampling/{}/vel_components_cube_filtered.fits'.format(molecule)
velcubetotalfile = 'nested-sampling/{}/vel_components_cube_filtered_QA.fits'.format(molecule)

params1gfileQA = 'nested-sampling/{}/NGC1333-SE-mle-x1_filtered_QA.fits'.format(molecule)
params2gfileQA = 'nested-sampling/{}/NGC1333-SE-mle-x2_filtered_QA.fits'.format(molecule)
if ncomps == 3: params3gfileQA = 'nested-sampling/{}/NGC1333-SE-mle-x3_filtered_QA.fits'.format(molecule)

mlex1, mle1head = fits.getdata('nested-sampling/{}/NGC1333-SE-mle-x1.fits'.format(molecule), header=True)
mlex2, mle2head = fits.getdata('nested-sampling/{}/NGC1333-SE-mle-x2.fits'.format(molecule), header=True)
#only for HC3N
if ncomps == 3: mlex3, mle3head = fits.getdata('nested-sampling/{}/NGC1333-SE-mle-x3.fits'.format(molecule), header=True)

minsize1g = 101
minsize2g = 7
minsize3g = 2
Kcut = 5  # the heuristical ln(Z1/Z2) cut for model selection

# we first save the results for 1, 2 and 3 Gaussians where they are most probable given the bayes frame results
# this saves the original state of things before filtering, basically

npeaks_original = fits.getdata(ngaussmapfile)
mask1G = np.where(npeaks_original==1, 1, 0)
mask1G = np.array([mask1G.astype(bool)]*6)
mask2G = np.where(npeaks_original==2, 1, 0)
mask2G = np.array([mask2G.astype(bool)]*12)
mlex1[~mask1G] = np.nan
mlex2[~mask2G] = np.nan
# 3 components
if ncomps == 3:
    mask3G = np.where(npeaks_original==3, 1, 0)
    mask3G = np.array([mask3G.astype(bool)]*18)
    mlex3[~mask3G] = np.nan

fits.writeto('nested-sampling/{}/NGC1333-SE-mle-x1_selected.fits'.format(molecule), mlex1, mle1head, overwrite=True)
fits.writeto('nested-sampling/{}/NGC1333-SE-mle-x2_selected.fits'.format(molecule), mlex2, mle2head, overwrite=True)
if ncomps ==3: fits.writeto('nested-sampling/{}/NGC1333-SE-mle-x3_selected.fits'.format(molecule), mlex3, mle3head, overwrite=True)

#### Removal of small islands of fits

# we load the original maps again 

mlex1, mle1head = fits.getdata('nested-sampling/{}/NGC1333-SE-mle-x1.fits'.format(molecule), header=True)
mlex2, mle2head = fits.getdata('nested-sampling/{}/NGC1333-SE-mle-x2.fits'.format(molecule), header=True)
#only for HC3N
if ncomps == 3: mlex3, mle3head = fits.getdata('nested-sampling/{}/NGC1333-SE-mle-x3.fits'.format(molecule), header=True)
file_KS_mol = 'nested-sampling/{}/NGC1333-SE-Ks.fits'.format(molecule)
Ks = fits.getdata(file_KS_mol)
if ncomps == 3: Ks[2][np.where(np.isnan(Ks[2]))] = -10

if not os.path.exists(ngaussmapfilefiltered) or overwrite_cubes:
    # make the ln(K)>Kcut based map of LoS component nu mbers
    npeaks_map = np.zeros(np.shape(Ks[0]))
    npeaks_map[Ks[0] <= Kcut] = 0
    npeaks_map[Ks[0] > Kcut] = 1
    Karr_clean = morphology.remove_small_objects(npeaks_map.astype(bool), min_size=minsize1g).astype(int)
    fits.writeto('nested-sampling/{0}/mask_5_s{1}.fits'.format(molecule, minsize1g), Karr_clean, header=header_flatten(fits.getheader(file_KS_mol)), overwrite=True)
    #2 components
    Karr2_clean = np.zeros(np.shape(Ks[0]))
    Karr2_clean[(Ks[0] > Kcut) & (Ks[1] <= Kcut)] = 0
    Karr2_clean[(Ks[0] > Kcut) & (Ks[1] > Kcut)] = 1
    Karr2_clean = morphology.remove_small_objects(Karr2_clean.astype(bool), min_size=minsize2g).astype(int)
    fits.writeto('nested-sampling/{0}/mask_5_2ndcomp_s{1}.fits'.format(molecule,minsize2g), Karr2_clean, header=header_flatten(fits.getheader(file_KS_mol)), overwrite=True)
    # 3 components
    if ncomps ==3:
        Karr3_clean = Karr2_clean.copy()
        Karr3_clean[(Ks[0] > Kcut) & (Ks[1] > Kcut) & (Ks[2] <= Kcut)] = 0
        Karr3_clean[(Ks[0] > Kcut) & (Ks[1] > Kcut) & (Ks[2] > Kcut)] = 1
        Karr3_clean = morphology.remove_small_objects(Karr3_clean.astype(bool), min_size=minsize3g).astype(int)
        fits.writeto('nested-sampling/{0}/mask_5_3rdcomp_s{1}.fits'.format(molecule, minsize3g), Karr3_clean, header=header_flatten(fits.getheader(file_KS_mol)), overwrite=True)

    npeaks_map[(Karr_clean == 0)] = 0
    npeaks_map[np.where(np.isnan(Ks[0]))] = np.nan
    npeaks_map[Karr_clean == 1] = 1
    npeaks_map[Karr2_clean == 1] = 2
    if ncomps == 3: npeaks_map[Karr3_clean == 1] = 3

    # for i in range(1, Ks_lines-1):
    #     npeaks_map[(Ks[i-1] > Kcut) & (Ks[i] > Kcut)] = i+1

    npeakshead = header_flatten(fits.getheader(file_KS_mol))
    hdu_npeaks = fits.PrimaryHDU(npeaks_map,
                                 header=npeakshead)
    hdu_npeaks.writeto(ngaussmapfilefiltered, overwrite=True)

else:
    npeaks_map, npeakshead = fits.getdata(ngaussmapfilefiltered, header=True)

#### Save the MLE results only where they correspond

mask1G = np.where(npeaks_map==1, 1, 0)
mask1G = np.array([mask1G.astype(bool)]*6)
mask2G = np.where(npeaks_map==2, 1, 0)
mask2G = np.array([mask2G.astype(bool)]*12)
mlex1[~mask1G] = np.nan
mlex2[~mask2G] = np.nan
# 3 components
if ncomps == 3:
    mask3G = np.where(npeaks_map==3, 1, 0)
    mask3G = np.array([mask3G.astype(bool)]*18)
    mlex3[~mask3G] = np.nan

fits.writeto('nested-sampling/{}/NGC1333-SE-mle-x1_filtered.fits'.format(molecule), mlex1, mle1head, overwrite=True)
fits.writeto('nested-sampling/{}/NGC1333-SE-mle-x2_filtered.fits'.format(molecule), mlex2, mle2head, overwrite=True)
if ncomps ==3: fits.writeto('nested-sampling/{}/NGC1333-SE-mle-x3_filtered.fits'.format(molecule), mlex3, mle3head, overwrite=True)

#### Saving the velocity in layers before QA
if not os.path.exists(velcubetotalfile_noQA) or overwrite_cubes:
    paramstot = np.zeros((3 if (ncomps==3) else 2, mle1head['NAXIS2'], mle1head['NAXIS1'])) * np.nan # 3
    paramstothead = npeakshead.copy()
    paramstothead['NAXIS'] = 3
    paramstothead['WCSAXES'] = 2
    paramstothead['NAXIS3'] = 3 if (ncomps == 3) else  2
    paramstothead['CTYPE3']  = 'BEST FIT VELOCITY'
    paramstothead['BUNIT'] = 'km s-1'
    paramstothead['PLANE1'] = 'COMPONENT 1'
    paramstothead['PLANE2'] = 'COMPONENT 2'
    # 3 components
    if ncomps ==3: paramstothead['PLANE3'] = 'COMPONENT 3'
    
    index1gcomp = np.where(npeaks_map == 1)
    index2gcomp = np.where(npeaks_map == 2)
    # 3 components
    if ncomps == 3: index3gcomp = np.where(npeaks_map == 3)
    
    for yy, xx in zip(index1gcomp[0], index1gcomp[1]):
        paramstot[0,yy,xx] = mlex1[1, yy, xx]
    
    # the component with the least velocity is first in nested sampling results

    for yy, xx in zip(index2gcomp[0], index2gcomp[1]):
        paramstot[0,yy,xx] = mlex2[1, yy, xx]
        paramstot[1,yy,xx] = mlex2[4, yy, xx]
    # 3 components
    if ncomps == 3:
        for yy, xx in zip(index3gcomp[0], index3gcomp[1]):
            paramstot[0,yy,xx] = mlex3[1, yy, xx]
            paramstot[1,yy,xx] = mlex3[4, yy, xx]
            paramstot[2,yy,xx] = mlex3[7, yy, xx]
            
    fits.writeto(velcubetotalfile_noQA, paramstot, paramstothead, overwrite=True)
else:
    print('Filtered paramcube (beforeQA) already exists.')

#### Quality assessment

xarray = np.linspace(0, npeakshead['NAXIS1']-1, npeakshead['NAXIS1']).astype(int)
yarray = np.linspace(0, npeakshead['NAXIS2']-1, npeakshead['NAXIS2']).astype(int)

chanwidth = np.abs(fits.getheader(cubefile)['CDELT3']) #in km/s
chanerror = chanwidth



if not os.path.exists(params1gfileQA) or not os.path.exists(params2gfileQA) or not os.path.exists(npeaksfileQA) or overwrite_cubes:
    for x in xarray:
        for y in yarray:
            npeaks = npeaks_map[y, x]
            if npeaks == 0 or np.isnan(npeaks_map[y, x]): continue
            elif npeaks == 1:
                if mlex1[4,y,x]> chanerror: 
                    for i in range(6): mlex1[i, y, x] = np.nan
                    npeaks_map[y, x] =0
                else: continue
            
            elif npeaks_map[y, x] ==2:
                if mlex2[7,y,x] > chanerror and mlex2[10,y,x] <= chanerror: #first component is wrong only
                    # we eliminate the first and save the second
                    for i in range(3):  mlex1[i,y,x] = mlex2[i+3, y, x] 
                    for i in range(3, 6):  mlex1[i,y,x] = mlex2[i+6, y, x] 
                    for i in range(12): mlex2[i, y, x] = np.nan
                    npeaks_map[y, x] =1
                elif mlex2[7,y,x] <= chanerror and mlex2[10,y,x] > chanerror: #second component is wrong only
                    # we eliminate the second and save the first
                    for i in range(3):  mlex1[i,y,x] = mlex2[i, y, x] 
                    for i in range(3, 6):  mlex1[i,y,x] = mlex2[i+3, y, x] 
                    for i in range(12): mlex2[i, y, x] = np.nan
                    npeaks_map[y, x] =1
                elif mlex2[7,y,x] <= chanerror and mlex2[10,y,x] <= chanerror: 
                    continue #both are right
                else: #both are wrong
                    for i in range(12): mlex2[i, y, x] = np.nan
                    npeaks_map[y, x] = 0
            #only applies for HC3N at the moment        
            elif npeaks_map[y, x] == 3:
                keep = [mlex3[10,y,x] <= chanerror, mlex3[13,y,x] <= chanerror, mlex3[16,y,x] <= chanerror]
                if ~np.any(keep): #all bad
                    for i in range(18): mlex3[i, y, x] = np.nan
                    npeaks_map[y, x] = 0
                elif np.all(keep): #all good
                    continue
                else:
                    indexkeep = np.where(keep)[0]
                    if len(indexkeep) == 2:
                        aux = 0 #without aux, you rewrite the first component and the second is nan
                        for j in indexkeep:
                            for i in range(3): mlex2[i+3*aux, y, x] = mlex3[i+3*j, y, x]
                            for i in range(6, 9):  mlex2[i+3*aux,y,x] = mlex3[i+3+3*j, y, x]
                            aux +=1
                        npeaks_map[y, x] = 2
                    elif len(indexkeep) == 1:
                        j = indexkeep[0]
                        for i in range(3): mlex1[i, y, x] = mlex3[i+3*j, y, x]
                        for i in range(3, 6):  mlex1[i,y,x] = mlex3[i+6+3*j, y, x]
                        npeaks_map[y, x] = 1
                    for i in range(18): mlex3[i, y, x] = np.nan


    fits.writeto(params1gfileQA, mlex1, mle1head, overwrite=True)
    fits.writeto(params2gfileQA, mlex2, mle2head, overwrite=True)
    if ncomps == 3: fits.writeto(params3gfileQA, mlex3, mle3head, overwrite=True) # 3 components
    fits.writeto(npeaksfileQA, npeaks_map, npeakshead, overwrite=True)
else:
    mlex1 = fits.getdata(params1gfileQA)
    mlex2 = fits.getdata(params2gfileQA)
    if ncomps == 3: mlex3 = fits.getdata(params3gfileQA) # 3 components
    npeaks_map = fits.getdata(npeaksfileQA)

#### Saving the velocity layers after QA
if not os.path.exists(velcubetotalfile) or overwrite_cubes:
    paramstot = np.zeros((3 if (ncomps==3) else 2, mle1head['NAXIS2'], mle1head['NAXIS1'])) * np.nan
    paramstothead = npeakshead.copy()
    paramstothead['NAXIS'] = 3
    paramstothead['WCSAXES'] = 2
    paramstothead['NAXIS3'] = 3 if (ncomps ==3) else 2 #3
    paramstothead['CTYPE3']  = 'BEST FIT VELOCITY'
    paramstothead['BUNIT'] = 'km s-1'
    paramstothead['PLANE1'] = 'COMPONENT 1'
    paramstothead['PLANE2'] = 'COMPONENT 2'
    # 3 components
    if ncomps ==3: paramstothead['PLANE3'] = 'COMPONENT 3'
    
    index1gcomp = np.where(npeaks_map == 1)
    index2gcomp = np.where(npeaks_map == 2)
    # 3 components
    if ncomps ==3: index3gcomp = np.where(npeaks_map == 3)
    
    for yy, xx in zip(index1gcomp[0], index1gcomp[1]):
        paramstot[0,yy,xx] = mlex1[1, yy, xx]
    
    # the component with the least velocity is first in nested sampling results

    for yy, xx in zip(index2gcomp[0], index2gcomp[1]):
        paramstot[0,yy,xx] = mlex2[1, yy, xx]
        paramstot[1,yy,xx] = mlex2[4, yy, xx]
    # 3 components
    if ncomps ==3:
        for yy, xx in zip(index3gcomp[0], index3gcomp[1]):
            paramstot[0,yy,xx] = mlex3[1, yy, xx]
            paramstot[1,yy,xx] = mlex3[4, yy, xx]
            paramstot[2,yy,xx] = mlex3[7, yy, xx]
            
    fits.writeto(velcubetotalfile, paramstot, paramstothead, overwrite=True)
else:
    print('Filtered paramcube already exists.')