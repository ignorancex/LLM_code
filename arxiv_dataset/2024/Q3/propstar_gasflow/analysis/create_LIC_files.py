'''
This file uses licpy to work, which requires tensorflow 1.14 (not tensorflow 2.0 at the time of writing).
Please make sure you are in an environment where you can use tensorflow. The package cannot be
downloaded if regions==0.5 or radio-beam==0.3.2 are installed (at least through pip)

This code loads the 
'''
from licpy.lic import runlic
import numpy as np
# from licpy.plot import grey_save
import os
from astropy.io import fits

overwrite = True
# L is the length of the streamline that will follow subsequent gradients.
# it should be correlated with the length of the beam
gradientbasefile = 'vel_grad_{}_HC3N'
gradientxfile = gradientbasefile.format('x') + '.fits'
gradientyfile = gradientbasefile.format('y') + '.fits'
equivdiam = 3.7 * 2 # we obtain this value from where we calculate the gradients

nablay_map = fits.getdata(gradientyfile)
nablax_map = fits.getdata(gradientxfile)
gradheader = fits.getheader(gradientxfile)
L = np.array([1,2,3,4]) # in radiuslengths
for Li in L:
    dest2 = gradientbasefile.format('LIC') + '_L{}'.format(Li)
    # licpy transposes and inverts the vectors!
    tex = runlic(nablay_map, nablax_map, Li* equivdiam)
    tex[np.where(tex==0)] = np.nan
    # grey_save(dest2+'.pdf', tex)
    if not os.path.exists(dest2+'.fits') or overwrite: fits.writeto(dest2+'.fits', tex, gradheader, overwrite=True)