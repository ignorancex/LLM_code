# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 06:18:21 2018

@author: 구준모
"""

import numpy as np
import PIL

list_im = ['Filled-Bristol_OA_1.png', 'Filled-bristol_LSOA_1.png']
imgs    = [ PIL.Image.open(i) for i in list_im ]
# pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

# save that beautiful picture
imgs_comb = PIL.Image.fromarray( imgs_comb)
imgs_comb.save( 'OA_LSOA_eigen01_map.png' )    