# import cv2 as cv2
import numpy as np
from PIL import Image
import os
import SimpleITK as sitk

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
#import torch
from skimage.transform import resize
import glob
import openslide
import matplotlib.pyplot as plt
#import xmltodict
import pandas as pd
#import bioformats
#import javabridge
#javabridge.start_vm(class_path=bioformats.JARS)
import tifffile

import scipy.ndimage as ndi
import cv2

def scn_to_png_Leeds(svs_file,type,output_dir,lv):
    name = os.path.basename(svs_file).replace('.scn','_%s.png' % (type)).replace('.scn','_%s.png' % (type)).replace('.svs','_%s.png' % (type)).replace('.svs','_%s.png' % (type))

    case_name = name.split('_')[0]

    X5_output_folder = os.path.join(output_dir,case_name, '5X')

    if os.path.exists(os.path.join(X5_output_folder, name)):
        print('already done: %s' % (name))
        return


    try:
        A = openslide.open_slide(svs_file)
        cimg = np.array(A.read_region((1,1), lv, (A.level_dimensions[lv])))
        cimg = ndi.zoom(cimg, (0.5, 0.5, 1), order=1)
        # A = openslide.open_slide(svs_file)
        # cimg = A
    except:
        print('wrong case: %s' % svs_file)
        return

    if cimg.max() == 1.:
        cimg = (cimg / cimg.max() * 255).astype(np.uint8)

    print(os.path.basename(name), type, cimg.shape, cimg.max())

    if not os.path.exists(X5_output_folder):
        os.makedirs(X5_output_folder)

    try:
        plt.imsave(os.path.join(X5_output_folder, name), cimg)
        print('USE plt')
    except:
        cv2.imwrite(os.path.join(X5_output_folder, name), cimg)
        print('USE cv2')
    del cimg


if __name__ == "__main__":
    lv = 0

    scn_dir = '/Data2/MolecularEL/IHC for CD31+PAS/PAS/'
    output_dir = '/Data2/MolecularEL/PAS+Endo_png/'

    PAS_list = glob.glob(os.path.join(scn_dir,'*svs'))

    for PAS in PAS_list:
        Prot = PAS.replace('/PAS/','/CD31_svs/')# .replace('.svs','.scn')
        scn_to_png_Leeds(Prot, 'IHC', output_dir, 1)
        #scn_to_png_Leeds(PAS, 'PAS', output_dir, 1)