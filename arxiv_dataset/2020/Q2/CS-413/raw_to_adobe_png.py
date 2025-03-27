import rawpy
import numpy as np
#import imageio
import cv2
import os
import sys
import time
import argparse

# Save the file named imgname as destname assuming imgname is a raw picture format. 
# The resulting file, destname, should en with ".png" and is the input image after
# minimal demosaicing in png format with no compression.
def saveRawToPNG(imgname, destname):
    
    params = rawpy.Params(demosaic_algorithm=None, #parce que ça change rien de toute facon sur nos images
                     fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Full, #min noise
                     use_auto_wb=True, # calculé auto par rawpy, selon le cours
                     no_auto_bright=True, # pas de brightness auto pour ne pas modifier sans raison
                     no_auto_scale=False, # selon moi la linéarisation parlé dans le cours
                     output_color=rawpy.ColorSpace.Adobe, # car les artistes ont traviallé dans ADOBE qqch
                     output_bps=8) # output jusqu'à 256


    
    rawimg = rawpy.imread(imgname)
    
    rawpng = rawimg.postprocess(params) 
    # image qui se rapproche le plus de ce que les artistes ont importés dans leur programme qqch avant
    # de les retoucher, du coup le gan n'apprendrait li
    rawpng = cv2.cvtColor(rawpng, cv2.COLOR_BGR2RGB)
    cv2.imwrite(destname, rawpng, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    #imageio.imsave(destname, rawpng)
    
# reduce all image found under 'orig' with extention 'ext' to 'targetSize' 
# for the longest border and save the results under 'dest' with png format
# and a compression of 0
def convert(orig, dest):
    ext = ".dng"
    progress = 0
    for subdir, dirs, files in os.walk(orig):
        for file in files:
            filename = os.path.join(subdir, file)
            if file.endswith(ext):
                saveRawToPNG(filename, (dest+file).replace(ext,".png"))
                progress += 1
                print("progress: {}".format(progress), end='\r')

# Converts all raw .dng image found under the file of name provided in the
# input string parameter --origin to the folder of name provided in the input 
# string parameter --destination. The resulting format is png with no compression.
# The resulting image is a demosaiced version of the original.
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    
    parse.add_argument("-orig", "--origin", help = 'the source folder', type = str)
    parse.add_argument("-dest", "--destination", help = 'the destination folder', type = str)
    
    args = parse.parse_args()
    
    orig = args.origin 
    dest = args.destination 
    
    if orig is None:
        print("Origin folder required. Use --help.")
        sys.exit()
    if dest is None:
        print("Destination folder required. Use --help.")
        sys.exit()

start = time.perf_counter()
convert(orig, dest)
print("process took: {}s".format(time.perf_counter()-start))