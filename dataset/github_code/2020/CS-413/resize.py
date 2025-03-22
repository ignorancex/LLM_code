import cv2
import numpy as np
import os
import sys
import time
import argparse

# Resize the image of name imgname with cubic spline interpolation
# such that its longest edge is resized to longEdgeSize and it keeps 
# the shortest with original proportionality. The image is then saved 
# under imgname in png format with no compression. The imgname should 
# end with ".png"
def resizeAndSave(imgname, destname, longEdgeSize):
    
    img = cv2.imread(imgname)
    
    height, width, _ = img.shape
    if height>width:
        width = round(width/height*longEdgeSize) 
        height = longEdgeSize
    else:
        height=round(height/width*longEdgeSize)
        width=longEdgeSize
        
    res = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(destname, res, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    
# reduce all image found under 'orig' with extention 'ext' to 'targetSize' 
# for the longest border and the the results under 'dest' with png format
# and a compression of 0.
def convert(orig, dest, ext, targetSize):
    progress = 0
    for subdir, dirs, files in os.walk(orig):
        for file in files:
            filename = os.path.join(subdir, file)
            if file.endswith(ext):
                resizeAndSave(filename, (dest+file).replace(ext,".png"), targetSize)
                progress += 1
                print("progress: {}".format(progress), end='\r')

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    
    parse.add_argument("-orig", "--origin", help = 'the source folder', type = str)
    parse.add_argument("-dest", "--destination", help = 'the destination folder', type = str)
    parse.add_argument("-ext", "--extention", help = 'the extention of the images to read', type = str)
    parse.add_argument("-targSize", "--targetSize", help = 'the final size of the longest border of the images', type = int, default = 500)
    
    
    args = parse.parse_args()
    
    orig = args.origin #"/ivrldata1/students/dtomar/fivek_dataset/gts/d"
    dest = args.destination #"/ivrldata1/students/team2/reducedDataSet/D/"
    ext = args.extention #".tif"
    
    if orig is None:
        print("Origin folder required. Use --help.")
        sys.exit()
    if dest is None:
        print("Destination folder required. Use --help.")
        sys.exit()
    if ext is None:
        print("Extention required. Use --help.")
        sys.exit()

    targetSize = args.targetSize

start = time.perf_counter()
convert(orig, dest, ext, targetSize)
print("process took: {}s".format(time.perf_counter()-start))