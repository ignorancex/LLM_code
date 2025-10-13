
# Update directory paths based on your operating system

import numpy as np
import os
import sys

sys.path.append('/home/ntorbati/PycharmProjects/PumaSubmit/Docker/DockerTrack2')
os.chdir('/home/ntorbati/PycharmProjects/PumaSubmit/Docker/DockerTrack2')

from Docker.DockerTrack2.post_process_once import run

sys.path.append('/home/ntorbati/PycharmProjects/PumaSubmit/Docker/DockerTrack2/inference')
os.chdir('/home/ntorbati/PycharmProjects/PumaSubmit/Docker/DockerTrack2/inference')

from inference_cell_tissue_unetpp10 import inference_tissue as inference_cell
from inference_tissue_from_nuclei import inference_tissue_nuclei
from inference_4th_submission import inference_tissue


# this is where model predictons are saved
out_path = '/home/ntorbati/PycharmProjects/PumaSubmit/preds/'

# input images path
path = '/home/ntorbati/PycharmProjects/PumaSubmit/Sample/'


if not os.path.exists(out_path):
    os.makedirs(out_path)

# all_tissue_data = np.sort([path + image for image in os.listdir(path)])
images = [path + f for f in os.listdir(path) if f.endswith('.tif') and not f.endswith('_context.tif')][0:1]
res = [out_path + f for f in os.listdir(path) if f.endswith('.tif') and not f.endswith('_context.tif')][0:1]


# Initial tissue prediction
inference_tissue(images, res, show_result = True)

# Nuclei class prediction using tissue
inference_cell(images, res,show_result = True)

# Tissue prediction using Nuclei
inference_tissue_nuclei(images, res, show_result = True)

# Instnce segmentation using HoverNext
os.chdir('/home/ntorbati/PycharmProjects/PumaSubmit/Docker/DockerTrack2/')
for image in images:
    run(input_pth = image, pred_path = image.replace(path, out_path),local_process = True)


