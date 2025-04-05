"""
Converts the LabelMe Images into VGG16 output where VGG16 model is only
trained on ImageNet.
"""
import csv
import os
from time import perf_counter, process_time

import h5py
import numpy as np

from psych_metric.datasets import data_handler
from experiment.research import predictors

labelme = data_handler.load_dataset(
    'LabelMe',
    'psych_metric/datasets/crowd_layer/',
)
images, labels = labelme.load_images()

vgg16 = predictors.load_model('vgg16', parts='vgg16')

start_perf_time = perf_counter()
start_process_time = process_time()

vgg16_output = vgg16.predict(images)

time_process = process_time() - start_process_time
time_perf = perf_counter() - start_perf_time

output_dir = os.path.join(labelme.data_dir, labelme.dataset)

np.save_txt(
    os.path.join(output_dir, 'labelme_vgg16_encoded.csv'),
    vgg16_output,
    delimiter=',',
)

output_h5 = h5py.File(os.path.join(''), 'w')
output_h5.create_dataset('images_vgg16_encoded', data=vgg16_output)
output_h5.close()

with open(os.path.join(output_dir, 'runtimes_labelme_vgg16_encoded.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['performance_runtime', time_perf])
    writer.writerow(['process_runtime', time_process])
