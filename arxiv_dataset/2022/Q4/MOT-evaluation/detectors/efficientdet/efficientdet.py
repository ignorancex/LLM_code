
import os
import sys
import pathlib
import torchvision

import numpy as np
# from keras import label_util

from Detector import Detector

newpath = os.path.join(pathlib.Path().absolute(), 'detectors', "efficientdet")
sys.path.insert(0, newpath)

from efficientdet.model_inspect import ModelInspector


class efficientdet(Detector):

    def __init__(self, batch_size):

        super().__init__('efficientdet', batch_size)


        self.runmode = 'saved_model_infer'

        # self.input_image = os.path.join('detectors', "efficientdet", '1.jpg')
        self.output_image_dir = os.path.join('detectors', "efficientdet", 'tmp/')

        model_name = 'efficientdet-d2'
        logdir = os.path.join('detectors', "efficientdet", 'tmp/deff/')
        trace_filename = None
        threads = 0
        bm_runs = 10
        tensorrt = None
        delete_logdir = True
        freeze = False
        use_xla = False
        batch_size = batch_size
        ckpt_path  = model_name
        export_ckpt = None
        hparams = ''
        line_thickness = None
        max_boxes_to_draw = 100
        min_score_thresh = 0
        nms_method = 'gaussian'
        saved_model_dir = os.path.join('detectors', "efficientdet", 'tmp/saved_model')
        tflite_path = None


        # Go to keras/label_util.py
        self.label_permited = [1, 2, 3, 4, 6, 7, 8, 9]


        self.inspector = ModelInspector(
              model_name=model_name,
              logdir=logdir,
              tensorrt=tensorrt,
              use_xla=use_xla,
              ckpt_path=ckpt_path,
              export_ckpt=export_ckpt,
              saved_model_dir=saved_model_dir,
              tflite_path=tflite_path,
              batch_size=batch_size,
              hparams=hparams,
              score_thresh=min_score_thresh,
              max_output_size=max_boxes_to_draw,
              nms_method=nms_method)



    def eval_set(self, dataset, loader, device, verbose=0, display=False):
        '''Run evaluation over loaded data.
        '''

        input_images = np.asarray(dataset.imgs)[:, 0].tolist()

        self.inspector.run_model(self.runmode,
                                 input_image=input_images,
                                 output_image_dir=self.output_image_dir,
                                 loader=loader,
                                 label_permited=self.label_permited,
                                 verbose=verbose,
                                 display=display)
