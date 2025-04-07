
import os
import argparse

import tensorflow.compat.v1 as tf

from absl import logging
from model_inspect import ModelInspector


def parse_input():

    parser = argparse.ArgumentParser(description='Detectors demo')

    # Select an able detector from list.
    list_detectors = ['efficientdet-d0', 'efficientdet-d1', 'efficientdet-d2', 'efficientdet-d3', 'efficientdet-d4',
                      'efficientdet-d5', 'efficientdet-d6', 'efficientdet-d7', 'efficientdet-d7x']

    parser.add_argument("--model", help="Name of the model to use.", choices=list_detectors, required=True)
    parser.add_argument("--batch", help="Batch size to use.", required=True, type=int)

    return parser.parse_args()



def setUp(args):

    model_name = args.model
    ckpt_path  = model_name
    saved_model_dir = 'tmp/saved_model'
    batch_size = args.batch


    # model_name =
    logdir = 'detectors/efficientdet/tmp/deff/'
    # runmode =
    trace_filename = None
    threads = 0
    bm_runs = 10
    tensorrt = None
    delete_logdir = True
    freeze = False
    use_xla = False
    # ckpt_path =
    export_ckpt = None
    hparams = ''
    input_image = None
    output_image_dir = None
    input_video = None
    output_video = None
    line_thickness = None
    max_boxes_to_draw = 100
    min_score_thresh = 0.4
    nms_method = 'hard'
    # saved_model_dir =
    tflite_path = None


    inspector = ModelInspector(
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


    return inspector


def logs():

    logging.set_verbosity(logging.WARNING)
    tf.enable_v2_tensorshape()
    tf.disable_eager_execution()


def download_files(model_name, saved_model_dir):

    os.system("wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/%s.tar.gz" %(model_name))
    os.system("tar xf %s.tar.gz" %(model_name))
    os.system("rm -r %s" %(saved_model_dir))


def remove_files(model_name):

    os.system("rm %s.tar.gz" %(model_name))
    os.system("rm -r %s" %(model_name))




if __name__ == '__main__':

    logs()

    # Parse
    args = parse_input()

    # Download model
    download_files(args.model, 'tmp/saved_model')

    # Run setup
    inspector = setUp(args)
    inspector.run_model('saved_model')

    # Remove temporal files
    remove_files(args.model)
