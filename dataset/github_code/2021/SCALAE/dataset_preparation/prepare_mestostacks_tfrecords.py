import zipfile
import tqdm
from defaults import get_cfg_defaults
import sys
import logging

from skimage.transform import resize
from net import *
import numpy as np
import pickle
import random
import argparse
import os
# from dlutils.pytorch.cuda_helper import *
import tensorflow as tf
import imageio
from PIL import Image, ImageEnhance
import gdal

def log_norm_pop(pop_batch):
    pop_batch = np.log(pop_batch)
    pop_batch = np.clip(pop_batch, 0, 8)
    pop_batch = pop_batch / 8
    return pop_batch

def exp_unnorm_pop(pop_batch):
    pop_batch = pop_batch * 8
    pop_batch = np.exp(pop_batch)
    return pop_batch

def parse_tif_file(file, channels):
    tif_file = gdal.Open(file)  
    results = {}
    for k in channels.keys():
        img_list =[]
        for c in channels[k]:
            img_list.append(tif_file.GetRasterBand(c).ReadAsArray())
        img = np.stack(img_list, axis=-1)
        results[k] = img.squeeze()
    return results


def prepare_dataset(cfg, logger, train=True):
    if train:
        directory = os.path.dirname(cfg.DATASET.PATH)
    else:
        directory = os.path.dirname(cfg.DATASET.PATH_TEST)

    os.makedirs(directory, exist_ok=True)

    images = []
    # The official way of generating CelebA-HQ can be challenging.
    # Please refer to this page: https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download
    # You can get pre-generated dataset from: https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P
    source_path = cfg.DATASET.ORIGINAL_SOURCE_PATH
    for i, filename in tqdm.tqdm(enumerate(os.listdir(source_path))):
        if filename.endswith('.tif'):
            images.append((filename, filename))

    print("Total count: %d" % len(images))
    if train:
        images = images[:cfg.DATASET.SIZE]
    else:
        images = images[:cfg.DATASET.SIZE_TEST]

    count = len(images)
    print("Count: %d" % count)

    random.seed(0)
    random.shuffle(images)
    
    img_size = 2 ** cfg.DATASET.MAX_RESOLUTION_LEVEL

    if train:
        folds = cfg.DATASET.PART_COUNT
    else:
        folds = cfg.DATASET.PART_COUNT_TEST
    celeba_folds = [[] for _ in range(folds)]

    count_per_fold = count // folds
    print("folds: %d" % folds)
    for i in range(folds):
        celeba_folds[i] += images[i * count_per_fold: (i + 1) * count_per_fold]

    for i in range(folds):
        print("fold: %d" % i)
        if train:
            path = cfg.DATASET.PATH
        else:
            path = cfg.DATASET.PATH_TEST

        writers = {}
        for lod in range(cfg.DATASET.MAX_RESOLUTION_LEVEL, 1, -1):
            tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
            part_path = path % (lod, i)
            os.makedirs(os.path.dirname(part_path), exist_ok=True)
            tfr_writer = tf.python_io.TFRecordWriter(part_path, tfr_opt)
            writers[lod] = tfr_writer

        for label, filename in tqdm.tqdm(celeba_folds[i]):
            images = parse_tif_file(os.path.join(source_path, filename), {'rgb': [3,2,1], 'pop': [6]})
            for k in images.keys():
                if k == 'pop':
                    images[k] = log_norm_pop(images[k])
                    images[k] = np.clip((images[k]*255).astype(np.uint8), 0, 255)
                    images[k] = Image.fromarray(images[k])
                elif k == 'rgb':
                    images[k] = np.clip((images[k]*255).astype(np.uint8), 0, 255)
                    images[k] = Image.fromarray(images[k])
                    images[k] = ImageEnhance.Brightness(images[k]).enhance(1.4)
                    images[k] = ImageEnhance.Contrast(images[k]).enhance(1.4)
                images[k] = images[k].resize((img_size, img_size))
                
                test_image_path = path.split('/')[:-2] + [f'test_image_{k}.jpg']
                test_image_path = '/'.join(test_image_path)
                if not os.path.exists(test_image_path):
                    print(f'Saving example image to {test_image_path}')
                    images[k].save(test_image_path)
                    
                images[k] = np.asarray(images[k])
                if k == 'pop':
                    images[k] = np.expand_dims(images[k], -1)
                images[k] = images[k].transpose((2, 0, 1))
                
            for lod in range(cfg.DATASET.MAX_RESOLUTION_LEVEL, 1, -1):
                ex = tf.train.Example(features=tf.train.Features(feature={
                    'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=images['rgb'].shape)),
                    'rgb': tf.train.Feature(bytes_list=tf.train.BytesList(value=[images['rgb'].tostring()])),
                    'pop': tf.train.Feature(bytes_list=tf.train.BytesList(value=[images['pop'].tostring()]))
                }))
                writers[lod].write(ex.SerializeToString())

                for k in images.keys():
                    c = images[k].shape[0]
                    h = images[k].shape[1]
                    w = images[k].shape[2]
                    image = torch.tensor(np.asarray(images[k], dtype=np.float32)).view(1, c, h, w)
                    image_down = F.avg_pool2d(image, 2, 2).clamp_(0, 255).to('cpu', torch.uint8).view(
                        c, h // 2, w // 2).numpy()
                    images[k] = image_down


def run():
    parser = argparse.ArgumentParser(description="Adversarial, hierarchical style VAE")
    parser.add_argument(
        "--config-file",
        default="configs/popgan.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    prepare_dataset(cfg, logger, True)


if __name__ == '__main__':
    run()

