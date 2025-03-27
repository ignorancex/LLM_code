# Written by Chun Kit Wong and CIRC under MIT license:
# https://github.com/wong-ck/DeepSegment/blob/master/LICENSE

import os

import numpy as np
import tensorflow as tf

from utilities import log
from utilities import config
from utilities.tfwrapper import estimator
from utilities.tfwrapper import input_fn
from utilities.tfwrapper import hook
from utilities.io import writer


def main():
    # load configuration from config file and command line
    ds_config = config.load_configuration()

    # create logger
    logger = log.create_or_get_logger(level=ds_config.general_log_level)

    # setup tensorflow
    tf.logging.set_verbosity(ds_config.general_log_level)
    tf.set_random_seed(ds_config.general_random_seed)

    # create segmenter with loaded configuration
    segmenter = estimator.create_segmenter(ds_config)

    # create prediction hooks
    pred_hooks = hook.create_hooks(ds_config, mode="pred")

    # extract required configuration params
    image_type = ds_config.image_type
    output_type = ds_config.predict_output_type

    # create input functions
    pred_input_fn = input_fn.create_input_fn(
        ds_config,
        mode="pred",
        epochs=1,
    )

    # create image writer
    img_writer = create_writer(ds_config)

    # generate predictions using model
    predictions = segmenter.predict(input_fn=pred_input_fn, hooks=pred_hooks)

    # export predictions
    for prediction in predictions:
        if image_type == "hdf5":
            img_writer.write_data(prediction[output_type], name=output_type)
        elif image_type == "nii":
            img_writer.write_data(prediction[output_type])

    # close writer to finish writing images
    img_writer.close()

    logger.info("done!")


def create_writer(ds_config):
    logger = log.create_or_get_logger()

    # extract required configuration params
    img_type = ds_config.image_type
    img_shape = ds_config.image_size
    img_nclasses = ds_config.image_nclasses
    img_batchsize = ds_config.image_batchsize

    predict_output_path = ds_config.predict_output_path
    predict_output_type = ds_config.predict_output_type

    if img_type == "hdf5":
        pass
    elif img_type == "nii":
        img_slicedim = ds_config.image_slicedim
        img_res = ds_config.image_resolution

        path_refnii = ds_config.image_paths[0]
        predict_imgkey = ds_config.predict_imgkey_feature
        if (predict_imgkey is not None) and (predict_imgkey != ''):
            path_refnii = os.path.join(path_refnii, predict_imgkey)
    else:
        msg = "unknown image_type {:}"
        msg = msg.format(img_type)
        logger.error(msg)
        raise ValueError(msg)

    # determine variables dependent on configuration params
    if predict_output_type == "labels":
        dtype = np.uint8
        resample_order = 0
        nchannels = None
    elif predict_output_type == "probabilities":
        dtype = np.float32
        resample_order = 1
        nchannels = img_nclasses

        if img_type == "hdf5":
            img_shape = tuple(list(img_shape) + [nchannels])
    else:
        msg = "unknown predict_output_type {:}".format(predict_output_type)
        msg += "; allowed values: {:}".format(["labels", "probabilities"])
        logger.error(msg)
        raise ValueError(msg)

    # create writer object
    if img_type == "hdf5":
        img_writer = writer.HDF5Writer(
            path=predict_output_path,
            write_freq=5 * img_batchsize,  # write every 5 batches
            resize_chunk=50 * img_batchsize,  # resize every 50 batches
        )

        img_writer.create_dataset(
            name=predict_output_type,
            shape=img_shape,
            dtype=dtype,
        )
    elif img_type == "nii":
        img_writer = writer.NiftiWriter(
            path_output=predict_output_path,
            path_refnii=path_refnii,
            input_res=img_res,
            input_shape=img_shape,
            slice_dim=img_slicedim,
            nchannels=nchannels,
            resample_order=resample_order,
            dtype=dtype,
            async_write=True,
            max_async_write=3,
        )

    return img_writer


if __name__ == "__main__":
    main()
