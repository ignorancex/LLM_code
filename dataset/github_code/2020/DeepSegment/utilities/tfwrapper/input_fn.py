# Written by Chun Kit Wong and CIRC under MIT license:
# https://github.com/wong-ck/DeepSegment/blob/master/LICENSE

import random

import tensorflow as tf

import utilities
from utilities.io import reader


def create_input_fn(ds_config, mode, epochs=None):
    logger = utilities.log.create_or_get_logger()

    # extract required parameters
    filepaths = ds_config.image_paths
    image_type = ds_config.image_type

    output_res = ds_config.image_resolution
    output_shape = ds_config.image_size
    slice_dim = ds_config.image_slicedim

    block_lengths = ds_config.image_blocklengths
    cyclic = ds_config.image_cyclic
    batchsize = ds_config.image_batchsize
    shuffle_buffersize = ds_config.image_shuffle_buffer

    prefetch_buffersize = 2 * batchsize

    if mode.lower() == "train":
        key_featureimg = ds_config.train_imgkey_feature
        key_labelimg = ds_config.train_imgkey_label
        key_presence = ds_config.train_imgkey_presence
        shuffle = ds_config.train_shuffle_image
    elif mode.lower() == "eval":
        key_featureimg = ds_config.evaluate_imgkey_feature
        key_labelimg = ds_config.evaluate_imgkey_label
        key_presence = ds_config.evaluate_imgkey_presence
        shuffle = ds_config.evaluate_shuffle_image
    elif mode.lower() == "pred":
        key_featureimg = ds_config.predict_imgkey_feature
        key_labelimg = None
        key_presence = None
        shuffle = False
    else:
        msg = "unknown mode {:}".format(mode)
        logger.error(msg)
        raise ValueError(msg)

    # additional checkings
    if key_featureimg is not None:
        if key_featureimg == '':
            key_featureimg = None

    if key_labelimg is not None:
        if key_labelimg == '':
            key_labelimg = None

    if key_presence is not None:
        if key_presence == '':
            key_presence = None

    if image_type.lower() == "hdf5":
        pass
    elif image_type.lower() == "nii":
        if (mode.lower() == "pred") and (len(filepaths) != 1):
            msg = "under {:} mode, length of filepaths "
            msg += "must be 1 if image_type is {:}"
            msg = msg.format(mode, image_type.lower())
            logger.error(msg)
            raise ValueError(msg)
        if key_presence is not None:
            msg = "presence array is not supported if image_type is {:}"
            msg = msg.format(image_type.lower())
            logger.error(msg)
            raise ValueError(msg)
    else:
        msg = "unknown image_type {:}"
        msg = msg.format(image_type)
        logger.error(msg)
        raise ValueError(msg)

    # create input_fn and return
    return lambda: _input_fn(
        filepaths=filepaths,
        image_type=image_type,
        key_featureimg=key_featureimg,
        key_labelimg=key_labelimg,
        key_presence=key_presence,
        block_lengths=block_lengths,
        cyclic=cyclic,
        output_res=output_res,
        output_shape=output_shape,
        slice_dim=slice_dim,
        shuffle=shuffle,
        shuffle_buffersize=shuffle_buffersize,
        epochs=epochs,
        batchsize=batchsize,
        prefetch_buffersize=prefetch_buffersize,
    )


def _input_fn(
    filepaths,
    image_type,
    key_featureimg,
    key_labelimg,
    key_presence,
    block_lengths,
    cyclic,
    output_res,
    output_shape,
    slice_dim,
    shuffle,
    shuffle_buffersize,
    epochs,
    batchsize,
    prefetch_buffersize,
):
    logger = utilities.log.create_or_get_logger()

    # create dataset
    if image_type.lower() == "hdf5":
        dataset = _create_hdf5_dataset(
            filepaths=filepaths,
            key_featureimg=key_featureimg,
            key_labelimg=key_labelimg,
            key_presence=key_presence,
            block_lengths=block_lengths,
            cyclic=cyclic,
            slice_dim=slice_dim,
            shuffle=shuffle,
        )
    elif image_type.lower() == "nii":
        dataset = _create_nii_dataset(
            filepaths=filepaths,
            key_featureimg=key_featureimg,
            key_labelimg=key_labelimg,
            block_lengths=block_lengths,
            cyclic=cyclic,
            output_res=output_res,
            output_shape=output_shape,
            slice_dim=slice_dim,
            shuffle=shuffle,
        )
    else:
        # msg = "generating input_fn from image_type {:} is not supported"
        msg = "unknown image_type {:}"
        msg = msg.format(image_type)
        logger.error(msg)
        raise ValueError(msg)

    # tf.data.Dataset: map
    def _map_func(feature_img, label_img=None, presence_array=None):
        # create feature dict
        features = {"images": feature_img}

        # create label dict
        labels = {}
        if label_img is not None:
            labels["images"] = label_img
        if presence_array is not None:
            labels["presences"] = presence_array

        return (features, labels)

    dataset = dataset.map(_map_func)

    # tf.data.Dataset: shuffle
    if (len(filepaths) > 1) and shuffle:
        s = random.randint(1, 2**31)
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffersize,
            seed=s,
            reshuffle_each_iteration=True
        )

    # tf.data.Dataset: repeat
    dataset = dataset.repeat(epochs)

    # tf.data.Dataset: batch
    dataset = dataset.batch(batchsize)

    # tf.data.Dataset: prefetch
    dataset = dataset.prefetch(prefetch_buffersize)

    # make iterator and return
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()


def _create_hdf5_dataset(
    filepaths,
    key_featureimg,
    key_labelimg=None,
    key_presence=None,
    block_lengths=None,
    cyclic=False,
    slice_dim=0,
    shuffle=False,
):
    logger = utilities.log.create_or_get_logger()

    # define datasets to extract from hdf5
    keys = [key_featureimg]
    if key_labelimg is not None:
        keys += [key_labelimg]
    if key_presence is not None:
        keys += [key_presence]

    # define output types
    output_types = [tf.float32]
    if key_labelimg is not None:
        output_types += [tf.uint8]
    if key_presence is not None:
        output_types += [tf.uint8]

    # create generator
    if len(filepaths) == 1:
        generator = reader.HDF5Generator(
            filepaths[0], keys=keys, slice_dim=slice_dim
        )
    elif len(filepaths) > 1:
        hdf5_generators = [
            reader.HDF5Generator(
                f, keys=keys, slice_dim=slice_dim, shuffle=shuffle
            ) for f in filepaths
        ]

        generator = reader.InterleavedGenerator(
            generators=hdf5_generators,
            block_lengths=block_lengths,
            cyclic=cyclic,
        )
    else:
        msg = "at least one hdf5 filename has to be specified!"
        logger.error(msg)
        raise ValueError(msg)

    # determine output shapes
    dataset_shapes = generator.get_dataset_shapes()
    output_shapes = [tf.TensorShape(dataset_shapes[k]) for k in keys]

    # create tf.data.Dataset
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=tuple(output_types),
        output_shapes=tuple(output_shapes),
    )

    return dataset


def _create_nii_dataset(
    filepaths,
    key_featureimg,
    key_labelimg=None,
    block_lengths=None,
    cyclic=False,
    output_res=None,
    output_shape=None,
    slice_dim=0,
    shuffle=False,
):
    logger = utilities.log.create_or_get_logger()

    # define subdirectories (of each filepath)
    # containing required nifti images
    keys = [key_featureimg]
    if key_labelimg is not None:
        keys += [key_labelimg]

    # define output types
    output_types = [tf.float32]
    if key_labelimg is not None:
        output_types += [tf.uint8]

    # define resample orders
    # resample_orders = [3]
    resample_orders = [1]
    if key_labelimg is not None:
        resample_orders += [0]

    # create generator
    if len(filepaths) == 1:
        generator = reader.NiftiGenerator(
            filepaths[0],
            keys=keys,
            resample_orders=resample_orders,
            output_res=output_res,
            output_shape=output_shape,
            slice_dim=slice_dim
        )
    elif len(filepaths) > 1:
        nii_generators = [
            reader.NiftiGenerator(
                f,
                keys=keys,
                resample_orders=resample_orders,
                output_res=output_res,
                output_shape=output_shape,
                slice_dim=slice_dim,
                shuffle=shuffle
            ) for f in filepaths
        ]

        generator = reader.InterleavedGenerator(
            generators=nii_generators,
            block_lengths=block_lengths,
            cyclic=cyclic,
        )
    else:
        msg = "at least one directory (containing nifti images)"
        msg += "has to be specified!"
        logger.error(msg)
        raise ValueError(msg)

    # determine output shapes
    dataset_shapes = generator.get_dataset_shapes()
    output_shapes = [tf.TensorShape(dataset_shapes[k]) for k in keys]

    # create tf.data.Dataset
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=tuple(output_types),
        output_shapes=tuple(output_shapes),
    )

    return dataset
