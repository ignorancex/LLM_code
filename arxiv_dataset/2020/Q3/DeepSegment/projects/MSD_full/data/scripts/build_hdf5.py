# Written by Chun Kit Wong and CIRC under MIT license:
# https://github.com/wong-ck/DeepSegment/blob/master/LICENSE

import os
import sys

import nibabel as nib

import numpy as np
import skimage
from skimage import transform

# search for utilities module under root dir
DIR_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
DIR_ROOT = os.path.abspath(DIR_ROOT)

sys.path.insert(0, DIR_ROOT)
import utilities
from utilities.io.writer import HDF5Writer
sys.path.pop(0)

# define paths
DIR_NII = os.path.join(os.path.dirname(__file__), "..", "nii")
DIR_NII = os.path.abspath(DIR_NII)

DIR_HDF5 = os.path.join(os.path.dirname(__file__), "..", "hdf5")
DIR_HDF5 = os.path.abspath(DIR_HDF5)

# define parameters controlling script's performance
HDF5_RESIZE_CHUNK = 500
HDF5_WRITE_FREQ = 20  # 300

# define list of slice directions and other image properties
RESAMPLE_CONFIGS = []
RESAMPLE_CONFIGS += [
    {
        "slice_dim": 0,
        "slice_dim_label": "sagittal",
        "target_size": (256, 256),
        "target_res": (3.5, 1.5, 1.5),
        "crop": True
    }
]
RESAMPLE_CONFIGS += [
    {
        "slice_dim": 1,
        "slice_dim_label": "coronal",
        "target_size": (256, 256),
        "target_res": (1.5, 3.5, 1.5),
        "crop": True
    }
]
RESAMPLE_CONFIGS += [
    {
        "slice_dim": 2,
        "slice_dim_label": "axial",
        "target_size": (256, 256),
        "target_res": (1.5, 1.5, 3.5),
        "crop": True
    }
]

# define datasets and the correspoding label map
DATASETS = []
DATASETS += [
    {
        "name": "Liver",
        "src_dir": os.path.join(DIR_NII, "Task03_Liver"),
        "dest_dir": DIR_HDF5,
        "label_map": {
            0: 0,
            1: 1,
            2: 2
        },
        "max_subj": {
            "train": None,
            "test": None
        }
    }
]
DATASETS += [
    {
        "name": "Pancreas",
        "src_dir": os.path.join(DIR_NII, "Task07_Pancreas"),
        "dest_dir": DIR_HDF5,
        "label_map": {
            0: 0,
            1: 3,
            2: 4
        },
        "max_subj": {
            "train": None,
            "test": None
        }
    }
]
DATASETS += [
    {
        "name": "Spleen",
        "src_dir": os.path.join(DIR_NII, "Task09_Spleen"),
        "dest_dir": DIR_HDF5,
        "label_map": {
            0: 0,
            1: 5
        },
        "max_subj": {
            "train": None,
            "test": None
        }
    }
]


def process_subject(
    nib_img, nib_label, target_res, slice_size, slice_dim, crop
):
    # create logger
    logger = utilities.log.create_or_get_logger()

    # basic checking
    if len(slice_size) != 2:
        msg = "  " * 4
        msg += "- unexpected dimension of slice_size ({:}, expected 2)"
        logger.error(msg.format(len(slice_size)))

        return None

    # get image data
    np_img = nib_img.get_data().copy()
    np_label = nib_label.get_data().copy()

    # basic checking
    if not all(x == y for x, y in zip(np_img.shape, np_label.shape)):
        logger.error("  " * 4 + "- unexpected size mismatch:")
        logger.error("  " * 5 + "- image: {:}".format(np_img.shape))
        logger.error("  " * 5 + "- label: {:}".format(np_label.shape))

        return None

    logger.debug("  " * 4 + "- original shape: " + str(np_img.shape))

    # extract image info
    nii_res = nib_img.header.structarr['pixdim'][1:4]
    nii_size = np_img.shape

    # basic checking
    if len(nii_size) != 3:
        msg = "  " * 4
        msg += "- unexpected dimension of input image ({:}, expected 3)"
        msg = msg.format(len(nii_size))
        logger.error(msg)

        return None

    # resample to match resolution
    # feature image
    np_img = utilities.image.resample_image(
        np_img, nii_res, target_res, order=3, disable_anti_aliasing=False
    )

    # label image
    # - set order = 0 (nearest neighbour):
    #   - should only interpolate with NN while resampling label image
    # - turn off anti-aliasing if scikit-image version >= 0.14:
    #   - AA leads to erroneous behavious in some cases
    np_label = utilities.image.resample_image(
        np_label, nii_res, target_res, order=0, disable_anti_aliasing=True
    )

    logger.debug("  " * 4 + "- after resampling: " + str(np_img.shape))

    # crop/extract slices if necessary
    if crop:
        a = np.nonzero(np_label)
        ndim = len(np_label.shape)

        # determine bounding box for cropping
        bbox = []
        for i in range(ndim):
            if i == slice_dim:
                # crop with no buffer in the slicing direction
                bbox.append(
                    slice(
                        max(np.min(a[i]), 0),
                        min(np.max(a[i]), np_label.shape[i])
                    )
                )
            else:
                # crop with large buffer in all other directions
                a_min = np.min(a[i])
                a_max = np.max(a[i])

                if i < slice_dim:
                    ss = slice_size[i]
                else:
                    ss = slice_size[i - 1]

                if (a_max - a_min) > ss:
                    msg = "  " * 4
                    msg += "- ROI needs larger than {:} to be fully covered"
                    msg += "; consider specifying a lower resolution"
                    logger.warn(msg.format(slice_size))

                    # do not crop
                    # image will be squashed into slice_size in next step
                    bbox.append(slice(None))
                else:
                    # attempt to crop (ss) pixels with ROI being centered
                    a_start = max(
                        a_min + int(0.5 * (a_max - a_min)) - int(0.5 * ss), 0
                    )
                    a_stop = a_start + ss

                    # make sure a_stop doesn't exceed image size
                    if a_stop > np_label.shape[i]:
                        a_start = max(
                            a_start - (a_stop - np_label.shape[i]), 0
                        )
                        a_stop = np_label.shape[i]

                    bbox.append(slice(a_start, a_stop))

        logger.info("  " * 4 + "- cropping with bounding box {:}".format(bbox))

        np_img = np_img[tuple(bbox)]
        np_label = np_label[tuple(bbox)]

        logger.debug("  " * 4 + "- after cropping: " + str(np_img.shape))

    # resize if necessary
    _output_size = list(np_img.shape)
    for i in range(len(_output_size)):
        if i < slice_dim:
            _output_size[i] = slice_size[i]
        elif i > slice_dim:
            _output_size[i] = slice_size[i - 1]

    np_img = utilities.image.crop_or_pad_image(
        np_img, _output_size, background_value=None
    )
    np_label = utilities.image.crop_or_pad_image(
        np_label, _output_size, background_value=0
    )

    logger.debug("  " * 4 + "- after resize: " + str(np_img.shape))

    return (np_img, np_label)


def process_data(resample_config, dataset):
    # create logger
    logger = utilities.log.create_or_get_logger()

    # determine total number of label class
    nclasses_raw = len(set(dataset["label_map"].keys()))

    _labels = [label for dt in DATASETS for label in dt["label_map"].values()]
    nclasses_mapped = len(set(_labels))

    # generate hdf5 filename
    filename_hdf5 = dataset["name"]

    filename_hdf5 += "_" + resample_config["slice_dim_label"]

    filename_hdf5 += "_cropped" if resample_config["crop"] else "_uncropped"

    filename_hdf5 += "_ncls_<ncls>"

    _size_str = [str(s) for s in resample_config["target_size"]]
    _size_str = "x".join(_size_str)
    filename_hdf5 += "_size_" + _size_str

    _res_str = [str(r) for r in resample_config["target_res"]]
    _res_str = "x".join(_res_str)
    filename_hdf5 += "_res_" + _res_str

    if all([v is None for v in dataset["max_subj"].values()]):
        filename_hdf5 += "_nsubj_all"
    else:
        nsubj = 0
        for tt in ["train", "test"]:
            if dataset["max_subj"][tt] is not None:
                nsubj += int(dataset["max_subj"][tt])

        if dataset["max_subj"]["train"] is None:
            nsubj = "all+" + str(nsubj)
        if dataset["max_subj"]["test"] is None:
            nsubj = str(nsubj) + "+all"

        filename_hdf5 += "_nsubj_" + str(nsubj)

    filename_hdf5 += ".hdf5"

    logger.info("  " * 2 + "- hdf5 filename: {:}".format(filename_hdf5))

    # create HDF5Writers
    if not os.path.exists(dataset["dest_dir"]):
        os.makedirs(dataset["dest_dir"])

    hdf5_writer_mapped = HDF5Writer(
        path=os.path.join(
            dataset["dest_dir"],
            filename_hdf5.replace("<ncls>", str(nclasses_mapped))
        ),
        write_freq=HDF5_WRITE_FREQ,
        resize_chunk=HDF5_RESIZE_CHUNK,
    )

    hdf5_writer_raw = HDF5Writer(
        path=os.path.join(
            dataset["dest_dir"],
            filename_hdf5.replace("<ncls>", str(nclasses_raw))
        ),
        write_freq=HDF5_WRITE_FREQ,
        resize_chunk=HDF5_RESIZE_CHUNK,
    )

    for tt in ['test', 'train']:
        # create datasets in HDF5Writer
        for hdf5_writer in [hdf5_writer_mapped, hdf5_writer_raw]:
            hdf5_writer.create_dataset(
                "image_%s" % tt,
                shape=list(resample_config["target_size"]),
                dtype=np.float32
            )
            hdf5_writer.create_dataset(
                "label_%s" % tt,
                shape=list(resample_config["target_size"]),
                dtype=np.uint8
            )

        hdf5_writer_mapped.create_dataset(
            "presence_%s" % tt, shape=[nclasses_mapped], dtype=np.uint8
        )
        hdf5_writer_raw.create_dataset(
            "presence_%s" % tt, shape=[nclasses_raw], dtype=np.uint8
        )

        logger.info("  " * 2 + "- processing %s subjects" % tt)

        # define paths
        dirname_img = os.path.join(dataset["src_dir"], "image_%s" % tt)
        filenames_img = sorted(os.listdir(dirname_img))

        dirname_label = os.path.join(dataset["src_dir"], "label_%s" % tt)
        filenames_label = sorted(os.listdir(dirname_label))

        # loop over each subjects
        counter_subj = 0
        for filename_img, filename_label in zip(
            filenames_img, filenames_label
        ):
            # check breaking condition
            if dataset["max_subj"][tt] is not None:
                if counter_subj >= dataset["max_subj"][tt]:
                    break

            # verify that filenames are the same
            if filename_img != filename_label:
                msg = "  " * 3 + "- unexpected filename mismatch between "
                msg += "image_{:} ({:}) and label_{:} ({:}); skipping..."
                msg = msg.format(tt, filename_img, tt, filename_label)
                logger.error(msg)

                counter_subj += 1
                continue
            else:
                logger.info("  " * 3 + "- " + filename_img)

            # load img and label (nifti)
            fullpath_img = os.path.join(dirname_img, filename_img)
            fullpath_label = os.path.join(dirname_label, filename_label)

            nib_img = nib.load(fullpath_img)
            nib_label = nib.load(fullpath_label)

            # process subject
            results = process_subject(
                nib_img,
                nib_label,
                target_res=resample_config["target_res"],
                slice_size=resample_config["target_size"],
                slice_dim=resample_config["slice_dim"],
                crop=resample_config["crop"]
            )

            if results is None:
                counter_subj += 1
                continue

            (img, label) = results

            # define presence array
            presence_raw = [True] * nclasses_raw

            # map value of label image according to map
            presence_mapped = [False] * nclasses_mapped
            label_mapped = label.copy() * 0

            for s, t in dataset["label_map"].items():
                presence_mapped[t] = True
                label_mapped[label == s] = t

            # write out
            logger.info("  " * 4 + "- writing to hdf5 [{:}]".format(tt))

            bbox_base = [slice(None)] * len(img.shape)
            for islice in range(img.shape[resample_config["slice_dim"]]):
                # create bounding box at current slice
                bbox = list(bbox_base)
                bbox[resample_config["slice_dim"]] = slice(islice, islice + 1)
                bbox = tuple(bbox)

                # write feature image
                hdf5_writer_mapped.write_data(
                    np.squeeze(img[bbox]), 'image_%s' % tt
                )
                hdf5_writer_raw.write_data(
                    np.squeeze(img[bbox]), 'image_%s' % tt
                )

                # write label image
                hdf5_writer_mapped.write_data(
                    np.squeeze(label_mapped[bbox]), 'label_%s' % tt
                )
                hdf5_writer_raw.write_data(
                    np.squeeze(label[bbox]), 'label_%s' % tt
                )

                # write presence array
                hdf5_writer_mapped.write_data(
                    presence_mapped, 'presence_%s' % tt
                )
                hdf5_writer_raw.write_data(presence_raw, 'presence_%s' % tt)

            # increment subject counter
            counter_subj += 1

    # flush remaining data and close hdf5 file
    hdf5_writer_mapped.close()
    hdf5_writer_raw.close()

    return


def main():
    # create logger
    logger = utilities.log.create_or_get_logger()

    # loop over each resample configuration
    for resample_config in RESAMPLE_CONFIGS:
        # print/log info
        logger.info("processing images with configurations:")
        utilities.log.print_dict(resample_config, logger=logger, indent=1)

        # loop over each datasets:
        for dataset in DATASETS:
            # print/log info
            msg = "  " * 1 + "- processing dataset: {:}"
            msg = msg.format(dataset["name"])
            logger.info(msg)
            utilities.log.print_dict(dataset, logger=logger, indent=2)

            # actually process data
            process_data(resample_config, dataset)


if __name__ == "__main__":
    # create logger
    logger = utilities.log.create_or_get_logger(level="DEBUG")

    main()
