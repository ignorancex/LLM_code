# Written by Chun Kit Wong and CIRC under MIT license:
# https://github.com/wong-ck/DeepSegment/blob/master/LICENSE

import os
import sys
import argparse

if sys.version_info[0] == 2:
    import ConfigParser as configparser
else:
    import configparser


# helper function to parse string to boolean
def _str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# helper function to create and configure argparse
def _create_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        action="store",
        required=True,
        dest="config",
        help="path to config file (.ini)",
    )

    # general
    parser.add_argument(
        "--log-level",
        action="store",
        type=str,
        required=False,
        dest="general_log_level",
        help="logging level",
        metavar="LEVEL",
    )

    parser.add_argument(
        "--random-seed",
        action="store",
        type=int,
        required=False,
        dest="general_random_seed",
        help="random seed for tf.set_random_seed()",
        metavar="SEED",
    )

    parser.add_argument(
        "--iteration-unit",
        action="store",
        type=str,
        choices=["epoch", "step"],
        required=False,
        dest="general_iteration_unit",
        help="unit of train_iteration and evaluate_freq",
    )

    # image
    parser.add_argument(
        "--image-paths",
        action="store",
        type=str,
        nargs='+',
        required=False,
        dest="image_paths",
        help="paths to hdf5 files (if image_type is 'hdf5') " +
        "or directories containing nifti images (if image_type is 'nii')",
        metavar="PATH",
    )

    parser.add_argument(
        "--image-blocklengths",
        action="store",
        type=int,
        nargs='+',
        required=False,
        dest="image_blocklengths",
        help="number of consecutive slices from each image_path " +
        "when interleaving data from image_paths",
        metavar="BLOCKLENGTH",
    )

    parser.add_argument(
        "--image-cyclic",
        action="store",
        type=_str2bool,
        required=False,
        dest="image_cyclic",
        help="if True, reset any iterator that is depleted " +
        "before every other iterator had depleted at least once " +
        "when interleaving data from image_paths",
        metavar="CYCLIC",
    )

    parser.add_argument(
        "--image-type",
        action="store",
        type=str,
        choices=["hdf5", "nii"],
        required=False,
        dest="image_type",
        help="type of image file/directory",
        metavar="TYPE",
    )

    parser.add_argument(
        "--image-size",
        action="store",
        type=int,
        nargs='+',
        required=False,
        dest="image_size",
        help="list of image sizes at every dimension " +
        "except the slicing dimension " +
        "(image slices are resized to this if image_type is 'nii')",
        metavar="SIZE",
    )

    parser.add_argument(
        "--image-slice-dimension",
        action="store",
        type=int,
        required=False,
        dest="image_slicedim",
        help="slice image at this dimension",
        metavar="DIM",
    )

    parser.add_argument(
        "--image-resolution",
        action="store",
        type=str,
        nargs='+',
        required=False,
        dest="image_resolution",
        help="resolution to resample images to " +
        "(works only if image_type is 'nii')",
        metavar="RES",
    )

    parser.add_argument(
        "--image-nclasses",
        action="store",
        type=int,
        required=False,
        dest="image_nclasses",
        help="number of classes in label image",
        metavar="NCLASSES",
    )

    parser.add_argument(
        "--image-classnames",
        action="store",
        type=str,
        nargs='+',
        required=False,
        dest="image_classnames",
        help="list of names of each class",
        metavar="CLASSNAME",
    )

    parser.add_argument(
        "--image-classweights",
        action="store",
        type=float,
        nargs='+',
        required=False,
        dest="image_classweights",
        help="list of weightages of each class",
        metavar="CLASSWEIGHT",
    )

    parser.add_argument(
        "--image-batchsize",
        action="store",
        type=int,
        required=False,
        dest="image_batchsize",
        help="mini-batch's size",
        metavar="BATCHSIZE",
    )

    parser.add_argument(
        "--image-shuffle-buffer",
        action="store",
        type=int,
        required=False,
        dest="image_shuffle_buffer",
        help="size of buffer used in storing shuffled images temporarily",
        metavar="SIZE",
    )

    # network
    parser.add_argument(
        "--network-model",
        action="store",
        type=str,
        required=False,
        dest="network_model",
        help="name of network model to use",
        metavar="MODEL",
    )

    parser.add_argument(
        "--network-batchnorm",
        action="store",
        type=_str2bool,
        required=False,
        dest="network_batchnorm",
        help="whether or not to perform batch normalization " +
        "wherever applicable in model",
        metavar="BATCHNORM",
    )

    parser.add_argument(
        "--network-dropout",
        action="store",
        type=float,
        required=False,
        dest="network_dropout",
        help="dropout rate, between 0.0 and 1.0; " +
        "set value to 0.0 to disable drop-out",
        metavar="DROPOUT",
    )

    # train
    parser.add_argument(
        "--train-imgkey-feature",
        action="store",
        type=str,
        required=False,
        dest="train_imgkey_feature",
        help="generator's key for feature image during model training; " +
        "if image_type is 'hdf5', this is name of the dataset within " +
        "the .hdf5 file (specified via image_path); if image_type is 'nii', " +
        "this is the subdirectory under image_path",
        metavar="KEY",
    )

    parser.add_argument(
        "--train-imgkey-label",
        action="store",
        type=str,
        required=False,
        dest="train_imgkey_label",
        help="generator's key for label image during model training " +
        "(see --train-imgkey-feature for details)",
        metavar="KEY",
    )

    parser.add_argument(
        "--train-imgkey-presence",
        action="store",
        type=str,
        required=False,
        dest="train_imgkey_presence",
        help="generator's key for presence array during model training " +
        "(see --train-imgkey-feature for details)",
        metavar="KEY",
    )

    parser.add_argument(
        "--train-iterations",
        action="store",
        type=int,
        required=False,
        dest="train_iterations",
        help="number of iterations (epochs/steps) to train network for",
        metavar="NITERS",
    )

    parser.add_argument(
        "--train-learning-rate",
        action="store",
        type=float,
        required=False,
        dest="train_learning_rate",
        help="optimizer's learning rate",
        metavar="LEARNINGRATE",
    )

    parser.add_argument(
        "--train-l2-reg-factor",
        action="store",
        type=float,
        required=False,
        dest="train_l2_reg_factor",
        help="scale/weightage of L2 regularization",
        metavar="L2SCALE",
    )

    parser.add_argument(
        "--train-optimizer",
        action="store",
        type=str,
        required=False,
        dest="train_optimizer",
        help="name of optimizer",
        metavar="OPTIMIZER",
    )

    parser.add_argument(
        "--train-loss",
        action="store",
        type=str,
        required=False,
        dest="train_loss",
        help="name of loss function",
        metavar="LOSS",
    )

    parser.add_argument(
        "--train-weightage-mask",
        action="store",
        type=str,
        required=False,
        dest="train_weightage_mask",
        help="name of weightage mask's mode",
        metavar="MODE",
    )

    parser.add_argument(
        "--train-shuffle-image",
        action="store",
        type=_str2bool,
        required=False,
        dest="train_shuffle_image",
        help="whether or not to shuffle image " +
        "before feeding to network during model training",
        metavar="SHUFFLE",
    )

    # evaluate
    parser.add_argument(
        "--evaluate-imgkey-feature",
        action="store",
        type=str,
        required=False,
        dest="evaluate_imgkey_feature",
        help="generator's key for feature image during model evaluation " +
        "(see --train-imgkey-feature for details)",
        metavar="KEY",
    )

    parser.add_argument(
        "--evaluate-imgkey-label",
        action="store",
        type=str,
        required=False,
        dest="evaluate_imgkey_label",
        help="generator's key for label image during model evaluation " +
        "(see --train-imgkey-feature for details)",
        metavar="KEY",
    )

    parser.add_argument(
        "--evaluate-imgkey-presence",
        action="store",
        type=str,
        required=False,
        dest="evaluate_imgkey_presence",
        help="generator's key for presence array during model evaluation " +
        "(see --train-imgkey-feature for details)",
        metavar="KEY",
    )

    parser.add_argument(
        "--evaluate-freq",
        action="store",
        type=int,
        required=False,
        dest="evaluate_freq",
        help="evaluation frequency; specify in unit of iterations",
        metavar="FREQ",
    )

    parser.add_argument(
        "--evaluate-shuffle-image",
        action="store",
        type=_str2bool,
        required=False,
        dest="evaluate_shuffle_image",
        help="whether or not to shuffle image before " +
        "feeding to network during model evaluation",
        metavar="SHUFFLE",
    )

    # predict
    parser.add_argument(
        "--predict-imgkey-feature",
        action="store",
        type=str,
        required=False,
        dest="predict_imgkey_feature",
        help="generator's key for feature image during prediction " +
        "(see --train-imgkey-feature for details)",
        metavar="KEY",
    )

    parser.add_argument(
        "--predict-output-path",
        action="store",
        type=str,
        required=False,
        dest="predict_output_path",
        help="paths to hdf5 file (if image_type is 'hdf5') " +
        "or directory (if image_type is 'nii') for saving predictions",
        metavar="PATH",
    )

    parser.add_argument(
        "--predict-output-type",
        action="store",
        type=str,
        choices=["labels", "probabilities"],
        required=False,
        dest="predict_output_type",
        help="type of prediction image to save",
        metavar="TYPE",
    )

    # checkpoint
    parser.add_argument(
        "--checkpoint-path",
        action="store",
        type=str,
        required=False,
        dest="checkpoint_path",
        help="path to directory for saving checkpoints " +
        "(model parameters, graph, etc)",
        metavar="PATH",
    )

    parser.add_argument(
        "--checkpoint-freq",
        action="store",
        type=int,
        required=False,
        dest="checkpoint_freq_steps",
        help="checkpointing frequency; specify as multiples of steps",
        metavar="FREQ",
    )

    parser.add_argument(
        "--checkpoint-keep-max",
        action="store",
        type=int,
        required=False,
        dest="checkpoint_keep_max",
        help="maximum number of checkpoints to be kept",
        metavar="KEEPMAX",
    )

    # summary
    parser.add_argument(
        "--summary-freq",
        action="store",
        type=int,
        required=False,
        dest="summary_freq_steps",
        help="frequency of tf.summary creation; specify as multiples of steps",
        metavar="FREQ",
    )

    parser.add_argument(
        "--summary-nimages",
        action="store",
        type=int,
        required=False,
        dest="summary_nimages",
        help="maximum number of images to be summarized " +
        "at every call to tf.summary.image",
        metavar="NIMAGES",
    )

    parser.add_argument(
        "--summary-tensors",
        action="store",
        default=argparse.SUPPRESS,
        type=str,
        nargs='+',
        required=False,
        dest="summary_tensors",
        help="list of names of tensor to be summarized " +
        "(using tf.summary.scalar)",
        metavar="TENSOR",
    )

    return parser


def get_rootdir():
    rootdir = os.path.join(os.path.dirname(__file__), "..")
    rootdir = os.path.abspath(rootdir)

    return rootdir


def load_configuration(args=None):
    # create configuration object
    ds_config = DeepSegmentConfig()

    # obtain names of config defined in DeepSegmentConfig
    ds_config_names = ds_config.get_config_names()

    # load configuration from args
    parser = _create_argparser()
    config_cl = parser.parse_args(args)

    # load configuration from config file
    path_ini = config_cl.config

    config_ini = configparser.ConfigParser()
    config_ini.read(path_ini)

    for section in config_ini.sections():
        for key in config_ini[section]:
            # create config name and extract config value
            config_name = section + '_' + key
            config_val = config_ini.get(section, key)

            # make sure config_name is defined in DeepSegmentConfig
            if config_name not in ds_config_names:
                msg = "{:} not defined in DeepSegmentConfig"
                msg = msg.format(config_name)
                raise ValueError(msg)

            # set config value
            setattr(ds_config, config_name, config_val)

    # override configuration using command line options
    for config_name in vars(config_cl):
        # skip path to config.ini
        if config_name == "config":
            continue

        # make sure config_name is defined in DeepSegmentConfig
        if config_name not in ds_config_names:
            msg = "{:} not defined in DeepSegmentConfig"
            msg = msg.format(config_name)
            raise ValueError(msg)

        # extract and set config value
        config_val = getattr(config_cl, config_name)

        if config_val is not None:
            setattr(ds_config, config_name, config_val)

    # verify consistency of configs
    ds_config.check_consistency()

    return ds_config


class DeepSegmentConfig(object):
    def __init__(self, *args, **kwargs):
        super(DeepSegmentConfig, self).__init__(*args, **kwargs)

        # initialize all class property to None
        ds_config_names = self.get_config_names()

        for config_name in ds_config_names:
            setattr(self, config_name, None)

    def get_config_names(self):
        return [
            p for p in dir(self.__class__)
            if isinstance(getattr(self.__class__, p), property)
        ]

    def check_consistency(self):
        self._verify_image_labelproperties_consistency()

    # general properties
    @property
    def general_log_level(self):
        return self._general_log_level

    @general_log_level.setter
    def general_log_level(self, var):
        if var is None:
            log_level = None
        elif isinstance(var, str):
            allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if var.upper() not in allowed:
                raise ValueError("unknown log level '{:}'".format(var))
            else:
                log_level = var.upper()
        elif isinstance(var, int):
            allowed = [10, 20, 30, 40, 50]
            if var not in allowed:
                raise ValueError("unknown log level '{:}'".format(var))
            else:
                log_level = var
        else:
            raise TypeError("log level must be specified in int or string")

        self._general_log_level = log_level

    @property
    def general_random_seed(self):
        return self._general_random_seed

    @general_random_seed.setter
    def general_random_seed(self, var):
        self._general_random_seed = int(var) if (var is not None) else var

    @property
    def general_iteration_unit(self):
        return self._general_iteration_unit

    @general_iteration_unit.setter
    def general_iteration_unit(self, var):
        allowed = ["epoch", "epochs", "step", "steps"]

        if var is None:
            iteration_unit = None
        elif not isinstance(var, str):
            msg = "iteration_unit must be a string"
            raise TypeError(msg)
        elif var.lower() not in allowed:
            # check that iteration_unit is allowed
            msg = "iteration_unit must be either of these: {:}"
            msg = msg.format(allowed)
            raise ValueError(msg)
        else:
            iteration_unit = var

        # if plural, convert to singular
        if iteration_unit == "epochs":
            iteration_unit = "epoch"
        if iteration_unit == "steps":
            iteration_unit = "step"

        self._general_iteration_unit = iteration_unit

    # image properties
    @property
    def image_paths(self):
        return self._image_paths

    @image_paths.setter
    def image_paths(self, var):
        if var is None:
            image_paths = None
        elif isinstance(var, str):
            # split at comma
            image_paths = tuple([x.strip() for x in var.split(',')])
        elif isinstance(var, list):
            # convert values to str
            image_paths = tuple([str(x).strip() for x in var])
        else:
            msg = "image_paths must be either "
            msg += "a string of comma-separated paths "
            msg += "or a list of paths"
            raise TypeError(msg)

        self._image_paths = image_paths

    @property
    def image_blocklengths(self):
        return self._image_blocklengths

    @image_blocklengths.setter
    def image_blocklengths(self, var):
        if var is None:
            blocklengths = None
        elif isinstance(var, str):
            # remove brackets
            blocklengths = var
            for x in ['(', '[', ']', ')']:
                blocklengths = blocklengths.replace(x, '')

            if len(blocklengths) == 0:
                blocklengths = None
            else:
                # split at comma; convert values into int
                blocklengths = tuple(
                    [int(x.strip()) for x in blocklengths.split(',')]
                )
        elif isinstance(var, list):
            blocklengths = var

            if len(blocklengths) == 0:
                blocklengths = None
            else:
                # convert values to int
                blocklengths = tuple(
                    [int(str(x).strip()) for x in blocklengths]
                )
        else:
            msg = "image_blocklengths must be either "
            msg += "a string of comma-separated integers "
            msg += "or a list of strings/integers"
            raise TypeError(msg)

        self._image_blocklengths = blocklengths

    @property
    def image_cyclic(self):
        return self._image_cyclic

    @image_cyclic.setter
    def image_cyclic(self, var):
        if var is None:
            cyclic = None
        else:
            cyclic = str(var)

            if not cyclic:
                cyclic = None
            else:
                cyclic = _str2bool(cyclic)

        self._image_cyclic = cyclic

    @property
    def image_type(self):
        return self._image_type

    @image_type.setter
    def image_type(self, var):
        if var is None:
            image_type = None
        else:
            # check that image_type is allowed
            allowed = ["hdf5", "nii"]
            if str(var).lower() not in allowed:
                msg = "image_type must be either of these: {:}"
                msg = msg.format(allowed)
                raise ValueError(msg)

            image_type = str(var).lower() if (var is not None) else var

        self._image_type = image_type

    @property
    def image_size(self):
        return self._image_size

    @image_size.setter
    def image_size(self, var):
        if var is None:
            image_size = None
        elif isinstance(var, str):
            # remove brackets
            image_size = var
            for x in ['(', '[', ']', ')']:
                image_size = image_size.replace(x, '')

            if len(image_size) == 0:
                msg = "invalid image_size '{:}'".format(var)
                raise ValueError(msg)

            # split at comma; convert values into int
            image_size = tuple([int(x.strip()) for x in image_size.split(',')])
        elif isinstance(var, list):
            image_size = var

            if len(image_size) == 0:
                msg = "invalid image_size '{:}'".format(var)
                raise ValueError(msg)

            # convert values to int
            image_size = tuple([int(str(x).strip()) for x in image_size])
        else:
            msg = "image_size must be either "
            msg += "a string of comma-separated integers "
            msg += "or a list of strings/integers"
            raise TypeError(msg)

        self._image_size = image_size

    @property
    def image_slicedim(self):
        return self._image_slicedim

    @image_slicedim.setter
    def image_slicedim(self, var):
        self._image_slicedim = int(var) if (var is not None) else var

    @property
    def image_resolution(self):
        return self._image_resolution

    @image_resolution.setter
    def image_resolution(self, var):
        if var is None:
            img_res = None
        elif isinstance(var, str):
            # remove brackets
            img_res = var
            for x in ['(', '[', ']', ')']:
                img_res = img_res.replace(x, '')

            if len(img_res) == 0:
                msg = "invalid image_resolution '{:}'".format(var)
                raise ValueError(msg)

            # split at comma
            img_res = img_res.split(',')
        elif isinstance(var, list):
            img_res = var
        else:
            msg = "image_resolution must be either "
            msg += "a string of comma-separated floating point values "
            msg += "or a list of strings/floating point values"
            raise TypeError(msg)

        if img_res is not None:
            # check that img_res is not an empty list
            if len(img_res) == 0:
                msg = "invalid image_resolution '{:}'".format(var)
                raise ValueError(msg)

            # remove any spaces at beginning/end of string
            img_res = [str(x).strip() for x in img_res]

            # attempt to convert values to float
            for i, _ in enumerate(img_res):
                if img_res[i].lower() == 'none':
                    img_res[i] = None
                else:
                    img_res[i] = float(img_res[i])

            img_res = tuple(img_res)

        self._image_resolution = img_res

    @property
    def image_nclasses(self):
        return self._image_nclasses

    @image_nclasses.setter
    def image_nclasses(self, var):
        self._image_nclasses = int(var) if (var is not None) else var
        # self._verify_image_labelproperties_consistency()

    @property
    def image_classnames(self):
        return self._image_classnames

    @image_classnames.setter
    def image_classnames(self, var):
        if var is None:
            classnames = None
        elif isinstance(var, str):
            classnames = [x.strip() for x in var.split(',')]
        elif isinstance(var, list):
            classnames = [str(x).strip() for x in var]
        else:
            msg = "image_classnames must be either "
            msg += "a string of comma-separated values "
            msg += "or a list of strings"
            raise TypeError(msg)

        self._image_classnames = classnames

        # self._verify_image_labelproperties_consistency()

    @property
    def image_classweights(self):
        return self._image_classweights

    @image_classweights.setter
    def image_classweights(self, var):
        if var is None:
            weights = None
        elif isinstance(var, str):
            # split at comma; convert values into float
            weights = [float(x.strip()) for x in var.split(',')]
        elif isinstance(var, list):
            # convert values into float
            weights = [float(str(x).strip()) for x in var]
        else:
            msg = "image_classweights must be either "
            msg += "a string of comma-separated values "
            msg += "or a list of strings/numbers"
            raise TypeError(msg)

        self._image_classweights = weights
        # self._verify_image_labelproperties_consistency()

    def _verify_image_labelproperties_consistency(self):
        # determine if properties are set
        defined_nclasses = False
        if hasattr(self, "image_nclasses"):
            defined_nclasses = (self.image_nclasses is not None)

        defined_names = False
        if hasattr(self, "image_classnames"):
            defined_names = (self.image_classnames is not None)

        defined_weights = False
        if hasattr(self, "image_classweights"):
            defined_weights = (self.image_classweights is not None)

        # verify pairwise consistency
        if defined_nclasses and defined_names:
            assert (len(self.image_classnames) == self.image_nclasses)

        if defined_nclasses and defined_weights:
            assert (len(self.image_classweights) == self.image_nclasses)

        if defined_names and defined_weights:
            assert (len(self.image_classnames) == len(self.image_classweights))

    @property
    def image_batchsize(self):
        return self._image_batchsize

    @image_batchsize.setter
    def image_batchsize(self, var):
        self._image_batchsize = int(var) if (var is not None) else var

    @property
    def image_shuffle_buffer(self):
        return self._image_shuffle_buffer

    @image_shuffle_buffer.setter
    def image_shuffle_buffer(self, var):
        self._image_shuffle_buffer = int(var) if (var is not None) else var

    # network properties
    @property
    def network_model(self):
        return self._network_model

    @network_model.setter
    def network_model(self, var):
        self._network_model = str(var) if (var is not None) else var

    @property
    def network_batchnorm(self):
        return self._network_batchnorm

    @network_batchnorm.setter
    def network_batchnorm(self, var):
        bn = _str2bool(str(var)) if (var is not None) else var
        self._network_batchnorm = bn

    @property
    def network_dropout(self):
        return self._network_dropout

    @network_dropout.setter
    def network_dropout(self, var):
        self._network_dropout = float(var) if (var is not None) else var

    # train properties
    @property
    def train_imgkey_feature(self):
        return self._train_imgkey_feature

    @train_imgkey_feature.setter
    def train_imgkey_feature(self, var):
        self._train_imgkey_feature = str(var) if (var is not None) else var

    @property
    def train_imgkey_label(self):
        return self._train_imgkey_label

    @train_imgkey_label.setter
    def train_imgkey_label(self, var):
        self._train_imgkey_label = str(var) if (var is not None) else var

    @property
    def train_imgkey_presence(self):
        return self._train_imgkey_presence

    @train_imgkey_presence.setter
    def train_imgkey_presence(self, var):
        self._train_imgkey_presence = str(var) if (var is not None) else var

    @property
    def train_iterations(self):
        return self._train_iterations

    @train_iterations.setter
    def train_iterations(self, var):
        self._train_iterations = int(var) if (var is not None) else var

    @property
    def train_learning_rate(self):
        return self._train_learning_rate

    @train_learning_rate.setter
    def train_learning_rate(self, var):
        self._train_learning_rate = float(var) if (var is not None) else var

    @property
    def train_l2_reg_factor(self):
        return self._train_l2_reg_factor

    @train_l2_reg_factor.setter
    def train_l2_reg_factor(self, var):
        self._train_l2_reg_factor = float(var) if (var is not None) else var

    @property
    def train_optimizer(self):
        return self._train_optimizer

    @train_optimizer.setter
    def train_optimizer(self, var):
        self._train_optimizer = str(var) if (var is not None) else var

    @property
    def train_loss(self):
        return self._train_loss

    @train_loss.setter
    def train_loss(self, var):
        self._train_loss = str(var) if (var is not None) else var

    @property
    def train_weightage_mask(self):
        return self._train_weightage_mask

    @train_weightage_mask.setter
    def train_weightage_mask(self, var):
        self._train_weightage_mask = str(var) if (var is not None) else var

    @property
    def train_shuffle_image(self):
        return self._train_shuffle_image

    @train_shuffle_image.setter
    def train_shuffle_image(self, var):
        si = _str2bool(str(var)) if (var is not None) else var
        self._train_shuffle_image = si

    # evaluate properties
    @property
    def evaluate_imgkey_feature(self):
        return self._evaluate_imgkey_feature

    @evaluate_imgkey_feature.setter
    def evaluate_imgkey_feature(self, var):
        self._evaluate_imgkey_feature = str(var) if (var is not None) else var

    @property
    def evaluate_imgkey_label(self):
        return self._evaluate_imgkey_label

    @evaluate_imgkey_label.setter
    def evaluate_imgkey_label(self, var):
        self._evaluate_imgkey_label = str(var) if (var is not None) else var

    @property
    def evaluate_imgkey_presence(self):
        return self._evaluate_imgkey_presence

    @evaluate_imgkey_presence.setter
    def evaluate_imgkey_presence(self, var):
        self._evaluate_imgkey_presence = str(var) if (var is not None) else var

    @property
    def evaluate_freq(self):
        return self._evaluate_freq

    @evaluate_freq.setter
    def evaluate_freq(self, var):
        self._evaluate_freq = int(var) if (var is not None) else var

    @property
    def evaluate_shuffle_image(self):
        return self._evaluate_shuffle_image

    @evaluate_shuffle_image.setter
    def evaluate_shuffle_image(self, var):
        si = _str2bool(str(var)) if (var is not None) else var
        self._evaluate_shuffle_image = si

    # predict properties
    @property
    def predict_imgkey_feature(self):
        return self._predict_imgkey_feature

    @predict_imgkey_feature.setter
    def predict_imgkey_feature(self, var):
        self._predict_imgkey_feature = str(var) if (var is not None) else var

    @property
    def predict_output_path(self):
        return self._predict_output_path

    @predict_output_path.setter
    def predict_output_path(self, var):
        self._predict_output_path = str(var) if (var is not None) else var

    @property
    def predict_output_type(self):
        return self._predict_output_type

    @predict_output_type.setter
    def predict_output_type(self, var):
        if var is None:
            output_type = None
        else:
            # check that predict_output_type is allowed
            allowed = ["labels", "probabilities"]
            if str(var).lower() not in allowed:
                msg = "predict_output_type must be either of these: {:}"
                msg = msg.format(allowed)
                raise ValueError(msg)

            output_type = str(var).lower() if (var is not None) else var

        self._predict_output_type = output_type

    # checkpoint properties
    @property
    def checkpoint_path(self):
        return self._checkpoint_path

    @checkpoint_path.setter
    def checkpoint_path(self, var):
        self._checkpoint_path = str(var) if (var is not None) else var

    @property
    def checkpoint_freq_steps(self):
        return self._checkpoint_freq_steps

    @checkpoint_freq_steps.setter
    def checkpoint_freq_steps(self, var):
        self._checkpoint_freq_steps = int(var) if (var is not None) else var

    @property
    def checkpoint_keep_max(self):
        return self._checkpoint_keep_max

    @checkpoint_keep_max.setter
    def checkpoint_keep_max(self, var):
        self._checkpoint_keep_max = int(var) if (var is not None) else var

    # summary properties
    @property
    def summary_freq_steps(self):
        return self._summary_freq_steps

    @summary_freq_steps.setter
    def summary_freq_steps(self, var):
        self._summary_freq_steps = int(var) if (var is not None) else var

    @property
    def summary_nimages(self):
        return self._summary_nimages

    @summary_nimages.setter
    def summary_nimages(self, var):
        self._summary_nimages = int(var) if (var is not None) else var

    @property
    def summary_tensors(self):
        return self._summary_tensors

    @summary_tensors.setter
    def summary_tensors(self, var):
        if var is None:
            summary_tensors = None
        elif isinstance(var, str):
            summary_tensors = [t for t in var.split(',') if bool(t.strip())]
        elif isinstance(var, list):
            summary_tensors = [t for t in var if bool(t.strip())]
        else:
            msg = "summary_tensors must be either "
            msg += "a string of comma-separated values "
            msg += "or a list of strings"
            raise TypeError(msg)

        self._summary_tensors = summary_tensors
