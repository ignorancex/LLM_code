"""
The FlagSetting Class definition to set different FLAG values
"""

import tensorflow.compat.v1 as tf
from configparser import ConfigParser
import os
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

_SUPPORT_MODE=('train', 'val', 'test',)
_SUPPORT_DPN_VARIANT=('aspp', 'dense_aspp', 'li_aspp', 'li_dense_aspp','ignore',)
_IMPORTANT_FLAGS_TO_PRINT=(
    'model_variant', 'decoder_output_stride', 'add_image_level_feature',
    'aspp_with_batch_norm', 'base_learning_rate', 'training_number_of_steps','train_batch_size',
    'train_crop_size', 'tf_initial_checkpoint', 'initialize_last_layer', 'last_layer_gradient_multiplier',
    'fine_tune_batch_norm', 'atrous_rates', 'output_stride', 'dataset', 'dataset_dir','train_logdir',
    'train_split', 'val_split', 'test_split', 'val_logdir', 'test_logdir', 'num_clones', 'eval_batch_size',
    'eval_crop_size', 'eval_scales', 'checkpoint_dir', 'max_to_keep', 'checkpoint_path', 'checkpoint_name',
    'allow_growth', 'test_resdir', 'num_plots_to_save', 'freeze_backbone_network', 'last_layers_contain_logits_only',
    'save_interval_secs','save_checkpoint_steps', 'save_summaries_secs', 'save_summaries_steps',
    'min_resize_value', 'max_resize_value', 'resize_factor', 'add_flipped_images', 'keep_all_checkpoints',
    'dpn_variant', 'multi_grid', 'finetune_li_weight_only','li_weight_gradient_multiplier',
    'optimizer', 'adam_learning_rate','adam_epsilon','calculate_model_flops_and_params')


def load_sorted_ck_paths_from_val_result(val_res_path):
    if not os.path.exists(val_res_path):
        raise FileNotFoundError(
            'Cannot find validation result at: {}'.format(val_res_path))
    from scipy.io import loadmat
    val_res = loadmat(val_res_path)
    ck_mious = np.squeeze(val_res['miou'])
    ck_paths = np.squeeze(val_res['ck_path'])
    sort_idx = np.argsort(-ck_mious)
    ck_path_sorted = ck_paths[sort_idx]
    ck_path_sorted = [d.strip() for d in ck_path_sorted]
    best_ck_path = ck_path_sorted[0]
    return best_ck_path, ck_path_sorted


class FlagSetting(object):

    _instance = None
    mode = None
    config_file = None
    config_name = None

    def __new__(cls, mode='train'):

        assert mode in _SUPPORT_MODE, \
            'Unsupport mode {}\nSupported mode: {}'.format(
                mode, _SUPPORT_MODE)

        if cls._instance is None:
            cls._instance = object.__new__(cls)
            cls.mode = mode

            flags.DEFINE_string('config_file', None, 'Where the config file is')

            # if cls.task=='train_deeplabv3plus':

            flags.DEFINE_string('dataset', 'pascal_voc_seg',
                                'Name of the segmentation dataset.')
            flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

            # For aspp as dpn, set atrous_rates = [12,24,36] if os = 8, or
            # rates = [6,12,18] if os = 16. Note one could use different
            # atrous_rates/output_stride during training/evaluation.
            flags.DEFINE_multi_integer('atrous_rates', None,
                                       'Atrous rates for atrous spatial pyramid pooling.')
            flags.DEFINE_integer('output_stride', 16,
                                 'The ratio of input to output spatial resolution.')
            flags.DEFINE_boolean('allow_growth', False,
                                 'Allow gpu memory growth or not')

            flags.DEFINE_boolean('freeze_backbone_network', False,
                                 'Set true to freeze the weights of the backbone during training')

            flags.DEFINE_string(
                'dpn_variant', 'aspp',
                'The variant for the dense prediction network (such as aspp or dense aspp)')

            # flag setting for dilated conv with lateral inhibition
            flags.DEFINE_multi_integer(
                'li_zones_dpn', None, 'The li zones for dilation convs in dpn network, need to be odd integers')
            flags.DEFINE_multi_float(
                'li_decay_stds_dpn', None, 'The li decay stds for dilation convs in dpn network')
            flags.DEFINE_multi_integer(
                'li_rates_dpn', None, 'The li rates for dpn network')

            flags.DEFINE_integer(
                'li_zones_backbone', 3,
                'The li zones for dilation convs in backbone network, need to be odd integers')
            flags.DEFINE_float(
                'li_decay_stds_backbone', 1.0, 'The li decay stds for dilation convs in backbone network')
            flags.DEFINE_integer(
                'li_rates_backbone', 1, 'The li rates for backbone network')
            flags.DEFINE_string('li_location_for_expanded_conv', 'after_input',
                                'Where to place the li layer in expanded_conv block for li_mobilenet_v2 backbone')
            flags.DEFINE_string('li_backbone_option_mnv2', 'stride_1',
                                'Where to place the li layer in expanded_conv block for li_mobilenet_v2 backbone')

            flags.DEFINE_string('li_activation', 'relu', 'The lateral inhibition activation function')
            flags.DEFINE_string(
                'li_weight_initilizer', 'random_uniform', 'The initilizer for lateral inhibition weight')
            # flags.DEFINE_multi_float(
            #     'li_weight_initilizer_min_max', [0.0,0.0], 'The min and max values for li_weight_initilizer')
            flags.DEFINE_multi_float(
                'li_weight_initilizer_min_max_backbone', [0.0,0.0],
                'The min and max values for li_weight_initilizer in dpn network')
            flags.DEFINE_multi_float(
                'li_weight_initilizer_min_max_dpn', [0.0,0.0],
                'The min and max values for li_weight_initilizer in dpn network')
            flags.DEFINE_multi_float(
                'li_weight_clip_values', [0.0,1.0], 'The boundary of lateral inhibition weights')
            flags.DEFINE_boolean('li_with_batch_norm', False, 'Add batch norm for li layer')
            flags.DEFINE_boolean('li_with_weight_regularizer', False, 'Add regualizer for li weight or not')

            flags.DEFINE_string('li_resnet_option', 'a', 'The option to add LI layers in ResNet')
            flags.DEFINE_string('li_resnet_bottleneck_pos', 'after_conv1',
                                'Where to add LI layer in a bottleneck unit')
            flags.DEFINE_string('li_resnet_block_pos', 'last',
                                'Which bottleneck units in the block will have LI layer')
            flags.DEFINE_boolean('li_aspp_name_scope_compatible', False,
                                 'If True, the name scope of LI-ASPP will be aspp instead of '
                                 'li_aspp for compatility with baseline model')

            if cls.mode=='train':
                # Settings for multi-GPUs/multi-replicas training.
                flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy.')
                flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones.')
                flags.DEFINE_integer('num_replicas', 1, 'Number of worker replicas.')
                flags.DEFINE_integer('startup_delay_steps', 15,
                                     'Number of training steps between replicas startup.')
                flags.DEFINE_integer(
                    'num_ps_tasks', 0,
                    'The number of parameter servers. If the value is 0, then '
                    'the parameters are handled locally by the worker.')
                flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')
                flags.DEFINE_integer('task', 0, 'The task ID.')

                # Settings for logging.
                flags.DEFINE_string('train_logdir', None,
                                    'Where the checkpoint and logs are stored.')
                flags.DEFINE_integer('log_steps', 10,
                                     'Display logging information at every log_steps.')
                flags.DEFINE_integer('save_interval_secs', None,
                                     'How often, in seconds, we save the model to disk.')
                flags.DEFINE_integer('save_checkpoint_steps', None,
                                     'How often, in global steps, we save the model to disk.')
                flags.DEFINE_integer('save_summaries_secs', None,
                                     'How often, in seconds, we compute the summaries.')
                flags.DEFINE_integer('save_summaries_steps', None,
                                     'How often, in global steps, we compute the summaries.')
                flags.DEFINE_integer('max_to_keep', None,
                                     'The maximum number of checkpoints to keep')
                flags.DEFINE_boolean(
                    'save_summaries_images', False,
                    'Save sample inputs, labels, and semantic predictions as '
                    'images to summary.')

                # Settings for profiling.
                flags.DEFINE_string('profile_logdir', None,
                                    'Where the profile files are stored.')

                # Settings for training strategy.
                flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'],
                                  'Learning rate policy for training.')

                # Use 0.007 when training on PASCAL augmented training set, train_aug. When
                # fine-tuning on PASCAL trainval set, use learning rate=0.0001.
                flags.DEFINE_float('base_learning_rate', .0001,
                                   'The base learning rate for model training.')
                flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                                   'The rate to decay the base learning rate.')
                flags.DEFINE_integer('learning_rate_decay_step', 2000,
                                     'Decay the base learning rate at a fixed step.')
                flags.DEFINE_float('learning_power', 0.9,
                                   'The power value used in the poly learning policy.')
                flags.DEFINE_integer('training_number_of_steps', 30000,
                                     'The number of steps used for training')
                flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

                # When fine_tune_batch_norm=True, use at least batch size larger than 12
                # (batch size more than 16 is better). Otherwise, one could use smaller batch
                # size and set fine_tune_batch_norm=False.
                flags.DEFINE_integer('train_batch_size', 8,
                                     'The number of images in each batch during training.')

                # For weight_decay, use 0.00004 for MobileNet-V2 or Xcpetion model variants.
                # Use 0.0001 for ResNet model variants.
                flags.DEFINE_float('weight_decay', 0.00004,
                                   'The value of the weight decay for training.')

                # train_crop_size should be k*output_stride+1, where k is manually selected
                flags.DEFINE_list('train_crop_size', '513,513',
                                  'Image crop size [height, width] during training.')
                flags.DEFINE_float(
                    'last_layer_gradient_multiplier', 1.0,
                    'The gradient multiplier for last layers, which is used to '
                    'boost the gradient of last layers if the value > 1.')
                flags.DEFINE_boolean('upsample_logits', True,
                                     'Upsample logits during training.')
                # Hyper-parameters for NAS training strategy.

                flags.DEFINE_float(
                    'drop_path_keep_prob', 1.0,
                    'Probability to keep each path in the NAS cell when training.')

                # Settings for fine-tuning the network.
                flags.DEFINE_string('tf_initial_checkpoint', None,
                                    'The initial checkpoint in tensorflow format.')

                # Set to False if one does not want to re-use the trained classifier weights.
                flags.DEFINE_boolean('initialize_last_layer', True,
                                     'Initialize the last layer.')
                flags.DEFINE_boolean('last_layers_contain_logits_only', False,
                                     'Only consider logits as last layers or not.')
                flags.DEFINE_integer('slow_start_step', 0,
                                     'Training model with small learning rate for few steps.')
                flags.DEFINE_float('slow_start_learning_rate', 1e-4,
                                   'Learning rate employed during slow start.')

                # Set to True if one wants to fine-tune the batch norm parameters in DeepLabv3.
                # Set to False and use small batch size to save GPU memory.
                flags.DEFINE_boolean('fine_tune_batch_norm', True,
                                     'Fine tune the batch norm parameters or not.')
                flags.DEFINE_float('min_scale_factor', 0.5,
                                   'Mininum scale factor for data augmentation.')
                flags.DEFINE_float('max_scale_factor', 2.,
                                   'Maximum scale factor for data augmentation.')
                flags.DEFINE_float('scale_factor_step_size', 0.25,
                                   'Scale factor step size for data augmentation.')

                # Hard example mining related flags.
                flags.DEFINE_integer(
                    'hard_example_mining_step', 0,
                    'The training step in which exact hard example mining kicks off. Note we '
                    'gradually reduce the mining percent to the specified '
                    'top_k_percent_pixels. For example, if hard_example_mining_step=100K and '
                    'top_k_percent_pixels=0.25, then mining percent will gradually reduce from '
                    '100% to 25% until 100K steps after which we only mine top 25% pixels.')
                flags.DEFINE_float(
                    'top_k_percent_pixels', 1.0,
                    'The top k percent pixels (in terms of the loss values) used to compute '
                    'loss during training. This is useful for hard pixel mining.')

                # Quantization setting.
                flags.DEFINE_integer(
                    'quantize_delay_step', -1,
                    'Steps to start quantized training. If < 0, will not quantize model.')

                # Dataset settings.
                flags.DEFINE_string('train_split', 'train',
                                    'Which split of the dataset to be used for training')

                flags.DEFINE_boolean('finetune_li_weight_only', False,
                                     'Only fine-tune LI weights (freezing all other variables)')
                flags.DEFINE_float(
                    'li_weight_gradient_multiplier', 1.0,
                    'The gradient multiplier for lateral inhibition weights')

                flags.DEFINE_string('optimizer', 'sgd', 'Which optimizer to use, sgd or adam')
                flags.DEFINE_float('adam_learning_rate', 0.001,
                                   'Learning rate for the adam optimizer.')
                flags.DEFINE_float('adam_epsilon', 1e-08, 'Adam optimizer epsilon.')

                flags.DEFINE_boolean('display_warning', False,
                                     'Set True to display the warning message durning training.')

            else:
                flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

                # Settings for evaluating the model.
                flags.DEFINE_integer('eval_batch_size', 1,
                                     'The number of images in each batch during evaluation.')

                # eval_crop_size should be k*output_stride+1, where k is manually selected
                flags.DEFINE_list('eval_crop_size', '513,513',
                                  'Image crop size [height, width] for evaluation.')
                # flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                #                      'How often (in seconds) to run evaluation.')

                # Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
                flags.DEFINE_multi_float('eval_scales', [1.0],
                                         'The scales to resize images for evaluation.')

                # Change to True for adding flipped images during test.
                flags.DEFINE_bool('add_flipped_images', False,
                                  'Add flipped images for evaluation or not.')
                flags.DEFINE_integer(
                    'quantize_delay_step', -1,
                    'Steps to start quantized training. If < 0, will not quantize model.')
                flags.DEFINE_string('val_logdir', None, 'Where to write the event logs.')
                flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

                # Dataset settings.
                # flags.DEFINE_integer('max_number_of_evaluations', 0,
                                     # 'Maximum number of eval iterations. Will loop '
                                     # 'indefinitely upon nonpositive values.')
                if cls.mode == 'val':
                    flags.DEFINE_string('val_split', 'val',
                                        'Which split of the dataset used for evaluation')
                    flags.DEFINE_boolean('keep_all_checkpoints', False,
                                        'Whether to keep all train checkpoints. '
                                        'If False, all train checkpoints will be removed except the best and the last.')

                if cls.mode == 'test':
                    flags.DEFINE_string('test_split', 'test',
                                        'Which split of the dataset used for evaluation')
                    flags.DEFINE_string('checkpoint_path', None, 'The checkpoint path to be tested')
                    flags.DEFINE_string('checkpoint_name', None, 'Name of the checkpoint to be tested')
                    flags.DEFINE_string('test_logdir', None, 'Where to write the event logs.')
                    flags.DEFINE_string('test_resdir', None, 'Where to save testing logs.')
                    flags.DEFINE_integer('num_plots_to_save', 0,
                                         'The number of plots to save for visualization')
                    flags.DEFINE_boolean('calculate_model_flops_and_params', False,
                                        'Whether to calculate the FLOPs and parameters of the model')
                    flags.DEFINE_boolean('save_every_frame_miou', False,
                                        'Whether to save the mious for every frame')
                    flags.DEFINE_string('eval_scales_replace', None, 'The new eval scales')

        return cls._instance


    def read_config(self, config_file):

        FlagSetting.config_file = config_file
        FlagSetting.config_name = os.path.split(FlagSetting.config_file)[1][:-4]

        cf = ConfigParser(
            allow_no_value=True,
            inline_comment_prefixes=('#', ';',))
        cf.read(config_file)

        FLAGS.dataset = cf.get('data', 'dataset')
        FLAGS.dataset_dir = cf.get('data', 'dataset_dir')
        FLAGS.model_variant = cf.get('model', 'model_variant')
        if cf.has_option('model', 'decoder_output_stride'):
            FLAGS.decoder_output_stride = [cf.getint('model', 'decoder_output_stride')]
        else:
            FLAGS.decoder_output_stride = None
        FLAGS.add_image_level_feature = cf.getboolean('model', 'add_image_level_feature')
        FLAGS.aspp_with_batch_norm = cf.getboolean('model', 'aspp_with_batch_norm')

        if cf.has_option('model', 'dpn_variant'):
            FLAGS.dpn_variant = cf.get('model', 'dpn_variant')
            FLAGS.dpn_variant = FLAGS.dpn_variant.strip()

        if cf.has_option('model', 'multi_grid'):
            FLAGS.multi_grid = [
                int(d) for d in
                cf.get('model', 'multi_grid').split(',')]

        if cf.has_option('lateral_inhibition', 'li_zones_dpn'):
            FLAGS.li_zones_dpn = [
                int(d) for d in
                cf.get('lateral_inhibition', 'li_zones_dpn').split(',')]
        if cf.has_option('lateral_inhibition', 'li_decay_stds_dpn'):
            FLAGS.li_decay_stds_dpn = [
                float(d) for d in
                cf.get('lateral_inhibition','li_decay_stds_dpn').split(',')]
        if cf.has_option('lateral_inhibition', 'li_rates_dpn'):
            FLAGS.li_rates_dpn = [
                int(d) for d in
                cf.get('lateral_inhibition','li_rates_dpn').split(',')]

        if cf.has_option('lateral_inhibition', 'li_zones_backbone'):
            FLAGS.li_zones_backbone = cf.getint('lateral_inhibition', 'li_zones_backbone')

        if cf.has_option('lateral_inhibition', 'li_decay_stds_backbone'):
            FLAGS.li_decay_stds_backbone = cf.getfloat('lateral_inhibition', 'li_decay_stds_backbone')

        if cf.has_option('lateral_inhibition', 'li_rates_backbone'):
            FLAGS.li_rates_backbone = cf.getint('lateral_inhibition', 'li_rates_backbone')

        if cf.has_option('lateral_inhibition', 'li_location_for_expanded_conv'):
            FLAGS.li_location_for_expanded_conv = cf.get('lateral_inhibition', 'li_location_for_expanded_conv')

        if cf.has_option('lateral_inhibition', 'li_backbone_option_mnv2'):
            FLAGS.li_backbone_option_mnv2 = cf.get('lateral_inhibition', 'li_backbone_option_mnv2')

        if cf.has_option('lateral_inhibition', 'li_activation'):
            FLAGS.li_activation = cf.get('lateral_inhibition', 'li_activation')
        if cf.has_option('lateral_inhibition', 'li_weight_initilizer'):
            FLAGS.li_weight_initilizer = cf.get('lateral_inhibition','li_weight_initilizer')

        if cf.has_option('lateral_inhibition', 'li_weight_initilizer_min_max_backbone') \
                and cf.has_option('lateral_inhibition', 'li_weight_initilizer_min_max_dpn'):
            FLAGS.li_weight_initilizer_min_max_backbone = [
                float(d) for d in
                cf.get('lateral_inhibition','li_weight_initilizer_min_max_backbone').split(',')]
            FLAGS.li_weight_initilizer_min_max_dpn = [
                float(d) for d in
                cf.get('lateral_inhibition','li_weight_initilizer_min_max_dpn').split(',')]
        else:
            FLAGS.li_weight_initilizer_min_max_backbone = [
                float(d) for d in
                cf.get('lateral_inhibition','li_weight_initilizer_min_max').split(',')]
            FLAGS.li_weight_initilizer_min_max_dpn = [
                float(d) for d in
                cf.get('lateral_inhibition','li_weight_initilizer_min_max').split(',')]

        if cf.has_option('lateral_inhibition', 'li_weight_clip_values'):
            FLAGS.li_weight_clip_values = [
                float(d) for d in
                cf.get('lateral_inhibition','li_weight_clip_values').split(',')]
        if cf.has_option('lateral_inhibition', 'li_with_batch_norm'):
            FLAGS.li_with_batch_norm = cf.getboolean('lateral_inhibition', 'li_with_batch_norm')
        if cf.has_option('lateral_inhibition', 'li_with_weight_regularizer'):
            FLAGS.li_with_weight_regularizer = cf.getboolean('lateral_inhibition', 'li_with_weight_regularizer')

        if cf.has_option('lateral_inhibition', 'li_resnet_option'):
            FLAGS.li_resnet_option = cf.get('lateral_inhibition', 'li_resnet_option')

        if cf.has_option('lateral_inhibition', 'li_resnet_bottleneck_pos'):
            FLAGS.li_resnet_bottleneck_pos = cf.get('lateral_inhibition', 'li_resnet_bottleneck_pos')

        if cf.has_option('lateral_inhibition', 'li_resnet_block_pos'):
            FLAGS.li_resnet_block_pos = cf.get('lateral_inhibition', 'li_resnet_block_pos')

        if cf.has_option('model', 'li_aspp_name_scope_compatible'):
            FLAGS.li_aspp_name_scope_compatible = cf.getboolean('model', 'li_aspp_name_scope_compatible')

        if FlagSetting.mode == 'train':
            FLAGS.train_split = cf.get('data', 'train_split')
            FLAGS.train_logdir = os.path.join(
                cf.get('data', 'snapshot_root'), FlagSetting.config_name, 'train')

            if cf.has_option('train', 'save_interval_secs'):
                FLAGS.save_interval_secs = cf.getint('train', 'save_interval_secs')

            if cf.has_option('train', 'save_checkpoint_steps'):
                FLAGS.save_checkpoint_steps = cf.getint('train', 'save_checkpoint_steps')

            if cf.has_option('train', 'save_summaries_secs'):
                FLAGS.save_summaries_secs = cf.getint('train', 'save_summaries_secs')

            if cf.has_option('train', 'save_summaries_steps'):
                FLAGS.save_summaries_steps = cf.getint('train', 'save_summaries_steps')

            if cf.has_option('train', 'max_to_keep'):
                FLAGS.max_to_keep = cf.getint('train', 'max_to_keep')

            if cf.has_option('train', 'freeze_backbone_network'):
                FLAGS.freeze_backbone_network = cf.getboolean('train', 'freeze_backbone_network')

            if cf.has_option('train', 'last_layers_contain_logits_only'):
                FLAGS.last_layers_contain_logits_only = cf.getboolean('train', 'last_layers_contain_logits_only')

            FLAGS.base_learning_rate = cf.getfloat('train', 'base_learning_rate')
            FLAGS.learning_rate_decay_factor = cf.getfloat('train', 'learning_rate_decay_factor')
            FLAGS.learning_rate_decay_step = cf.getint('train', 'learning_rate_decay_step')
            FLAGS.learning_power = cf.getfloat('train', 'learning_power')
            FLAGS.training_number_of_steps = cf.getint('train', 'training_number_of_steps')
            FLAGS.momentum = cf.getfloat('train', 'momentum')
            FLAGS.train_batch_size = cf.getint('train', 'train_batch_size')
            FLAGS.weight_decay = cf.getfloat('train', 'weight_decay')
            FLAGS.train_crop_size = [d for d in cf.get('train', 'train_crop_size').split(',')]
            if cf.has_option('train', 'tf_initial_checkpoint'):
                FLAGS.tf_initial_checkpoint = cf.get('train', 'tf_initial_checkpoint')
            FLAGS.initialize_last_layer = cf.getboolean('train', 'initialize_last_layer')
            FLAGS.last_layer_gradient_multiplier = cf.getfloat('train', 'last_layer_gradient_multiplier')
            FLAGS.fine_tune_batch_norm = cf.getboolean('train', 'fine_tune_batch_norm')
            FLAGS.num_clones = cf.getint('train', 'num_clones')

            FLAGS.min_scale_factor = cf.getfloat('train', 'min_scale_factor')
            FLAGS.max_scale_factor = cf.getfloat('train', 'max_scale_factor')
            FLAGS.scale_factor_step_size = cf.getfloat('train', 'scale_factor_step_size')
            FLAGS.output_stride = cf.getint('train','output_stride')
            if cf.has_option('train', 'atrous_rates'):
                FLAGS.atrous_rates = [int(d) for d in cf.get('train', 'atrous_rates').split(',')]
            if cf.has_option('train', 'min_resize_value'):
                FLAGS.min_resize_value = cf.getint('train', 'min_resize_value')
            if cf.has_option('train', 'max_resize_value'):
                FLAGS.max_resize_value = cf.getint('train', 'max_resize_value')
            if cf.has_option('train', 'resize_factor'):
                FLAGS.resize_factor = cf.getint('train', 'resize_factor')
            if cf.has_option('train', 'finetune_li_weight_only'):
                FLAGS.finetune_li_weight_only = cf.getboolean('train', 'finetune_li_weight_only')
            if cf.has_option('train', 'li_weight_gradient_multiplier'):
                FLAGS.li_weight_gradient_multiplier = cf.getfloat('train', 'li_weight_gradient_multiplier')
            if cf.has_option('train', 'optimizer'):
                FLAGS.optimizer = cf.get('train', 'optimizer')
            if cf.has_option('train', 'adam_learning_rate'):
                FLAGS.adam_learning_rate = cf.getfloat('train', 'adam_learning_rate')
            if cf.has_option('train', 'adam_epsilon'):
                FLAGS.adam_epsilon = cf.getfloat('train', 'adam_epsilon')

        elif FlagSetting.mode == 'val':
            FLAGS.eval_crop_size = [d for d in cf.get('val', 'eval_crop_size').split(',')]
            FLAGS.eval_scales = [float(d) for d in cf.get('val', 'eval_scales').split(',')]
            FLAGS.val_split = cf.get('data', 'val_split')
            FLAGS.checkpoint_dir = os.path.join(
                cf.get('data', 'snapshot_root'), FlagSetting.config_name, 'train')
            FLAGS.val_logdir = os.path.join(
                cf.get('data', 'snapshot_root'), FlagSetting.config_name, 'val')
            FLAGS.output_stride = cf.getint('val','output_stride')
            if cf.has_option('val', 'atrous_rates'):
                FLAGS.atrous_rates = [int(d) for d in cf.get('val', 'atrous_rates').split(',')]
            if cf.has_option('val', 'min_resize_value'):
                FLAGS.min_resize_value = cf.getint('val', 'min_resize_value')
            if cf.has_option('val', 'max_resize_value'):
                FLAGS.max_resize_value = cf.getint('val', 'max_resize_value')
            if cf.has_option('val', 'resize_factor'):
                FLAGS.resize_factor = cf.getint('val', 'resize_factor')

        elif FlagSetting.mode == 'test':
            FLAGS.eval_crop_size = [d for d in cf.get('test', 'eval_crop_size').split(',')]
            FLAGS.eval_scales = [float(d) for d in cf.get('test', 'eval_scales').split(',')]
            FLAGS.checkpoint_dir = os.path.join(
                cf.get('data', 'snapshot_root'), FlagSetting.config_name, 'train')
            FLAGS.val_logdir = os.path.join(
                cf.get('data', 'snapshot_root'), FlagSetting.config_name, 'val')
            FLAGS.test_logdir = os.path.join(
                cf.get('data', 'snapshot_root'), FlagSetting.config_name, 'test')
            FLAGS.test_split = cf.get('data', 'test_split')
            FLAGS.output_stride = cf.getint('test','output_stride')
            if cf.has_option('test', 'atrous_rates'):
                FLAGS.atrous_rates = [int(d) for d in cf.get('test', 'atrous_rates').split(',')]
            if cf.has_option('test', 'num_plots_to_save'):
                FLAGS.num_plots_to_save = cf.getint('test', 'num_plots_to_save')
            if cf.has_option('test', 'min_resize_value'):
                FLAGS.min_resize_value = cf.getint('test', 'min_resize_value')
            if cf.has_option('test', 'max_resize_value'):
                FLAGS.max_resize_value = cf.getint('test', 'max_resize_value')
            if cf.has_option('test', 'resize_factor'):
                FLAGS.resize_factor = cf.getint('test', 'resize_factor')
            if cf.has_option('test', 'add_flipped_images'):
                FLAGS.add_flipped_images = cf.getboolean('test', 'add_flipped_images')

    def print_flags(self, print_all=False):
        print('Current config file: {}'.format(FlagSetting.config_file))
        if not print_all:
            print('Important flag values: ')
        for key,val in sorted(FLAGS.flag_values_dict().items()):
            if key in _IMPORTANT_FLAGS_TO_PRINT or print_all:
                print('  {}: {}'.format(key,val))
            elif FLAGS.dpn_variant.startswith('li') or FLAGS.model_variant.startswith('li'):
                if key.startswith('li'):
                    print('  {}: {}'.format(key,val))


    def check_options(self):
        assert FLAGS.dpn_variant in _SUPPORT_DPN_VARIANT, \
            'Unsupported dpn variant type: {}\n Supporting dpn types: {}'.format(
                FLAGS.dpn_variant, _SUPPORT_DPN_VARIANT)

    # set the correct checkpoint point to test based on different flag values
    def set_checkpoint_path_to_test(self):

        eval_scale_str = 'eval_scale'
        for d in FLAGS.eval_scales:
            eval_scale_str += '_'+str(d)

        # if FLAGS.checkpoint_name is specified, build checkpoint path based on it
        if FLAGS.checkpoint_name is not None:
            FLAGS.checkpoint_path = os.path.join(
                FLAGS.checkpoint_dir, FLAGS.checkpoint_name)
            FLAGS.test_resdir = os.path.join(
                FLAGS.test_logdir, FLAGS.checkpoint_name, eval_scale_str)

        # if FLAGS.checkpoint_path is specified, test this checkpoint
        if FLAGS.checkpoint_path is not None:
            FLAGS.checkpoint_name = os.path.split(FLAGS.checkpoint_path)[1]
            FLAGS.test_resdir = os.path.join(
                FLAGS.test_logdir, FLAGS.checkpoint_name, eval_scale_str)

        # if both not specified, use validation result to find best-performing checkpoint path
        if FLAGS.checkpoint_name is None and FLAGS.checkpoint_path is None:
            best_ck_path, _ = load_sorted_ck_paths_from_val_result(
                os.path.join(FLAGS.val_logdir, 'val_result.mat'))
            FLAGS.checkpoint_path, FLAGS.checkpoint_name = (best_ck_path, os.path.split(best_ck_path)[1])
            FLAGS.test_resdir = os.path.join(
                FLAGS.test_logdir,
                FLAGS.checkpoint_name + '(best_val)',
                eval_scale_str)

