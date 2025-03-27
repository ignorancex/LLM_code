"""
Implementation of the LI-MNV2 and LI-ResNet
"""

import tensorflow.compat.v1 as tf
import functools

from nets.mobilenet import conv_blocks as ops
from nets.mobilenet.mobilenet import op

expand_input = ops.expand_input_by_factor

from li_utils import conv_ops
from li_models import model_utils

# the expanded conv block with lateral inhibition
li_expanded_conv = conv_ops.expanded_conv_with_lateral_inhibition

from deeplab.core import resnet_v1_beta
from tensorflow.contrib.slim.nets import resnet_utils

# the bottleneck with lateral inhibitions for ResNet
li_resent_bottleneck = conv_ops.li_resnet_bottleneck

from tensorflow.contrib import slim

flags = tf.app.flags
FLAGS = flags.FLAGS


def get_li_mnv2_def():
    """
    Get a MobileNetV2 definition with lateral inhibition layers
    """

    li_weight_decay_mnv2 = 0.0001
    li_batch_norm_params = {
        'decay': 0.997,
        'center': True,
        'scale': True}

    li_params = model_utils.decode_flag_li_params_for_backbone(
        weight_decay=li_weight_decay_mnv2,
        normalizer_params=li_batch_norm_params)

    LI_MNV2_DEF = dict()

    LI_MNV2_DEF['defaults'] = {
        # Note: these parameters of batch norm affect the architecture
        # that's why they are here and not in training_scope.
        (slim.batch_norm,): {'center': True, 'scale': True},
        (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
            'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu6
        },
        (li_expanded_conv,): {
            'expansion_size': expand_input(6),
            'split_expansion': 1,
            'normalizer_fn': slim.batch_norm,
            'residual': True,
            'li_location_for_expanded_conv': FLAGS.li_location_for_expanded_conv,
            'li_params': li_params
        },
        (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'}
    }

    if FLAGS.li_backbone_option_mnv2 == 'a':
        specs = [
            op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
            op(li_expanded_conv,
               expansion_size=expand_input(1, divisible_by=1),
               num_outputs=16, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=24, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=24),
            op(li_expanded_conv, stride=2, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=32),
            op(li_expanded_conv, stride=1, num_outputs=32),
            op(li_expanded_conv, stride=2, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=160, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=160),
            op(li_expanded_conv, stride=1, num_outputs=160, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=320, li_params=None),
            op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
        ]
    elif FLAGS.li_backbone_option_mnv2 == 'b':
        specs = [
            op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
            op(li_expanded_conv,
               expansion_size=expand_input(1, divisible_by=1),
               num_outputs=16, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=24, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=24),
            op(li_expanded_conv, stride=2, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=32),
            op(li_expanded_conv, stride=2, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96),
            op(li_expanded_conv, stride=2, num_outputs=160, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=160, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=160),
            op(li_expanded_conv, stride=1, num_outputs=320, li_params=None),
            op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
        ]

    elif FLAGS.li_backbone_option_mnv2 == 'c':
        specs = [
            op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
            op(li_expanded_conv,
               expansion_size=expand_input(1, divisible_by=1),
               num_outputs=16, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=24, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=24, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=160, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=160, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=160, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=320),
            op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
        ]

    elif FLAGS.li_backbone_option_mnv2 == 'd':
        specs = [
            op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
            op(li_expanded_conv,
               expansion_size=expand_input(1, divisible_by=1),
               num_outputs=16, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=24, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=24, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=160, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=160, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=160),
            op(li_expanded_conv, stride=1, num_outputs=320, li_params=None),
            op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
        ]

    elif FLAGS.li_backbone_option_mnv2 == 'e':
        specs = [
            op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
            op(li_expanded_conv,
               expansion_size=expand_input(1, divisible_by=1),
               num_outputs=16, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=24, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=24, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96),
            op(li_expanded_conv, stride=2, num_outputs=160, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=160, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=160),
            op(li_expanded_conv, stride=1, num_outputs=320, li_params=None),
            op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
        ]

    elif FLAGS.li_backbone_option_mnv2 == 'f':
        specs = [
            op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
            op(li_expanded_conv,
               expansion_size=expand_input(1, divisible_by=1),
               num_outputs=16, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=24, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=24, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96),
            op(li_expanded_conv, stride=2, num_outputs=160, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=160, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=160),
            op(li_expanded_conv, stride=1, num_outputs=320, li_params=None),
            op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
        ]

    elif FLAGS.li_backbone_option_mnv2 == 'g':
        specs = [
            op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
            op(li_expanded_conv,
               expansion_size=expand_input(1, divisible_by=1),
               num_outputs=16),
            op(li_expanded_conv, stride=2, num_outputs=24),
            op(li_expanded_conv, stride=1, num_outputs=24),
            op(li_expanded_conv, stride=2, num_outputs=32),
            op(li_expanded_conv, stride=1, num_outputs=32),
            op(li_expanded_conv, stride=1, num_outputs=32),
            op(li_expanded_conv, stride=2, num_outputs=64),
            op(li_expanded_conv, stride=1, num_outputs=64),
            op(li_expanded_conv, stride=1, num_outputs=64),
            op(li_expanded_conv, stride=1, num_outputs=64),
            op(li_expanded_conv, stride=1, num_outputs=96),
            op(li_expanded_conv, stride=1, num_outputs=96),
            op(li_expanded_conv, stride=1, num_outputs=96),
            op(li_expanded_conv, stride=2, num_outputs=160),
            op(li_expanded_conv, stride=1, num_outputs=160),
            op(li_expanded_conv, stride=1, num_outputs=160),
            op(li_expanded_conv, stride=1, num_outputs=320),
            op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
        ]

    elif FLAGS.li_backbone_option_mnv2 == 'h':
        specs = [
            op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
            op(li_expanded_conv,
               expansion_size=expand_input(1, divisible_by=1),
               num_outputs=16),
            op(li_expanded_conv, stride=2, num_outputs=24),
            op(li_expanded_conv, stride=1, num_outputs=24),
            op(li_expanded_conv, stride=2, num_outputs=32),
            op(li_expanded_conv, stride=1, num_outputs=32),
            op(li_expanded_conv, stride=1, num_outputs=32),
            op(li_expanded_conv, stride=2, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64,li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64,li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64,li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96,li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=160, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=160,li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=160,li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=320,li_params=None),
            op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
        ]

    elif FLAGS.li_backbone_option_mnv2 == 'j':
        specs = [
            op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
            op(li_expanded_conv,
               expansion_size=expand_input(1, divisible_by=1),
               num_outputs=16, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=24, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=24, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96),
            op(li_expanded_conv, stride=2, num_outputs=160, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=160, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=160),
            op(li_expanded_conv, stride=1, num_outputs=320),
            op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
        ]

    elif FLAGS.li_backbone_option_mnv2 == 'k':
        specs = [
            op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
            op(li_expanded_conv,
               expansion_size=expand_input(1, divisible_by=1),
               num_outputs=16, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=24, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=24, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=160),
            op(li_expanded_conv, stride=1, num_outputs=160),
            op(li_expanded_conv, stride=1, num_outputs=160),
            op(li_expanded_conv, stride=1, num_outputs=320, li_params=None),
            op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
        ]

    elif FLAGS.li_backbone_option_mnv2 == 'l':
        specs = [
            op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
            op(li_expanded_conv,
               expansion_size=expand_input(1, divisible_by=1),
               num_outputs=16, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=24, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=24, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=160),
            op(li_expanded_conv, stride=1, num_outputs=160),
            op(li_expanded_conv, stride=1, num_outputs=160),
            op(li_expanded_conv, stride=1, num_outputs=320),
            op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
        ]

    elif FLAGS.li_backbone_option_mnv2 == 'm':
        specs = [
            op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
            op(li_expanded_conv,
               expansion_size=expand_input(1, divisible_by=1),
               num_outputs=16, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=24, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=24, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=32, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=64, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=96, li_params=None),
            op(li_expanded_conv, stride=2, num_outputs=160, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=160, li_params=None),
            op(li_expanded_conv, stride=1, num_outputs=160),
            op(li_expanded_conv, stride=1, num_outputs=320),
            op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
        ]

    else:
        raise ValueError('Invalid li_backbone_option_mnv2 options!')

    LI_MNV2_DEF['spec'] = specs

    return LI_MNV2_DEF


def li_resnet_v1_beta_block(scope, base_depth, num_units, stride,
                            li_position_bottleneck='after_conv1',
                            li_position_block=None,
                            li_params=None):
    """Helper function for creating a resnet_v1 beta variant bottleneck block with lateral inhibition layers

    Args:
        scope: The scope of the block.
        base_depth: The depth of the bottleneck layer for each unit.
        num_units: The number of units in the block.
        stride: The stride of the block, implemented as a stride in the last unit.
        All other units have stride=1.
        li_position_bottleneck: where to add LI layer in a bottleneck unit
        li_position_block: options regarding which bottleneck unit in the block will have LI layer
        li_params: Parameters for LI layers

    Returns:
        A resnet_v1 bottleneck block.
    """

    if li_params is None or li_position_block is None:
        # do not use LI-Convs
        block_args = [{
            'depth': base_depth * 4,
            'depth_bottleneck': base_depth,
            'stride': 1,
            'unit_rate': 1
        }] * (num_units - 1) + [{
            'depth': base_depth * 4,
            'depth_bottleneck': base_depth,
            'stride': stride,
            'unit_rate': 1}]

    elif li_position_block == 'last':
        # use LI-Convs in the top bottleneck
        block_args = [{
            'depth': base_depth * 4,
            'depth_bottleneck': base_depth,
            'stride': 1,
            'unit_rate': 1
        }] * (num_units - 1) + [{
            'depth': base_depth * 4,
            'depth_bottleneck': base_depth,
            'stride': stride,
            'unit_rate': 1,
            'li_position': li_position_bottleneck,
		    'li_params':li_params
        }]

    elif li_position_block == 'last_two':
        # use LI-Convs in the top two bottlenecks
        block_args = [{
            'depth': base_depth * 4,
            'depth_bottleneck': base_depth,
            'stride': 1,
            'unit_rate': 1
        }] * (num_units - 2)

        li_bottleneck_1 = [{'depth': base_depth * 4,
            'depth_bottleneck': base_depth,
            'stride': 1,
            'unit_rate': 1,
            'li_position': li_position_bottleneck,
            'li_params': li_params}]

        li_bottleneck_2 = [{
            'depth': base_depth * 4,
            'depth_bottleneck': base_depth,
            'stride': stride,
            'unit_rate': 1,
            'li_position': li_position_bottleneck,
            'li_params': li_params
        }]

        block_args = block_args + li_bottleneck_1 + li_bottleneck_2

    else:
        raise ValueError('Invalid li_position_block value: {}'.format(li_position_block))

    return resnet_utils.Block(scope, li_resent_bottleneck, block_args)


def li_resnet_v1_50_beta(
        inputs,
        num_classes=None,
        is_training=None,
        global_pool=False,
        output_stride=None,
        multi_grid=None,
        reuse=None,
        scope='resnet_v1_50'):
    """Resnet v1 50 beta variant with Lateral Inhibitions

    This variant modifies the first convolution layer of ResNet-v1-50. In
    particular, it changes the original one 7x7 convolution to three 3x3
    convolutions, and LI layers are added in certain blocks.

    The main scope is still set to 'resnet_v1_50' for compatibilities.

    Args:
        inputs: A tensor of size [batch, height_in, width_in, channels].
        num_classes: Number of predicted classes for classification tasks. If None
            we return the features before the logit layer.
        is_training: Enable/disable is_training for batch normalization.
        global_pool: If True, we perform global average pooling before computing the
            logits. Set to True for image classification, False for dense prediction.
        output_stride: If None, then the output will be computed at the nominal
        network stride. If output_stride is not None, it specifies the requested
            ratio of input to output spatial resolution.
        multi_grid: Employ a hierarchy of different atrous rates within network.
        reuse: whether or not the network and its variables should be reused. To be
            able to reuse 'scope' must be given.
        scope: Optional variable_scope.
        li_option: Options for determine how to add LI layers
        li_position_bottleneck: where to add LI layer in a bottleneck unit
        li_position_block: options regarding which bottleneck units in the block will have LI layer
        li_params: Parameters for LI layers

    Returns:
        net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
            If global_pool is False, then height_out and width_out are reduced by a
            factor of output_stride compared to the respective height_in and width_in,
            else both height_out and width_out equal one. If num_classes is None, then
            net is the output of the last ResNet block, potentially after global
            average pooling. If num_classes is not None, net contains the pre-softmax
            activations.
        end_points: A dictionary from components of the network to the corresponding
        activation.

    Raises:
        ValueError: if multi_grid is not None and does not have length = 3.
    """
    if multi_grid is None:
        multi_grid = resnet_v1_beta._DEFAULT_MULTI_GRID
    else:
        if len(multi_grid) != 3:
            raise ValueError('Expect multi_grid to have length 3.')

    li_option = FLAGS.li_resnet_option
    li_position_bottleneck = FLAGS.li_resnet_bottleneck_pos
    li_position_block = FLAGS.li_resnet_block_pos

    li_weight_decay_mnv2 = 0.0001
    li_batch_norm_params = {
        'decay': 0.997,
        'center': True,
        'scale': True}
    li_params = model_utils.decode_flag_li_params_for_backbone(
        weight_decay=li_weight_decay_mnv2,
        normalizer_params=li_batch_norm_params)

    # using LI-conv for all bottlenecks in block 4
    block4_args_all = [
        {'depth': 2048,
         'depth_bottleneck': 512,
         'stride': 1,
         'unit_rate': rate,
         'li_position': li_position_bottleneck,
         'li_params': li_params} for rate in multi_grid]

    # using LI-conv for the last bottleneck in block 4
    block4_args_last = [
        {'depth': 2048,
         'depth_bottleneck': 512,
         'stride': 1,
         'unit_rate': rate,
         'li_position': li_position_bottleneck,
         'li_params': None} for rate in multi_grid[0:2]]
    block4_args_last.append(
        {'depth': 2048,
         'depth_bottleneck': 512,
         'stride': 1,
         'unit_rate': multi_grid[2],
         'li_position': li_position_bottleneck,
         'li_params': li_params})

    if li_option == 'a': # use LI-Convs for block 3 (certain bottlenecks) and block 4 (all bottlenecks)
        blocks = [
            li_resnet_v1_beta_block(
                'block1', base_depth=64, num_units=3, stride=2),
            li_resnet_v1_beta_block(
                'block2', base_depth=128, num_units=4, stride=2),
            li_resnet_v1_beta_block(
                'block3', base_depth=256, num_units=6, stride=2,
                li_position_bottleneck=li_position_bottleneck,
                li_position_block=li_position_block,
                li_params=li_params),
            resnet_utils.Block('block4', li_resent_bottleneck, block4_args_all),
        ]

    elif li_option == 'b':  # use LI-Convs for block 4 (all bottlenecks)
        blocks = [
            li_resnet_v1_beta_block(
                'block1', base_depth=64, num_units=3, stride=2),
            li_resnet_v1_beta_block(
                'block2', base_depth=128, num_units=4, stride=2),
            li_resnet_v1_beta_block(
                'block3', base_depth=256, num_units=6, stride=2),
            resnet_utils.Block('block4', li_resent_bottleneck, block4_args_all),
        ]

    elif li_option == 'c': # use LI-Convs for block 3 (certain bottlenecks) and block 4 (top bottleneck)
        blocks = [
            li_resnet_v1_beta_block(
                'block1', base_depth=64, num_units=3, stride=2),
            li_resnet_v1_beta_block(
                'block2', base_depth=128, num_units=4, stride=2),
            li_resnet_v1_beta_block(
                'block3', base_depth=256, num_units=6, stride=2,
                li_position_bottleneck=li_position_bottleneck,
                li_position_block=li_position_block,
                li_params=li_params),
            resnet_utils.Block('block4', li_resent_bottleneck, block4_args_last),
        ]

    elif li_option == 'd': # use LI-Convs for block 4 (top bottleneck)
        blocks = [
            li_resnet_v1_beta_block(
                'block1', base_depth=64, num_units=3, stride=2),
            li_resnet_v1_beta_block(
                'block2', base_depth=128, num_units=4, stride=2),
            li_resnet_v1_beta_block(
                'block3', base_depth=256, num_units=6, stride=2),
            resnet_utils.Block('block4', li_resent_bottleneck, block4_args_last),
        ]

    elif li_option == 'e': # use LI-Convs for block 2,3 (certain bottlenecks) and block 4 (top bottleneck)
        blocks = [
            li_resnet_v1_beta_block(
                'block1', base_depth=64, num_units=3, stride=2),
            li_resnet_v1_beta_block(
                'block2', base_depth=128, num_units=4, stride=2,
                li_position_bottleneck=li_position_bottleneck,
                li_position_block=li_position_block,
                li_params=li_params),
            li_resnet_v1_beta_block(
                'block3', base_depth=256, num_units=6, stride=2,
                li_position_bottleneck=li_position_bottleneck,
                li_position_block=li_position_block,
                li_params=li_params),
            resnet_utils.Block('block4', li_resent_bottleneck, block4_args_last),
        ]

    elif li_option == 'f': # use LI-Convs for block 3 (certain bottlenecks)
        blocks = [
            li_resnet_v1_beta_block(
                'block1', base_depth=64, num_units=3, stride=2),
            li_resnet_v1_beta_block(
                'block2', base_depth=128, num_units=4, stride=2),
            li_resnet_v1_beta_block(
                'block3', base_depth=256, num_units=6, stride=2,
                li_position_bottleneck=li_position_bottleneck,
                li_position_block=li_position_block,
                li_params=li_params),
            resnet_utils.Block('block4', li_resent_bottleneck,
                               [{'depth': 2048,
                                 'depth_bottleneck': 512,
                                 'stride': 1,
                                 'unit_rate': rate} for rate in multi_grid]),
        ]

    else:
        raise ValueError('Invalid li_option: {}'.format(li_option))

    return resnet_v1_beta.resnet_v1_beta(
        inputs,
        blocks=blocks,
        num_classes=num_classes,
        is_training=is_training,
        global_pool=global_pool,
        output_stride=output_stride,
        root_block_fn=functools.partial(resnet_v1_beta.root_block_fn_for_beta_variant),
        reuse=reuse,
        scope=scope)


def li_resnet_v1_101_beta(
        inputs,
        num_classes=None,
        is_training=None,
        global_pool=False,
        output_stride=None,
        multi_grid=None,
        reuse=None,
        scope='resnet_v1_101'):

    """Resnet v1 101 beta variant with Lateral Inhibitions

    This variant modifies the first convolution layer of ResNet-v1-101. In
    particular, it changes the original one 7x7 convolution to three 3x3
    convolutions. LI layers are added in certain blocks.

    The main scope is still set to 'resnet_v1_101' for compatibilities

    Args:
        inputs: A tensor of size [batch, height_in, width_in, channels].
        num_classes: Number of predicted classes for classification tasks. If None
            we return the features before the logit layer.
        is_training: Enable/disable is_training for batch normalization.
        global_pool: If True, we perform global average pooling before computing the
            logits. Set to True for image classification, False for dense prediction.
        output_stride: If None, then the output will be computed at the nominal
            network stride. If output_stride is not None, it specifies the requested
            ratio of input to output spatial resolution.
        multi_grid: Employ a hierarchy of different atrous rates within network.
        reuse: whether or not the network and its variables should be reused. To be
            able to reuse 'scope' must be given.
        scope: Optional variable_scope.
        li_option: Options for determine how to add LI layers
        li_position_bottleneck: where to add LI layer in a bottleneck unit
        li_position_block: options regarding which bottleneck units in the block will have LI layer
        li_params: Parameters for LI layers

    Returns:
        net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
            If global_pool is False, then height_out and width_out are reduced by a
            factor of output_stride compared to the respective height_in and width_in,
            else both height_out and width_out equal one. If num_classes is None, then
            net is the output of the last ResNet block, potentially after global
            average pooling. If num_classes is not None, net contains the pre-softmax
            activations.
        end_points: A dictionary from components of the network to the corresponding
            activation.

    Raises:
      ValueError: if multi_grid is not None and does not have length = 3.
    """
    if multi_grid is None:
        multi_grid = resnet_v1_beta._DEFAULT_MULTI_GRID
    else:
        if len(multi_grid) != 3:
            raise ValueError('Expect multi_grid to have length 3.')

    li_option = FLAGS.li_resnet_option
    li_position_bottleneck = FLAGS.li_resnet_bottleneck_pos
    li_position_block = FLAGS.li_resnet_block_pos

    li_weight_decay_mnv2 = 0.0001
    li_batch_norm_params = {
        'decay': 0.997,
        'center': True,
        'scale': True}
    li_params = model_utils.decode_flag_li_params_for_backbone(
        weight_decay=li_weight_decay_mnv2,
        normalizer_params=li_batch_norm_params)

    # using LI-conv for all bottlenecks in block 4
    block4_args_all = [
        {'depth': 2048,
         'depth_bottleneck': 512,
         'stride': 1,
         'unit_rate': rate,
         'li_position': li_position_bottleneck,
         'li_params': li_params} for rate in multi_grid]

    # using LI-conv for the last bottleneck in block 4
    block4_args_last = [
       {'depth': 2048,
        'depth_bottleneck': 512,
        'stride': 1,
        'unit_rate': rate,
        'li_position': li_position_bottleneck,
        'li_params': None} for rate in multi_grid[0:2]]
    block4_args_last.append(
       {'depth': 2048,
        'depth_bottleneck': 512,
        'stride': 1,
        'unit_rate': multi_grid[2],
        'li_position': li_position_bottleneck,
        'li_params': li_params})

    if li_option == 'a': # use LI-Convs for block 3 (certain bottlenecks) and block 4 (all bottlenecks)
        blocks = [
        li_resnet_v1_beta_block(
            'block1', base_depth=64, num_units=3, stride=2),
        li_resnet_v1_beta_block(
            'block2', base_depth=128, num_units=4, stride=2),
        li_resnet_v1_beta_block(
            'block3', base_depth=256, num_units=23, stride=2,
            li_position_bottleneck=li_position_bottleneck,
            li_position_block=li_position_block,
            li_params=li_params),
        resnet_utils.Block('block4', li_resent_bottleneck, block4_args_all),
        ]

    elif li_option == 'b':  # use LI-Convs for block 4 (all bottlenecks)
        blocks = [
        li_resnet_v1_beta_block(
            'block1', base_depth=64, num_units=3, stride=2),
        li_resnet_v1_beta_block(
            'block2', base_depth=128, num_units=4, stride=2),
        li_resnet_v1_beta_block(
            'block3', base_depth=256, num_units=23, stride=2),
        resnet_utils.Block('block4', li_resent_bottleneck, block4_args_all),
        ]

    elif li_option == 'c':  # use LI-Convs for block 3 (certain bottlenecks) and block 4 (top bottleneck)
        blocks = [
        li_resnet_v1_beta_block(
            'block1', base_depth=64, num_units=3, stride=2),
        li_resnet_v1_beta_block(
            'block2', base_depth=128, num_units=4, stride=2),
        li_resnet_v1_beta_block(
            'block3', base_depth=256, num_units=23, stride=2,
            li_position_bottleneck=li_position_bottleneck,
            li_position_block=li_position_block,
            li_params=li_params),
        resnet_utils.Block('block4', li_resent_bottleneck, block4_args_last),
        ]

    elif li_option == 'd':  # use LI-Convs for block 4 (top bottleneck)
        blocks = [
        li_resnet_v1_beta_block(
            'block1', base_depth=64, num_units=3, stride=2),
        li_resnet_v1_beta_block(
            'block2', base_depth=128, num_units=4, stride=2),
        li_resnet_v1_beta_block(
            'block3', base_depth=256, num_units=23, stride=2),
        resnet_utils.Block('block4', li_resent_bottleneck, block4_args_last),
        ]

    else:
        raise ValueError('Invalid li_option: {}'.format(li_option))


    return resnet_v1_beta.resnet_v1_beta(
        inputs,
        blocks=blocks,
        num_classes=num_classes,
        is_training=is_training,
        global_pool=global_pool,
        output_stride=output_stride,
        root_block_fn=functools.partial(resnet_v1_beta.root_block_fn_for_beta_variant),
        reuse=reuse,
        scope=scope)
