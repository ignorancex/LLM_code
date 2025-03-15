"""
Training script
Example usage:
    python train.py --config_file ../config/ADE20K/liMnv2_liAspp_example.ini
"""

import sys
sys.path.insert(0,'../')
sys.path.insert(0, '../libs/deeplab_v3plus')
sys.path.insert(0, '../libs/deeplab_v3plus/slim')
import six

import tensorflow.compat.v1 as tf
from tensorflow import contrib as tf_contrib
from tensorflow.python.ops import math_ops
from deeplab import common
from deeplab.datasets import data_generator_extend
from deeplab.utils import train_utils
from li_models.deeplab_feature_extractor import name_scope as model_variant_to_name_scope
from li_models import deeplab_model as model_extend
from li_models.model_utils import get_li_weight_gradient_multipliers

flags = tf.app.flags
FLAGS = flags.FLAGS

from li_utils.flag_setting import FlagSetting
flag_setter = FlagSetting(mode='train')



def _build_deeplab(iterator, outputs_to_num_classes, ignore_label):
    """Builds a clone of DeepLab.

	Args:
	  iterator: An iterator of type tf.data.Iterator for images and labels.
	  outputs_to_num_classes: A map from output type to the number of classes. For
		example, for the task of semantic segmentation with 21 semantic classes,
		we would have outputs_to_num_classes['semantic'] = 21.
	  ignore_label: Ignore label.
	"""
    samples = iterator.get_next()

    # Add name to input and label nodes so we can add to summary.
    samples[common.IMAGE] = tf.identity(samples[common.IMAGE], name=common.IMAGE)
    samples[common.LABEL] = tf.identity(samples[common.LABEL], name=common.LABEL)

    model_options = common.ModelOptions(
        outputs_to_num_classes=outputs_to_num_classes,
        crop_size=[int(sz) for sz in FLAGS.train_crop_size],
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    outputs_to_scales_to_logits = model_extend.multi_scale_logits(
        samples[common.IMAGE],
        model_options=model_options,
        image_pyramid=FLAGS.image_pyramid,
        weight_decay=FLAGS.weight_decay,
        is_training=True,
        fine_tune_batch_norm=FLAGS.fine_tune_batch_norm,
        nas_training_hyper_parameters={
            'drop_path_keep_prob': FLAGS.drop_path_keep_prob,
            'total_training_steps': FLAGS.training_number_of_steps,
        })

    # Add name to graph node so we can add to summary.
    output_type_dict = outputs_to_scales_to_logits[common.OUTPUT_TYPE]
    output_type_dict[model_extend.MERGED_LOGITS_SCOPE] = tf.identity(
        output_type_dict[model_extend.MERGED_LOGITS_SCOPE], name=common.OUTPUT_TYPE)

    for output, num_classes in six.iteritems(outputs_to_num_classes):
        train_utils.add_softmax_cross_entropy_loss_for_each_scale(
            outputs_to_scales_to_logits[output],
            samples[common.LABEL],
            num_classes,
            ignore_label,
            loss_weight=1.0,
            upsample_logits=FLAGS.upsample_logits,
            hard_example_mining_step=FLAGS.hard_example_mining_step,
            top_k_percent_pixels=FLAGS.top_k_percent_pixels,
            scope=output)

        # Log the summary
        _log_summaries(samples[common.IMAGE], samples[common.LABEL], num_classes,
                       output_type_dict[model_extend.MERGED_LOGITS_SCOPE])


def _tower_loss(iterator, num_of_classes, ignore_label, scope, reuse_variable):
    """Calculates the total loss on a single tower running the deeplab model.

	Args:
	  iterator: An iterator of type tf.data.Iterator for images and labels.
	  num_of_classes: Number of classes for the dataset.
	  ignore_label: Ignore label for the dataset.
	  scope: Unique prefix string identifying the deeplab tower.
	  reuse_variable: If the variable should be reused.

	Returns:
	   The total loss for a batch of data.
	"""
    with tf.variable_scope(
            tf.get_variable_scope(), reuse=True if reuse_variable else None):
        _build_deeplab(iterator, {common.OUTPUT_TYPE: num_of_classes}, ignore_label)

    losses = tf.losses.get_losses(scope=scope)
    for loss in losses:
        tf.summary.scalar('Losses/%s' % loss.op.name, loss)

    regularization_loss = tf.losses.get_regularization_loss(scope=scope)
    tf.summary.scalar('Losses/%s' % regularization_loss.op.name,
                      regularization_loss)

    total_loss = tf.add_n([tf.add_n(losses), regularization_loss])
    return total_loss


def _average_gradients(tower_grads):
    """Calculates average of gradient for each shared variable across all towers.

	Note that this function provides a synchronization point across all towers.

	Args:
	  tower_grads: List of lists of (gradient, variable) tuples. The outer list is
		over individual gradients. The inner list is over the gradient calculation
		for each tower.

	Returns:
	   List of pairs of (gradient, variable) where the gradient has been summed
		 across all towers.
	"""
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads, variables = zip(*grad_and_vars)
        grad = tf.reduce_mean(tf.stack(grads, axis=0), axis=0)

        # All vars are of the same value, using the first tower here.
        average_grads.append((grad, variables[0]))

    return average_grads


def _log_summaries(input_image, label, num_of_classes, output):
    """Logs the summaries for the model.

	Args:
	  input_image: Input image of the model. Its shape is [batch_size, height,
		width, channel].
	  label: Label of the image. Its shape is [batch_size, height, width].
	  num_of_classes: The number of classes of the dataset.
	  output: Output of the model. Its shape is [batch_size, height, width].
	"""
    # Add summaries for model variables.
    for model_var in tf.model_variables():
        tf.summary.histogram(model_var.op.name, model_var)

    # Add summaries for images, labels, semantic predictions.
    if FLAGS.save_summaries_images:
        tf.summary.image('samples/%s' % common.IMAGE, input_image)

        # Scale up summary image pixel values for better visualization.
        pixel_scaling = max(1, 255 // num_of_classes)
        summary_label = tf.cast(label * pixel_scaling, tf.uint8)
        tf.summary.image('samples/%s' % common.LABEL, summary_label)

        predictions = tf.expand_dims(tf.argmax(output, 3), -1)
        summary_predictions = tf.cast(predictions * pixel_scaling, tf.uint8)
        tf.summary.image('samples/%s' % common.OUTPUT_TYPE, summary_predictions)



def _train_deeplab_model(iterator, num_of_classes, ignore_label,
                         freeze_backbone_network=False,
                         finetune_li_weight_only=False):
    """Trains the deeplab model.

	Args:
	  iterator: An iterator of type tf.data.Iterator for images and labels.
	  num_of_classes: Number of classes for the dataset.
	  ignore_label: Ignore label for the dataset.
	  freeze_backbone_network: Whether to freeze the backbone network's weights

	Returns:
	  train_tensor: A tensor to update the model variables.
	  summary_op: An operation to log the summaries.
	"""
    global_step = tf.train.get_or_create_global_step()

    learning_rate = train_utils.get_model_learning_rate(
        FLAGS.learning_policy, FLAGS.base_learning_rate,
        FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
        FLAGS.training_number_of_steps, FLAGS.learning_power,
        FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
    tf.summary.scalar('learning_rate', learning_rate)

    if FLAGS.optimizer == 'sgd':
        print('Using sgd with momentum optimizer ...')
        optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
    elif FLAGS.optimizer == 'adam':
        print('Using Adam optimizer')
        optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.adam_learning_rate, epsilon=FLAGS.adam_epsilon)
    else:
        raise ValueError('Unknown optimizer {}'.format(FLAGS.optimizer))

    tower_losses = []
    tower_grads = []
    for i in range(FLAGS.num_clones):
        with tf.device('/gpu:%d' % i):
            # First tower has default name scope.
            name_scope = ('clone_%d' % i) if i else ''
            with tf.name_scope(name_scope) as scope:
                loss = _tower_loss(
                    iterator=iterator,
                    num_of_classes=num_of_classes,
                    ignore_label=ignore_label,
                    scope=scope,
                    reuse_variable=(i != 0))
                tower_losses.append(loss)

    if FLAGS.quantize_delay_step >= 0:
        if FLAGS.num_clones > 1:
            raise ValueError('Quantization doesn\'t support multi-clone yet.')
        tf_contrib.quantize.create_training_graph(
            quant_delay=FLAGS.quantize_delay_step)

    for i in range(FLAGS.num_clones):
        with tf.device('/gpu:%d' % i):
            name_scope = ('clone_%d' % i) if i else ''
            with tf.name_scope(name_scope) as scope:
                if freeze_backbone_network:
                    # freeze variables in backbone network
                    backbone_name_scope = model_variant_to_name_scope[FLAGS.model_variant]
                    var_list = [v for v in tf.trainable_variables()
                                if not v.name.startswith(backbone_name_scope)]
                    grads = optimizer.compute_gradients(tower_losses[i], var_list=var_list)
                elif finetune_li_weight_only:
                    # only fine-tune the LI weights (all other variables are frozen)
                    var_list = [v for v in tf.trainable_variables()
                                if 'lateral_inhibition' in v.name]
                    grads = optimizer.compute_gradients(tower_losses[i], var_list=var_list)
                else:
                    grads = optimizer.compute_gradients(tower_losses[i])

                tower_grads.append(grads)

    with tf.device('/cpu:0'):
        grads_and_vars = _average_gradients(tower_grads)

        # Modify the gradients for biases and last layer variables.
        last_layers = model_extend.get_extra_layer_scopes(
            FLAGS.last_layers_contain_logits_only)
        grad_mult = train_utils.get_model_gradient_multipliers(
            last_layers, FLAGS.last_layer_gradient_multiplier)
        if grad_mult:
            grads_and_vars = tf_contrib.training.multiply_gradients(
                grads_and_vars, grad_mult)

        # Modify the gradients for LI weights
        li_weight_graident_mult = get_li_weight_gradient_multipliers(
            li_weight_gradient_multiplier=FLAGS.li_weight_gradient_multiplier)
        if li_weight_graident_mult:
            grads_and_vars = tf_contrib.training.multiply_gradients(
                grads_and_vars, li_weight_graident_mult)

        # Create gradient update op.
        grad_updates = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

        # Gather update_ops. These contain, for example,
        # the updates for the batch_norm variables created by model_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)

        total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

        # Print total loss to the terminal.
        # This implementation is mirrored from tf.slim.summaries.
        should_log = math_ops.equal(math_ops.mod(global_step, FLAGS.log_steps), 0)
        total_loss = tf.cond(
            should_log,
            lambda: tf.Print(total_loss, ['Step', global_step, 'Total loss', total_loss]),
            lambda: total_loss)

        tf.summary.scalar('total_loss', total_loss)

        li_vars = [v for v in tf.model_variables() if 'li_weights' in v.name]
        if len(li_vars) > 0:
            li_weights = tf.reduce_mean([tf.reduce_mean(var) for var in li_vars])
            tf.summary.scalar('Lateral_inhibition_mean_weight', li_weights)

        with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(total_loss, name='train_op')

        # Excludes summaries from towers other than the first one.
        summary_op = tf.summary.merge_all(scope='(?!clone_)')

    return train_tensor, summary_op


def main(unused_argv):

    if FLAGS.display_warning:
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        tf.logging.set_verbosity(tf.logging.ERROR)

    flag_setter.check_options()
    flag_setter.print_flags()

    tf.gfile.MakeDirs(FLAGS.train_logdir)
    print('Training on {} set'.format(FLAGS.train_split))

    graph = tf.Graph()
    with graph.as_default():
        with tf.device(tf.train.replica_device_setter(ps_tasks=FLAGS.num_ps_tasks)):
            assert FLAGS.train_batch_size % FLAGS.num_clones == 0, (
                'Training batch size not divisble by number of clones (GPUs).')

            clone_batch_size = FLAGS.train_batch_size // FLAGS.num_clones
            dataset = data_generator_extend.Dataset(
                dataset_name=FLAGS.dataset,
                split_name=FLAGS.train_split,
                dataset_dir=FLAGS.dataset_dir,
                batch_size=clone_batch_size,
                crop_size=[int(sz) for sz in FLAGS.train_crop_size],
                min_resize_value=FLAGS.min_resize_value,
                max_resize_value=FLAGS.max_resize_value,
                resize_factor=FLAGS.resize_factor,
                min_scale_factor=FLAGS.min_scale_factor,
                max_scale_factor=FLAGS.max_scale_factor,
                scale_factor_step_size=FLAGS.scale_factor_step_size,
                model_variant=FLAGS.model_variant,
                num_readers=2,
                is_training=True,
                should_shuffle=True,
                should_repeat=True)

            train_tensor, summary_op = _train_deeplab_model(
                dataset.get_one_shot_iterator(), dataset.num_of_classes,
                dataset.ignore_label,
                freeze_backbone_network=FLAGS.freeze_backbone_network,
                finetune_li_weight_only=FLAGS.finetune_li_weight_only)

            # Soft placement allows placing on CPU ops without GPU implementation.
            session_config = tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False)
            session_config.gpu_options.allow_growth = FLAGS.allow_growth

            last_layers = model_extend.get_extra_layer_scopes(
                FLAGS.last_layers_contain_logits_only)
            init_fn = None
            if FLAGS.tf_initial_checkpoint:
                init_fn = train_utils.get_model_init_fn(
                    FLAGS.train_logdir,
                    FLAGS.tf_initial_checkpoint,
                    FLAGS.initialize_last_layer,
                    last_layers,
                    ignore_missing_vars=True)

            scaffold = tf.train.Scaffold(
                init_fn=init_fn,
                summary_op=summary_op,
                saver=tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
            )

            stop_hook = tf.train.StopAtStepHook(
                last_step=FLAGS.training_number_of_steps)

            profile_dir = FLAGS.profile_logdir
            if profile_dir is not None:
                tf.gfile.MakeDirs(profile_dir)

            with tf_contrib.tfprof.ProfileContext(
                    enabled=profile_dir is not None, profile_dir=profile_dir):
                with tf.train.MonitoredTrainingSession(
                        master=FLAGS.master,
                        is_chief=(FLAGS.task == 0),
                        config=session_config,
                        scaffold=scaffold,
                        checkpoint_dir=FLAGS.train_logdir,
                        summary_dir=FLAGS.train_logdir,
                        log_step_count_steps=FLAGS.log_steps,
                        save_summaries_secs=FLAGS.save_summaries_secs,
                        save_summaries_steps=FLAGS.save_summaries_steps,
                        save_checkpoint_secs=FLAGS.save_interval_secs,
                        save_checkpoint_steps=FLAGS.save_checkpoint_steps,
                        hooks=[stop_hook]) as sess:
                    while not sess.should_stop():
                        sess.run([train_tensor])


if __name__ == '__main__':
    flags.mark_flag_as_required('config_file')
    config_file = sys.argv[2]
    flag_setter.read_config(config_file=config_file)
    tf.app.run()
