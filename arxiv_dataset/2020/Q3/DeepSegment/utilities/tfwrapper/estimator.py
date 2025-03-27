# Written by Chun Kit Wong and CIRC under MIT license:
# https://github.com/wong-ck/DeepSegment/blob/master/LICENSE

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.estimator import estimator as estimator_tf_orig
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.estimator import model_fn as model_fn_lib

from utilities.tfwrapper import model
from utilities.tfwrapper import loss
from utilities.tfwrapper import optimizer
from utilities.tfwrapper import summary
from utilities.tfwrapper import metric


def create_segmenter(ds_config):
    # create tf.estimator.RunConfig
    run_config = tf.estimator.RunConfig(
        model_dir=ds_config.checkpoint_path,
        tf_random_seed=ds_config.general_random_seed,
        save_summary_steps=ds_config.summary_freq_steps,
        save_checkpoints_steps=ds_config.checkpoint_freq_steps,
        keep_checkpoint_max=ds_config.checkpoint_keep_max
    )

    # wrap configurations required by estimator as dictionary
    segmenter_params = {}
    segmenter_params['image_nclasses'] = ds_config.image_nclasses
    segmenter_params['image_classnames'] = ds_config.image_classnames
    segmenter_params['image_classweights'] = ds_config.image_classweights

    segmenter_params['network_model'] = ds_config.network_model
    segmenter_params['network_batchnorm'] = ds_config.network_batchnorm
    segmenter_params['network_dropout'] = ds_config.network_dropout

    segmenter_params['train_learning_rate'] = ds_config.train_learning_rate
    segmenter_params['train_l2_reg_factor'] = ds_config.train_l2_reg_factor
    segmenter_params['train_loss'] = ds_config.train_loss
    segmenter_params['train_weightage_mask'] = ds_config.train_weightage_mask
    segmenter_params['train_optimizer'] = ds_config.train_optimizer

    segmenter_params['summary_nimages'] = ds_config.summary_nimages

    # create and return segmenter object
    segmenter = Segmenter(
        segmenter_model_fn, config=run_config, params=segmenter_params
    )

    return segmenter


def segmenter_model_fn(features, labels, mode, params, config):
    # extract required params
    nclasses = params['image_nclasses']
    classnames = params['image_classnames']
    classweights = params['image_classweights']

    model_str = params['network_model']
    batchnorm = params['network_batchnorm']
    dropout_rate = params['network_dropout']

    learning_rate = params['train_learning_rate']
    l2_scale = params['train_l2_reg_factor']
    optimizer_str = params['train_optimizer']
    loss_str = params['train_loss']
    weightmask_mode = params['train_weightage_mask']

    n_summary_images = params['summary_nimages']

    # dropout under training mode only
    if mode != tf.estimator.ModeKeys.TRAIN:
        dropout_rate = 0.0

    # create model
    model_func = model.create_model_from_string(model_str)

    # reshape input image to desired shape
    input_img = tf.expand_dims(features["images"], axis=-1)

    # feed input image to model
    logits = model_func(
        input=input_img,
        nclasses=nclasses,
        batchnorm=batchnorm,
        dropout_rate=dropout_rate,
        l2_scale=l2_scale
    )

    # generate predictions (mode: PREDICT, EVAL)
    predictions = {
        "labels": tf.argmax(input=logits, axis=-1),
        # add `softmax_tensor` to graph (for PREDICT)
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # create loss function (mode: TRAIN, EVAL)
    loss_func = loss.create_loss_from_string(loss_str)

    # calculate loss (mode: TRAIN, EVAL)
    loss_total = 0.0

    # - loss function
    if ("presences" in labels.keys()):
        presences = tf.convert_to_tensor(labels["presences"], name="presences")
    else:
        presences = None

    loss_total += loss_func(
        labels=labels["images"],
        logits=logits,
        presences=presences,
        weights=classweights,
        weightmask_mode=weightmask_mode
    )

    # - regularization
    loss_total += tf.identity(
        tf.losses.get_regularization_loss(), name="l2_regularization_loss"
    )

    loss_total = tf.identity(loss_total, name="total_loss")

    # tf.summary: add feature image (mode: TRAIN)
    summary.ndim_image(
        name='zimg',
        tensor=features["images"],
        max_outputs=n_summary_images,
    )

    # tf.summary: add ground-truth label image (mode: TRAIN)
    summary.ndim_image(
        name='gt',
        tensor=labels["images"],
        min_value=0,
        max_value=nclasses,
        max_outputs=n_summary_images,
    )

    # tf.summary: add predicted label image (mode: TRAIN)
    summary.ndim_image(
        name='pred',
        tensor=predictions["labels"],
        min_value=0,
        max_value=nclasses,
        max_outputs=n_summary_images,
    )

    # create evaluation metrics (mode: EVAL)
    eval_metric_ops = {}

    # tf.summary and tf.metric: add accuracy (mode: TRAIN and EVAL)
    accuracy = tf.metrics.accuracy(
        labels=labels["images"], predictions=predictions["labels"]
    )
    eval_metric_ops["classification_accuracy"] = accuracy

    tf.summary.scalar('classification_accuracy', accuracy[1])

    # tf.summary: add categorical dice scores (mode: TRAIN)
    summary.categorical_dices(
        labels=labels["images"],
        predictions=predictions["labels"],
        nclasses=nclasses,
        presences=presences,
        classnames=classnames,
    )

    # tf.metric: add categorical_dice (mode: EVAL)
    categorical_dices = metric.categorical_dices(
        labels=labels["images"],
        predictions=predictions["labels"],
        nclasses=nclasses,
        presences=presences
    )
    for classname, categorical_dice in zip(classnames, categorical_dices):
        eval_metric_ops["dice_" + classname] = categorical_dice

    # tf.summary: add average dice score (mode: TRAIN)
    summary.average_dice(
        labels=labels["images"],
        predictions=predictions["labels"],
        nclasses=nclasses,
        presences=presences,
        classnames=classnames,
        weights=classweights,
    )

    # tf.metric: add average dice score (mode: EVAL)
    average_dice = metric.average_dice(
        labels=labels["images"],
        predictions=predictions["labels"],
        nclasses=nclasses,
        presences=presences,
        weights=classweights
    )
    eval_metric_ops["dice_mean"] = average_dice

    if mode == tf.estimator.ModeKeys.TRAIN:
        # tf.summary: add trainable variables (mode: TRAIN)
        summary.trainables()

        # create optimizer
        opt_class = optimizer.create_optimizer_from_string(optimizer_str)

        # configure Training Op (mode: TRAIN)
        global_step = tf.train.get_global_step()
        train_op = opt_class(learning_rate=learning_rate)       \
                            .minimize(loss=loss_total,          \
                                      global_step=global_step)

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_total, train_op=train_op
        )
    else:
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_total, eval_metric_ops=eval_metric_ops
        )


class Segmenter(tf.estimator.Estimator):
    def __init__(self, model_fn, model_dir=None, config=None, params=None):
        super(Segmenter, self).__init__(model_fn, model_dir, config, params)

    def get_eval_metrics(self, input_fn):
        with ops.Graph().as_default() as g:
            # obtain features and labels from input_fn
            features, labels = self._get_features_and_labels_from_input_fn(
                input_fn, model_fn_lib.ModeKeys.EVAL
            )

            # pass features and labels to model_fn
            # and obtain estimator_spec (eval)
            estimator_spec = self._call_model_fn(
                features, labels, model_fn_lib.ModeKeys.EVAL, self.config
            )

            # add loss to estimator_spec
            estimator_spec.eval_metric_ops[model_fn_lib.LOSS_METRIC_KEY
                                          ] = metrics_lib.mean(
                                              estimator_spec.loss
                                          )

            # extract eval_dict from estimator_spec
            _, eval_dict = estimator_tf_orig._extract_metric_update_ops(
                estimator_spec.eval_metric_ops
            )

            # add global step to eval_dict
            global_step_tensor = self._create_and_assert_global_step(g)
            eval_dict[ops.GraphKeys.GLOBAL_STEP] = global_step_tensor

            # extract tensor names
            eval_metrics = {}
            for k, v in eval_dict.items():
                eval_metrics[k] = v.name

        return eval_metrics
