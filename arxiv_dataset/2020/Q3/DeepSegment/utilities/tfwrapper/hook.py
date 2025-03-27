# Written by Chun Kit Wong and CIRC under MIT license:
# https://github.com/wong-ck/DeepSegment/blob/master/LICENSE

import os

import tensorflow as tf
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs

from utilities.tfwrapper import input_fn


def create_hooks(ds_config, mode, segmenter=None):
    # check that mode is valid
    if mode.lower() not in ["train", "eval", "pred"]:
        msg = "unknown mode {:}".format(mode)
        raise ValueError(msg)

    hooks = []

    if mode.lower() == "train":
        # extract required param from ds_config
        log_freq = ds_config.summary_freq_steps

        # determine tensors to be logged
        tensors_to_log = {}
        if len(ds_config.summary_tensors) != 0:
            for tensor in ds_config.summary_tensors:
                tensors_to_log[tensor] = tensor

        # create and append hook
        logging_hook = LoggingTensorAndLossHook(
            every_n_iter=log_freq,
            tensors=tensors_to_log,
        )

        hooks += [logging_hook]

    if mode.lower() == "eval":
        if segmenter is None:
            msg = "segmenter must be provided "
            msg += "to create BestCheckpointSaverHook"
            raise ValueError(msg)

        # extract required param from ds_config
        checkpoint_dir = ds_config.checkpoint_path
        max_to_keep = ds_config.checkpoint_keep_max

        # modify checkpoint_dir to eval subdirectory
        checkpoint_dir = os.path.join(checkpoint_dir, "eval")

        # create a dummy eval_input_fn
        # and generate eval_dict
        eval_input_fn = input_fn.create_input_fn(ds_config, mode="eval")
        eval_metrics = segmenter.get_eval_metrics(eval_input_fn)

        # specify which tensor to be used in determining best ckpt
        # and whether it should be maximized or minimized
        maximize_metrics = {}
        maximize_metrics['classification_accuracy'] = True
        maximize_metrics['loss'] = False
        maximize_metrics['dice_mean'] = True

        # create and append hook
        bestckptsaver_hook = BestCheckpointSaverHook(
            checkpoint_dir=checkpoint_dir,
            eval_metrics=eval_metrics,
            maximize_metrics=maximize_metrics,
            max_to_keep=max_to_keep,
        )

        hooks += [bestckptsaver_hook]

    if mode.lower() == "pred":
        # extract required param from ds_config
        image_batchsize = ds_config.image_batchsize

        # create and append hook
        predprogress_hook = PredictionProgressHook(batchsize=image_batchsize)

        hooks += [predprogress_hook]

    return hooks


# hook to report tensor values and progress
class LoggingTensorAndLossHook(tf.train.LoggingTensorHook):
    def begin(self):
        # add global step
        self._tensors['step'] = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0]

        # add regularization loss
        self._tensors['regularization_loss'
                     ] = tf.losses.get_regularization_loss()

        # add all other losses
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            self._tensors[loss.name.split('/')[0].split(':')[0]] = loss

        # required by base class; order of tensors when logging
        self._tag_order = self._tensors.keys()

        return super(LoggingTensorAndLossHook, self).begin()


# hook to save best checkpoint
class BestCheckpointSaverHook(tf.train.SessionRunHook):
    def __init__(
        self,
        eval_metrics=None,
        maximize_metrics=None,
        checkpoint_dir=None,
        max_to_keep=5,
        listeners=None
    ):
        tf.logging.info("Create BestCheckpointSaverHook.")

        # save variables
        self._eval_metrics = eval_metrics
        self._maximize_metrics = maximize_metrics
        self._ckpt_dir = checkpoint_dir
        self._max_to_keep = max_to_keep
        self._listeners = listeners or []

        # check and validate eval_metrics and maximize_metrics
        self._validate_eval_maximize_metrics()

        # initialize a dictionary to record best value
        self._best = {}
        for k, v in self._maximize_metrics.items():
            self._best[k] = float('Inf') * (-1.0 if v else 1.0)

    def _validate_eval_maximize_metrics(self):
        # create empty dictionary if not specified
        if self._eval_metrics is None:
            self._eval_metrics = {}

        if self._maximize_metrics is None:
            self._maximize_metrics = {}

        # check type
        if not isinstance(self._eval_metrics, dict):
            msg = "eval_metrics must be a dictionary"
            raise TypeError(msg)

        if not isinstance(self._maximize_metrics, dict):
            if not isinstance(self._maximize_metrics, bool):
                msg = "maximize_metrics must be either "
                msg += "a boolean value or a dictionary"
                raise TypeError(msg)

        # apply value to all keys of eval_metrics
        # if maximize_metrics is boolean
        if isinstance(self._maximize_metrics, bool):
            maximize = self._maximize_metrics

            self._maximize_metrics = {}
            for k, v in self._eval_metrics.items():
                self._maximize_metrics[k] = maximize

        # keep keys that exist in both
        # eval_metrics and maximize_metrics only
        for k in list(self._eval_metrics.keys()):
            if k not in self._maximize_metrics.keys():
                self._eval_metrics.pop(k, None)

        for k in list(self._maximize_metrics.keys()):
            if k not in self._eval_metrics.keys():
                self._maximize_metrics.pop(k, None)

    def _recover_best_metrics(self, metric):
        latest_ckpt_path = os.path.join(self._ckpt_dir, "best_" + str(metric))
        latest_ckpt = tf.train.latest_checkpoint(latest_ckpt_path)

        if latest_ckpt is not None:
            # ckpt_basename: model.{}-{}.ckpt

            best_recovered = os.path.basename(latest_ckpt)  \
                                    .split('.ckpt')[0]      \
                                    .split('model.')[1]     \
                                    .split('-')[-1]
            best_recovered = float(best_recovered)

            msg = "recovered best {:} ({:}) from checkpoint filename; "
            msg += "replacing current value in record {:}"
            msg = msg.format(metric, best_recovered, self._best[metric])
            tf.logging.debug(msg)

            self._best[metric] = best_recovered

    def begin(self):
        self._global_step = training_util._get_or_create_global_step_read()
        if self._global_step is None:
            msg = "global step should be created to use BestCheckpointSaverHook"
            raise RuntimeError(msg)

        self._savers = {}
        for metric in self._eval_metrics.keys():
            # create saver for metric
            self._savers[metric] = tf.train.Saver(
                max_to_keep=self._max_to_keep, save_relative_paths=True
            )

            # define output directory
            save_dir = os.path.join(self._ckpt_dir, "best_" + str(metric))
            if not os.path.exists(save_dir):
                # if not exist, create directory
                os.makedirs(save_dir)
            else:
                # if exist, attempt to recover last checkpoints
                states = tf.train.get_checkpoint_state(save_dir)
                if states is not None:
                    self._savers[metric].recover_last_checkpoints(
                        states.all_model_checkpoint_paths
                    )

                # attempt to recover previous best metrics from filename
                self._recover_best_metrics(metric)

        for l in self._listeners:
            l.begin()

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return SessionRunArgs(self._global_step)

    def end(self, session):
        last_step = session.run(self._global_step)

        for metric, tensor_name in self._eval_metrics.items():
            # # attempt to recover previous best metrics from checkpoint filename
            # self._recover_best_metrics(metric)

            # get tensor by name
            tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)

            # obtain tensor value
            metric_val = session.run(tensor)

            msg = "From BestCheckpointSaverHook: {:} - {:}"
            msg = msg.format(metric, metric_val)
            tf.logging.debug(msg)

            # check if metric_val is better than current record
            if self._maximize_metrics[metric]:
                new_best_found = metric_val > self._best[metric]
            else:
                new_best_found = metric_val < self._best[metric]

            if new_best_found:
                msg = "saving best {:}: {:} (previous best: {:})"
                msg = msg.format(metric, metric_val, self._best[metric])
                tf.logging.info(msg)

                # create ckpt filename
                ckpt_basename = "model.{}-{}.ckpt"
                ckpt_fullpath = os.path.join(
                    self._ckpt_dir, "best_" + str(metric),
                    ckpt_basename.format(metric, metric_val)
                )

                if not os.path.exists(os.path.dirname(ckpt_fullpath)):
                    os.makedirs(os.path.dirname(ckpt_fullpath))

                # save checkpoint
                self._savers[metric].save(
                    session, ckpt_fullpath, global_step=last_step
                )

                # overwrite value in self._best
                self._best[metric] = metric_val

        for l in self._listeners:
            l.end(session, last_step)


# hook to report prediction progress
class PredictionProgressHook(tf.train.SessionRunHook):
    def __init__(self, batchsize, **kwargs):
        self._batchsize = batchsize
        self._count = 0

        super(PredictionProgressHook, self).__init__(**kwargs)

    def after_run(self, run_context, run_values):
        self._count += 1

        msg = "predicted {:} mini-batches ({:} images)"
        msg = msg.format(self._count, self._count * self._batchsize)
        tf.logging.info(msg)

        return super(PredictionProgressHook,
                     self).after_run(run_context, run_values)
