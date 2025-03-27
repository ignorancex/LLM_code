# Written by Chun Kit Wong and CIRC under MIT license:
# https://github.com/wong-ck/DeepSegment/blob/master/LICENSE

import tensorflow as tf

from utilities import log
from utilities import config
from utilities.tfwrapper import estimator
from utilities.tfwrapper import input_fn
from utilities.tfwrapper import hook


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

    # create training and evaluation hooks
    train_hooks = hook.create_hooks(ds_config, mode="train")
    eval_hooks = hook.create_hooks(ds_config, mode="eval", segmenter=segmenter)

    # extract required configuration params
    iter_unit = ds_config.general_iteration_unit
    if iter_unit not in ["epoch", "step"]:
        msg = "unknown iteration unit {:}".format(iter_unit)
        logger.error(msg)
        raise ValueError(msg)

    eval_freq = ds_config.evaluate_freq
    niters = ds_config.train_iterations

    # train and evaluate model
    it = 0
    while it < niters:
        # calculate number of iterations (epochs or steps) to perform
        if iter_unit == "epoch":
            train_epochs = min(niters - it, eval_freq)
            train_steps = None
        elif iter_unit == "step":
            train_epochs = None
            train_steps = min(niters - it, eval_freq)

        # create input functions
        train_input_fn = input_fn.create_input_fn(
            ds_config,
            mode="train",
            epochs=train_epochs,
        )
        eval_input_fn = input_fn.create_input_fn(
            ds_config,
            mode="eval",
            epochs=1,
        )

        # train
        segmenter.train(
            input_fn=train_input_fn,
            steps=train_steps,
            hooks=train_hooks,
        )

        # evaluate
        eval_results = segmenter.evaluate(
            input_fn=eval_input_fn,
            hooks=eval_hooks,
        )

        logger.info("Validation result: " + str(eval_results))

        # increment iteration counter
        it += min(niters - it, eval_freq)

    logger.info("done!")


if __name__ == "__main__":
    main()
