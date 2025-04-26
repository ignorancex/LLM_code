import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Callable
import tensorflow as tf
from transformers import get_cosine_schedule_with_warmup, AdamW, \
    get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
import torch

# https://www.pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/
# https://towardsdatascience.com/learning-rate-schedule-in-practice-an-example-with-keras-and-tensorflow-2-0-2f48b2888a0c


from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import PolynomialDecay,\
    CosineDecay, LinearCosineDecay, ExponentialDecay, CosineDecayRestarts,\
    NoisyLinearCosineDecay, InverseTimeDecay, PiecewiseConstantDecay, LearningRateSchedule

from src.utils import disable_tensorflow_gpus


class StepDecay(LearningRateSchedule):
    def __init__(self,
                 initial_learning_rate: float,
                 drop_factor: float,
                 drop_evey_step: int,
                 minimum_learning_rate=0.00001):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.drop_factor = drop_factor
        self.drop_evey_step = drop_evey_step
        self.minimum_learning_rate = minimum_learning_rate

    def __call__(self, step):
        # compute the learning rate for the current step
        exp = np.floor((1 + step) / self.drop_evey_step)
        alpha = self.initial_learning_rate * (self.drop_factor ** exp)
        if alpha < self.minimum_learning_rate:
            return float(self.minimum_learning_rate)
        else:
            # return the learning rate
            return float(alpha)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "drop_factor": self.drop_factor,
            "drop_evey_step": self.drop_evey_step
        }



# based on https://huggingface.co/transformers/_modules/transformers/optimization_tf.html#WarmUp
class WarmUpDecay(LearningRateSchedule):
    """
    Applies a warmup schedule on a given learning rate decay schedule.

    Args:
        initial_learning_rate (:obj:`float`):
            The initial learning rate for the schedule after the warmup (so this will be the learning rate at the end
            of the warmup).
        decay_schedule_fn (:obj:`Callable`):
            The schedule function to apply after the warmup for the rest of training.
        warmup_steps (:obj:`int`):
            The number of steps for the warmup part of training.
        power (:obj:`float`, `optional`, defaults to 1):
            The power to use for the polynomial warmup (defaults is a linear warmup).
        name (:obj:`str`, `optional`):
            Optional name prefix for the returned tensors during the schedule.
    """

    def __init__(
        self,
        initial_learning_rate: float,
        decay_schedule_fn: Callable,
        warmup_steps: int,
        power: float = 1.0,
        name: str = None,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step - self.warmup_steps),
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }


def get_transformer_polynomial_decay_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by `lr_end`, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        lr_end (:obj:`float`, `optional`, defaults to 1e-7):
            The end LR.
        power (:obj:`float`, `optional`, defaults to 1.0):
            Power factor.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Note: `power` defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """

    lr_init = optimizer.defaults["lr"]
    assert lr_init > lr_end, f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})"

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def main():
    disable_tensorflow_gpus()

    #draw all optimizers
    steps = 10_000
    init_lr = 0.01
    end_lr = 0.0001

    # keras_poly = PolynomialDecay(initial_learning_rate=init_lr, decay_steps=steps,end_learning_rate=end_lr)
    # draw_scheduler(steps, keras_poly, os.path.join(SCHEDULERS_VIS_DIR, 'keras-poly.png') ,"LR Rate scheduler for Keras PolyDecay")
    #
    # keras_poly = PolynomialDecay(initial_learning_rate=init_lr, decay_steps=steps, cycle=True, end_learning_rate=end_lr)
    # draw_scheduler((steps*10), keras_poly, os.path.join(SCHEDULERS_VIS_DIR, 'keras-poly-repeat.png'),
    #                "LR Rate scheduler for Keras PolyDecay cycle true")
    #
    # # Cosine
    # keras_cosine = CosineDecay(init_lr, steps)
    # draw_scheduler(steps, keras_cosine, os.path.join(SCHEDULERS_VIS_DIR, 'keras-cosine.png'),
    #                "LR Rate scheduler for Keras Cosine")
    #
    # # alpha je minimum na ktery to klesne, zlomek (tj. procento) na ktery to klesne z initial learning rate
    # keras_cosine = CosineDecay(init_lr, steps, alpha=0.4)
    # draw_scheduler(steps, keras_cosine, os.path.join(SCHEDULERS_VIS_DIR, 'keras-cosine-alpha.png'),
    #                "LR Rate scheduler for Keras Cosine alpha=0.4")
    #
    # keras_cosine = CosineDecay(init_lr, steps)
    # draw_scheduler((steps*10), keras_cosine, os.path.join(SCHEDULERS_VIS_DIR, 'keras-cosine-more.png'),
    #                "LR Rate scheduler for Keras Cosine more")
    #
    # keras_cosine = CosineDecay(init_lr, steps, alpha=0.4)
    # draw_scheduler((steps * 10), keras_cosine, os.path.join(SCHEDULERS_VIS_DIR, 'keras-cosine-more-alpha.png'),
    #                "LR Rate scheduler for Keras Cosine more alpha")
    #
    # # Linear cosine
    # keras_cosine_lin = LinearCosineDecay(init_lr, steps)
    # draw_scheduler(steps, keras_cosine_lin, os.path.join(SCHEDULERS_VIS_DIR, 'linear-keras-cosine.png'),
    #                "LR Rate scheduler for Linear Keras Cosine")
    #
    # keras_cosine_lin = LinearCosineDecay(init_lr, steps)
    # draw_scheduler((steps * 10), keras_cosine_lin, os.path.join(SCHEDULERS_VIS_DIR, 'linear-keras-cosine-more.png'),
    #                "LR Rate scheduler for Linear Keras Cosine more")
    #
    # # CosineDecayRestarts
    # keras_cosine_res = CosineDecayRestarts(init_lr, steps)
    # draw_scheduler(steps, keras_cosine_res, os.path.join(SCHEDULERS_VIS_DIR, 'base-rest-keras-cosine.png'),
    #                "LR Rate scheduler for Linear Keras Cosine more")
    #
    # keras_cosine_res = CosineDecayRestarts(init_lr, 1000)
    # draw_scheduler(steps, keras_cosine_res, os.path.join(SCHEDULERS_VIS_DIR, 'rest-keras-cosine.png'),
    #                "LR Rate scheduler for Linear Keras Cosine more")
    #
    # keras_cosine_res = CosineDecayRestarts(init_lr, 1000)
    # draw_scheduler((steps * 10), keras_cosine_res, os.path.join(SCHEDULERS_VIS_DIR, 'rest-keras-cosine-more.png'),
    #                "LR Rate scheduler for Linear Keras Cosine more")
    #
    # # Noisy linear decay
    # keras_noisy_cos = NoisyLinearCosineDecay(init_lr, steps)
    # draw_scheduler(steps, keras_noisy_cos, os.path.join(SCHEDULERS_VIS_DIR, 'noisy-keras-cosine.png'),
    #                "LR Rate scheduler for Noisy Keras Cosine more")
    #
    # keras_noisy_cos = NoisyLinearCosineDecay(init_lr, steps)
    # draw_scheduler((steps * 10), keras_noisy_cos, os.path.join(SCHEDULERS_VIS_DIR, 'noisy-keras-cosine-more.png'),
    #                "LR Rate scheduler for Noisy Keras Cosine more")
    #
    # # Exponential
    # keras_expon = ExponentialDecay(init_lr, steps, 0.05)
    # draw_scheduler(steps, keras_expon, os.path.join(SCHEDULERS_VIS_DIR, 'exp-keras-dec-0.05.png'),
    #                "LR Rate scheduler for Exp Keras ")
    #
    # keras_expon = ExponentialDecay(init_lr, steps, 0.1)
    # draw_scheduler(steps, keras_expon, os.path.join(SCHEDULERS_VIS_DIR, 'exp-keras-dec-0.1.png'),
    #                "LR Rate scheduler for Exp Keras ")
    #
    # keras_expon = ExponentialDecay(init_lr, steps, 0.1,staircase=True)
    # draw_scheduler(steps, keras_expon, os.path.join(SCHEDULERS_VIS_DIR, 'exp-keras-dec-0.1-stair.png'),
    #                "LR Rate scheduler for Exp Keras stair")
    #
    # keras_expon = ExponentialDecay(init_lr, steps, 0.05)
    # draw_scheduler((steps * 10), keras_expon, os.path.join(SCHEDULERS_VIS_DIR, 'exp-keras-dec-0.05-more.png'),
    #                "LR Rate scheduler for Exp Keras ")
    #
    # keras_expon = ExponentialDecay(init_lr, steps, 0.05, staircase=True)
    # draw_scheduler(steps, keras_expon, os.path.join(SCHEDULERS_VIS_DIR, 'exp-keras-dec-0.05-stair-more.png'),
    #                "LR Rate scheduler for Exp Keras stair")
    #
    # # InverseTimeDecay
    # keras_invers = InverseTimeDecay(init_lr, steps, 0.05)
    # draw_scheduler(steps, keras_invers, os.path.join(SCHEDULERS_VIS_DIR, 'inverse-keras-dec-0.05.png'),
    #                "Inverse time k")
    #
    # keras_invers = InverseTimeDecay(init_lr, steps, 0.01)
    # draw_scheduler(steps, keras_invers, os.path.join(SCHEDULERS_VIS_DIR, 'inverse-keras-dec-0.01.png'),
    #                "Inverse time k")
    #
    # keras_invers = InverseTimeDecay(init_lr, steps, 0.01)
    # draw_scheduler((steps*10), keras_invers, os.path.join(SCHEDULERS_VIS_DIR, 'inverse-keras-dec-0.01-more.png'),
    #                "Inverse time k")
    #
    # # Piecewise
    # boundaries = [3000, 5000]
    # values = [1.0, 0.5, 0.1]
    # keras_constant = PiecewiseConstantDecay(boundaries, values)
    # draw_scheduler(steps, keras_constant,os.path.join(SCHEDULERS_VIS_DIR, 'piecewise.png'),
    #                'Piecewise constant')
    #
    # boundaries = [3000, 5000]
    # values = [1.0, 0.5, 0.1]
    # keras_constant = PiecewiseConstantDecay(boundaries, values)
    # draw_scheduler((steps*10), keras_constant, os.path.join(SCHEDULERS_VIS_DIR, 'piecewise-more.png'),
    #                'Piecewise constant more')
    #
    # # Warmup cosine
    #
    # keras_cosine = CosineDecay(init_lr, steps)
    # cosine_warm = WarmUpDecay(init_lr, keras_cosine, 1000)
    # draw_scheduler(steps, cosine_warm, os.path.join(SCHEDULERS_VIS_DIR, 'warm-up-cosine.png'),
    #                'Warm up with cosine')
    #
    # step_decay = StepDecay(init_lr, 0.5, 1250)
    # cosine_warm = WarmUpDecay(init_lr, step_decay, 1000)
    # draw_scheduler(steps, cosine_warm, os.path.join(SCHEDULERS_VIS_DIR, 'warm-up-step_by.png'),
    #                'Warm up with step')
    #
    # # Step decay
    # draw_scheduler(steps, step_decay, os.path.join(SCHEDULERS_VIS_DIR, 'step_by.png'),
    #                'Decay up with step')


    # Transformers schedulers
    # model = torch.nn.Sequential(
    #     torch.nn.Linear(3, 1),
    #     torch.nn.Flatten(0, 1)
    # )
    # optimizer = AdamW(model.parameters(), lr=init_lr, correct_bias=False)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=1000,
    #     num_training_steps=steps
    # )
    #
    # draw_scheduler_transformers(steps, scheduler, os.path.join(SCHEDULERS_VIS_DIR, 'transformer-linear-wrp.png'))
    #
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=1000,
    #     num_training_steps=steps
    # )
    #
    # draw_scheduler_transformers(steps, scheduler, os.path.join(SCHEDULERS_VIS_DIR, 'transformer-cosine-wrp.png'))
    #
    # scheduler = get_transformer_polynomial_decay_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=1000,
    #     num_training_steps=steps,
    #     power=2
    # )
    #
    # draw_scheduler_transformers(steps, scheduler, os.path.join(SCHEDULERS_VIS_DIR, 'transformer-polynomial-wrp.png'))




def draw_scheduler_transformers(steps, scheduler, fig_path, title="LR Rate scheduler"):
    to_plot = []
    for n in range(steps):
        scheduler.step()
        to_plot.append(scheduler.get_lr()[0])

    steps = np.arange(0, steps)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(steps, to_plot)
    plt.title(title)
    plt.xlabel("Steps #")
    plt.ylabel("Learning Rate")
    plt.savefig(fig_path)
    plt.clf()
    plt.close()


def draw_scheduler(steps, scheduler, fig_path, title="LR Rate scheduler"):
    lrs = [scheduler(i) for i in range(steps)]
    steps = np.arange(0, steps)
    # the learning rate schedule
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(steps, lrs)
    plt.title(title)
    plt.xlabel("Steps #")
    plt.ylabel("Learning Rate")
    plt.savefig(fig_path)
    plt.clf()
    plt.close()



if __name__ == '__main__':
    main()