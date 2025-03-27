import math

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.experimental import CosineDecay
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class WarmUpAndCosineDecay(LearningRateSchedule):
    """Applies a warmup schedule on a given learning rate decay schedule."""

    def __init__(self, base_learning_rate: float, learning_rate_scaling: str, warmup_epochs: int, epochs: int,
                 batches_per_epoch: int, batch_size: int, name=None):
        super(WarmUpAndCosineDecay, self).__init__()

        self.base_learning_rate: float = base_learning_rate
        self.learning_rate_scaling: str = learning_rate_scaling
        self.warmup_epochs: int = warmup_epochs
        self.epochs: int = epochs
        self.batches_per_epoch: int = batches_per_epoch
        self.batch_size: int = batch_size
        self._name = name

        self.warmup_steps: int = self.warmup_epochs * self.batches_per_epoch
        self.total_steps: int = self.epochs * self.batches_per_epoch
        if self.learning_rate_scaling == 'linear':
            self.scaled_lr = self.base_learning_rate * self.batch_size / 256.
        elif self.learning_rate_scaling == 'sqrt':
            self.scaled_lr = self.base_learning_rate * math.sqrt(self.batch_size)
        else:
            raise ValueError('Unknown learning rate scaling {}'.format(self.learning_rate_scaling))
        self.cosine_decay: CosineDecay = CosineDecay(self.scaled_lr, self.total_steps - self.warmup_steps)

    def __call__(self, step: Tensor):
        with tf.name_scope(self._name or 'WarmUpAndCosineDecay'):
            if self.warmup_steps:
                learning_rate = tf.cast(step, tf.float32) / float(self.warmup_steps) * self.scaled_lr
            else:
                learning_rate = self.scaled_lr

            # learning_rate = (step / float(warmup_steps) * scaled_lr if warmup_steps else scaled_lr)

            # Cosine decay learning rate schedule
            learning_rate = tf.where(step < self.warmup_steps, learning_rate,
                                     self.cosine_decay(step - self.warmup_steps))

            return learning_rate

    def get_config(self):
        # TODO
        return NotImplementedError()
