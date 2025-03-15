import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.losses import categorical_crossentropy

LARGE_NUM = 1e9


def __contrastive_loss(hidden, hidden_norm: bool = True, temperature: float = 1.0, weights: float = 1.0):
    """
    Notes on original method:
    - hidden: Tensor, shape is (1024, 128) where 1024 are the training samples (512 "original" batch size),
              and 128 is the feature vector. It seems that hidden[0:512, :] are the 512 training samples
              from the first augmentation, and hidden[512:1024, :] are the 512 training samples from the
              second augmentation
    """
    if hidden_norm:
        hidden = tf.math.l2_normalize(hidden, -1)

    hidden1, hidden2 = tf.split(hidden, 2, 0)
    batch_size = tf.shape(hidden1)[0]

    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_aa = tf.matmul(hidden1, hidden1, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(hidden2, hidden2, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(hidden1, hidden2, transpose_b=True) / temperature
    logits_ba = tf.matmul(hidden2, hidden1, transpose_b=True) / temperature
    # NOTE logits_ba is actually logits_ab.transpose() but we compute it as per the original implementation

    loss_a = categorical_crossentropy(labels, tf.concat([logits_ab, logits_aa], 1), True)
    loss_b = categorical_crossentropy(labels, tf.concat([logits_ba, logits_bb], 1), True)
    loss = (loss_a + loss_b) / 2
    loss = tf.math.reduce_mean(loss)

    return loss, logits_ab, labels


def create_contrastive_loss(hidden_norm: bool = True, temperature: float = 1.0, weights: float = 1.0):
    """
    Creates a loss function based on Keras definition: https://keras.io/api/losses/#creating-custom-losses

    The loss function has the following signature:
        ``(y_true_unused: Tensor, y_pred: Tensor) -> Tensor``

    Note that the ground-truth labels, ``y_true_unused``, are not used at all (hence "unused"), ``y_pred`` should be
    logits, and a single scalar is returned as the loss.

    :param hidden_norm: Whether to apply L2 normalization to ``y_pred``
    :param temperature:
    :param weights:
    :return: A loss function
    """
    # TODO weights are not supported yet
    if weights != 1.0:
        return NotImplementedError("Weights is not implemented yet")

    def cl(_y_true: Tensor, y_pred: Tensor) -> Tensor:
        """Note that the ground-truth labels, ``_y_true``, are not used at all in this function."""
        loss, logits_ab, labels = __contrastive_loss(y_pred, hidden_norm, temperature, weights)

        return loss

    return cl
