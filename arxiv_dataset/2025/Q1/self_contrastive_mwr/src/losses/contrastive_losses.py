import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.losses.metric_learning import pairwise_distance


def contrastive_loss(margin=1.0):
    """
    Compute the contrastive loss for a batch of samples.

    Args:
        margin (float, optional): The margin value for the contrastive loss. Defaults to 1.0.

    Returns:
        A function that computes the batchwise contrastive loss.
    """
    def batchwise_contrastive_loss(y_true, y_pred):
        # y_pred 2D tensor
        pair_distances = pairwise_distance(feature=y_pred, squared=False)
        flatten_pair_distanes = tf.reshape(pair_distances, (-1,))
        
        label_matrix = tf.math.abs(tf.expand_dims(y_true, axis=1) - tf.expand_dims(y_true, axis=0))
        # Change values so 1 indicates same pair
        label_matrix = tf.math.abs(label_matrix - 1)
        flatten_label_matrix = tf.reshape(label_matrix, (-1,))
        
        return tfa.losses.contrastive_loss(flatten_label_matrix, flatten_pair_distanes,
                                           margin=margin)
    
    return batchwise_contrastive_loss


def npairs_loss():
    """
    Computes the npairs loss for a batch of samples.

    Returns:
        A function that computes the npairs loss for a batch of samples.
    """
    def batchwise_npairs_loss(y_true, y_pred):
        # y_pred 2D tensor
        pair_distances = pairwise_distance(feature=y_pred, squared=False)
        
        return tfa.losses.npairs_loss(tf.squeeze(y_true), pair_distances)
    
    return batchwise_npairs_loss