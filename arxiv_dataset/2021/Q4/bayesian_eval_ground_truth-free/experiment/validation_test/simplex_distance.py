import numpy as np

from experiment.sjd import sjd_log_prob_exp


def transform_to(sample, transform_matrix, origin_adjust=None):
    """Transforms the sample from discrete distribtuion space into the
    probability simplex space of one dimension less. The orgin adjustment
    is used to move to the correct origin of the probability simplex space.
    """
    if origin_adjust is None:
        origin_adjust = np.zeros(transform_matrix.shape[1])
        origin_adjust[0] = 1
    #return transform_matrix @ (sample - origin_adjust)
    return (sample - origin_adjust) @ transform_matrix.T


def transform_from(sample, transform_matrix, origin_adjust=None):
    if origin_adjust is None:
        origin_adjust = np.zeros(transform_matrix.shape[1])
        origin_adjust[0] = 1
    return (transform_matrix.T @ sample) + origin_adjust


def get_simplex_distances(
    dir_path,
    weights_file,
    label_src,
):
    # visualize the probability simplex distances of the given data.
    model, features, labels = sjd_log_prob_exp.load_eval_fold(
        dir_path,
        weights_file,
        label_src,
    )

    pred = (model.predict(features[0]), model.predict(features[1]))
    del model
    del features

    # get transform matrix
    # Create n-1 spanning vectors, Using first dim as origin of new basis.
    spanning_vectors = np.vstack((
        -np.ones(len(labels[0][0]) -1),
        np.eye(len(labels[0][0]) -1),
    ))

    # Create orthonormal basis of simplex
    transform_matrix = np.linalg.qr(spanning_vectors)[0].T
    origin_adjust = np.zeros(len(labels[0][0]))

    # transform the labels and predictions into prob simplex space
    pred = (
        transform_to(pred[0], transform_matrix, origin_adjust),
        transform_to(pred[1], transform_matrix, origin_adjust),
    )
    labels = (
        transform_to(labels[0], transform_matrix, origin_adjust),
        transform_to(labels[1], transform_matrix, origin_adjust),
    )

    return pred[0] - labels[0], pred[1] - labels[1], transform_matrix, origin_adjust
