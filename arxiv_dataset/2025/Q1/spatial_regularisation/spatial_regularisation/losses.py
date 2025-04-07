import torch
from spatial_regularisation.utils import matrix_to_angles, make_gaussian


def xfm_loss(true, pred, weight_r=5.e3, weight_t=1.e3):
    """Supervised loss between the predicted and GT transforms in homogeneous representations.
    The translation loss is always L2, be rotation can also be geodesic. Different weights can be assigned to
    translation/rotation parts."""

    # rotation loss
    true_r = true[:, :3, :3]
    pred_r = pred[:, :3, :3]
    err_r = (true_r - pred_r) ** 2
    err_r = err_r.mean()

    # translation loss
    true_t = true[:, :3, 3]
    pred_t = pred[:, :3, 3]
    err_t = (true_t - pred_t) ** 2
    err_t = err_t.mean()

    return weight_r * err_r + weight_t * err_t


def image_loss(true, pred, loss_type='l2'):
    """Image voxel-wise loss. Can either be l2 or l1."""
    loss = true - pred
    if loss_type == 'l2':
        loss = torch.square(loss)
    elif loss_type == 'l1':
        loss = torch.abs(loss)
    else:
        raise ValueError('loss_type should be l2 (default) or l1, had %s' % loss_type)
    dim_to_sum = list(range(1, 2 + len(true.shape[2:])))  # sum along [C, *] where * are the image dimensions
    return torch.sum(loss, dim=dim_to_sum).mean()


def gaussian_kl_loss(tensor,
                     means,
                     covariance_matrices,
                     gaussian_type='anisotropic',
                     make_probabilistic=True):
    """This looks at whether the input tensor has a Gaussian topology by computing a loss (KL or L2) with a Gaussian
    distribution of the same mean and covariance matrix.
    :param tensor: input tensor for which we want to verify the Gaussian topology. Size needs to be [B, C, *],
    where * is the field of view ([H, W] in 2D or [H, W, D] in 3D).
    :param means: spatial means per batch and channel of the input tensors. Size is [B, C, n_dims].
    :param covariance_matrices: spatial covariance matrices of the input tensors. Size can be [B, C] if we want to
    compare the input tensor to an isotropic Gaussian or [B, C, n_dims, n_dims] (anisotropic Gaussians).
    :param gaussian_type: type of the Gaussian to use. This needs to match the shape of the provided covariance
    matrices. Can be "anisotropic" (default) or "isotropic".
    :param make_probabilistic: whether to make the input tensor a probability distribution where all values are in
    [0, 1] and each channel sums to 1.
    """

    # dimensions
    n_dims = means.shape[-1]
    im_shape = tensor.shape[-n_dims:]
    dim_to_sum = list(range(1, 2 + n_dims))  # sum along [C, *] where * are the image dimensions

    # make probabilistic
    if make_probabilistic:
        tensor = torch.abs(tensor)
        tensor = tensor / tensor.sum(dim=list(range(2, 2 + n_dims)) , keepdim=True)

    # create GT Gaussians
    with torch.no_grad():
        gaussian = make_gaussian(means, covariance_matrices, im_shape, gaussian_type)

    # compute loss
    loss = tensor * (torch.log(tensor + 1e-24) - torch.log(gaussian + 1e-24))

    return torch.sum(loss, dim=dim_to_sum).mean()


def fast_dice(true, pred, label_list=None, eps=1e-12):
    """Compute Dice function between two input tensors for the labels given in label_list. It returns a tensor of shape
    [N] where N is the length of the given label_list. If label_list is None, the function assumes the input tesnors are
    0/1 tensors and will compute the binary Dice (so the output is a scalar tensor).
    This function cannot be used for training as histogramdd is not differentiable. For training use sof_dice_loss."""

    if label_list is None or label_list == []:
        return dice(true, pred, eps).mean()

    elif len(label_list) == 1:
        return dice(true, pred, eps).mean()

    else:
        # build bins for histograms
        labels_sorted = torch.sort(label_list)[0]
        label_edges = torch.sort(torch.concat((labels_sorted - 0.1, labels_sorted + 0.1)))[0].to(torch.float32)
        bins = [label_edges, label_edges]

        # get histogram
        x = true.flatten().to(torch.float32)
        y = pred.flatten().to(torch.float32)
        hist = torch.histogramdd(torch.stack((x, y), dim=-1).cpu(), bins).hist.to(x.device)

        # get dice
        idx = torch.arange(start=0, end=2 * len(labels_sorted), step=2)
        dice_score = 2 * torch.diag(hist)[idx] / (torch.sum(hist, 0)[idx] + torch.sum(hist, 1)[idx] + 1e-12)
        return dice_score[torch.searchsorted(labels_sorted, label_list)]


def dice(true, pred, eps=1e-12):
    """Dice metric between binary masks (i.e., with 0/1 integers) of size [B, C, *] in torch."""
    dim_to_sum = list(range(2, 2 + len(true.shape[2:])))  # sum along [C, *] where * are the image dimensions
    intersection = torch.sum(pred * true, dim=dim_to_sum)
    cardinality = torch.sum(pred + true, dim=dim_to_sum)
    return 2. * intersection / (cardinality + eps)  # [B, C]


def soft_dice_loss(true, pred, eps=1e-12):
    """Computes soft Dice loss over 2 float32 tensors of shape [B, n_labels, *]"""
    dim_to_sum = list(range(2, 2 + len(true.shape[2:])))  # sum along [C, *] where * are the image dimensions
    intersection = torch.sum(pred * true, dim=dim_to_sum)
    cardinality = torch.sum(pred**2 + true**2, dim=dim_to_sum)
    dice_per_channel = 2. * (intersection + eps) / (cardinality + eps)
    return 1 - dice_per_channel.mean()


def l1_angle_from_matrix(true, pred):
    """L1 loss for rotations (in degrees) (useful to use as metric for testing)"""
    return torch.abs(matrix_to_angles(true) - matrix_to_angles(pred)) * 180 / torch.pi


def l1_translation(true, pred):
    """L1 loss for translation (useful to use as metric for testing)"""
    return torch.abs(true - pred)


def frobenius_norm(covariance):
    """returns frobenius norm of covariance, a torch tensor of shape [B, C, n_dims, n_dims]"""
    return torch.sum(covariance ** 2, dim=[2, 3]) if len(covariance.shape) > 2 else covariance ** 2  # [B, C]


def spectral_norm(covariance):
    """returns the mean spectral norm (i.e. maximum eigen value) of a torch tensor of shape [B, C, n_dims, n_dims]"""
    return torch.max(torch.linalg.eigvalsh(covariance), dim=-1)[0] if len(covariance.shape) > 2 else covariance  # [B,C]


def trace_norm(covariance):
    """returns the trace normalised by n_dims of an input tensor of shape [B, C, n_dims, n_dims]"""
    return torch.trace(covariance) / covariance.shape[-1] if len(covariance.shape) > 2 else covariance  # [B, C]


def repulsive_loss(points, temperature=1, mask=None):
    """Repulsive loss computed by maximising the spatial spread of points, which are given as tensor [B, C, n_dims].
    The spread is either measured by the spatial variance of the points, or the inter-points distances.
    Maximising spread in ]0; +Inf[ <=> maximising sigmoid(spread) in ]0.5; 1[
                                   <=> minimising 1 - sigmoid(spread) in ]0; 0.5[
    :param points: tensor of shape [B, C, n_dims]
    :param temperature: temperature of the sigmoid. Default is 1.
    :param mask: boolean mask of the strictly upper triangle. Providing the mask saves us from re-computing it every
    time we call this function"""
    dist = torch.cdist(points, points)
    mask = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1) if mask is None else mask
    return (mask * (1 - 1 / (1 + torch.exp(-dist / temperature)))).sum(dim=[1, 2]).mean()

def mean_point_dist(points):
    """Computes average distance between N points given as torch tensor [B, N, n_dims]. Returns a tensor of size [B]"""
    dist = torch.cdist(points, points)
    mask = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)
    return (dist * mask / torch.sum(mask, dim=[1, 2], keepdim=True)).sum(dim=[1, 2])  # [B]
