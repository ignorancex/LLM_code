import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn
import itk
import copy
from collections import namedtuple


def scale_map(map, sz, spacing):
    """
    Scales the map to the [-1,1]^d format
    :param map: map in BxCxXxYxZ format
    :param sz: size of image being interpolated in XxYxZ format
    :param spacing: spacing of image in XxYxZ format
    :return: returns the scaled map
    """

    map_scaled = torch.zeros_like(map)
    ndim = len(spacing)

    # This is to compensate to get back to the [-1,1] mapping of the following form
    # id[d]*=2./(sz[d]-1)
    # id[d]-=1.

    for d in range(ndim):
        if sz[d + 2] > 1:
            map_scaled[:, d, ...] = (
                map[:, d, ...] * (2.0 / (sz[d + 2] - 1.0) / spacing[d])
                - 1.0
                # map[:, d, ...] * 2.0 - 1.0
            )
        else:
            map_scaled[:, d, ...] = map[:, d, ...]

    return map_scaled


class STNFunction_ND_BCXYZ:
    """
    Spatial transform function for 1D, 2D, and 3D. In BCXYZ format (this IS the format used in the current toolbox).
    """

    def __init__(
        self, spacing, zero_boundary=False, using_bilinear=True, using_01_input=True
    ):
        """
        Constructor
        :param ndim: (int) spatial transformation of the transform
        """
        self.spacing = spacing
        self.ndim = len(spacing)
        # zero_boundary = False
        self.zero_boundary = "zeros" if zero_boundary else "border"
        self.mode = "bilinear" if using_bilinear else "nearest"
        self.using_01_input = using_01_input

    def forward_stn(self, input1, input2, ndim):
        if ndim == 1:
            # use 2D interpolation to mimick 1D interpolation
            # now test this for 1D
            phi_rs = input2.reshape(list(input2.size()) + [1])
            input1_rs = input1.reshape(list(input1.size()) + [1])

            phi_rs_size = list(phi_rs.size())
            phi_rs_size[1] = 2

            phi_rs_ordered = torch.zeros(
                phi_rs_size, dtype=phi_rs.dtype, device=phi_rs.device
            )
            # keep dimension 1 at zero
            phi_rs_ordered[:, 1, ...] = phi_rs[:, 0, ...]

            output_rs = torch.nn.functional.grid_sample(
                input1_rs,
                phi_rs_ordered.permute([0, 2, 3, 1]),
                mode=self.mode,
                padding_mode=self.zero_boundary,
                align_corners=True,
            )
            output = output_rs[:, :, :, 0]

        if ndim == 2:
            # todo double check, it seems no transpose is need for 2d, already in height width design
            input2_ordered = torch.zeros_like(input2)
            input2_ordered[:, 0, ...] = input2[:, 1, ...]
            input2_ordered[:, 1, ...] = input2[:, 0, ...]

            if input2_ordered.shape[0] == 1 and input1.shape[0] != 1:
                input2_ordered = input2_ordered.expand(input1.shape[0], -1, -1, -1)
            output = torch.nn.functional.grid_sample(
                input1,
                input2_ordered.permute([0, 2, 3, 1]),
                mode=self.mode,
                padding_mode=self.zero_boundary,
                align_corners=True,
            )
        if ndim == 3:
            input2_ordered = torch.zeros_like(input2)
            input2_ordered[:, 0, ...] = input2[:, 2, ...]
            input2_ordered[:, 1, ...] = input2[:, 1, ...]
            input2_ordered[:, 2, ...] = input2[:, 0, ...]
            if input2_ordered.shape[0] == 1 and input1.shape[0] != 1:
                input2_ordered = input2_ordered.expand(input1.shape[0], -1, -1, -1, -1)
            output = torch.nn.functional.grid_sample(
                input1,
                input2_ordered.permute([0, 2, 3, 4, 1]),
                mode=self.mode,
                padding_mode=self.zero_boundary,
                align_corners=True,
            )
        return output

    def __call__(self, input1, input2):
        """
        Perform the actual spatial transform
        :param input1: image in BCXYZ format
        :param input2: spatial transform in BdimXYZ format
        :return: spatially transformed image in BCXYZ format
        """

        assert len(self.spacing) + 2 == len(input2.size())
        if self.using_01_input:
            output = self.forward_stn(
                input1, scale_map(input2, input1.shape, self.spacing), self.ndim
            )
        else:
            output = self.forward_stn(input1, input2, self.ndim)
        # print(STNVal(output, ini=-1).sum())
        return output


class STN_ND_BCXYZ:
    """
    Spatial transform code for nD spatial transoforms. Uses the BCXYZ image format.
    """

    def __init__(
        self,
        spacing,
        zero_boundary=False,
        use_bilinear=True,
        use_01_input=True,
        use_compile_version=False,
    ):
        self.spacing = spacing
        """spatial dimension"""
        if use_compile_version:
            if use_bilinear:
                self.f = STNFunction_ND_BCXYZ_Compile(self.spacing, zero_boundary)
            else:
                self.f = partial(get_nn_interpolation, spacing=self.spacing)
        else:
            self.f = STNFunction_ND_BCXYZ(
                self.spacing,
                zero_boundary=zero_boundary,
                using_bilinear=use_bilinear,
                using_01_input=use_01_input,
            )

        """spatial transform function"""

    def __call__(self, input1, input2):
        """
        Simply returns the transformed input
        :param input1: image in BCXYZ format
        :param input2: map in BdimXYZ format
        :return: returns the transformed image
        """
        return self.f(input1, input2)


def compute_warped_image_multiNC(
    I0, phi, spacing, spline_order, zero_boundary=False, use_01_input=True
):
    """Warps image.
    :param I0: image to warp, image size BxCxXxYxZ
    :param phi: map for the warping, size BxdimxXxYxZ
    :param spacing: image spacing [dx,dy,dz]
    :return: returns the warped image of size BxCxXxYxZ
    """

    dim = I0.dim() - 2
    if dim == 1:
        return _compute_warped_image_multiNC_1d(
            I0, phi, spacing, spline_order, zero_boundary, use_01_input=use_01_input
        )
    elif dim == 2:
        return _compute_warped_image_multiNC_2d(
            I0, phi, spacing, spline_order, zero_boundary, use_01_input=use_01_input
        )
    elif dim == 3:
        return _compute_warped_image_multiNC_3d(
            I0, phi, spacing, spline_order, zero_boundary, use_01_input=use_01_input
        )
    else:
        raise ValueError("Images can only be warped in dimensions 1 to 3")


def _compute_warped_image_multiNC_1d(
    I0, phi, spacing, spline_order, zero_boundary=False, use_01_input=True
):

    if spline_order not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        raise ValueError("Currently only orders 0 to 9 are supported")

    if spline_order == 0:
        stn = STN_ND_BCXYZ(
            spacing, zero_boundary, use_bilinear=False, use_01_input=use_01_input
        )
    elif spline_order == 1:
        stn = STN_ND_BCXYZ(
            spacing, zero_boundary, use_bilinear=True, use_01_input=use_01_input
        )
    else:
        stn = SplineInterpolation_ND_BCXYZ(spacing, spline_order)

    I1_warped = stn(I0, phi)

    return I1_warped


def _compute_warped_image_multiNC_2d(
    I0, phi, spacing, spline_order, zero_boundary=False, use_01_input=True
):

    if spline_order not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        raise ValueError("Currently only orders 0 to 9 are supported")

    if spline_order == 0:
        stn = STN_ND_BCXYZ(
            spacing, zero_boundary, use_bilinear=False, use_01_input=use_01_input
        )
    elif spline_order == 1:
        stn = STN_ND_BCXYZ(
            spacing, zero_boundary, use_bilinear=True, use_01_input=use_01_input
        )
    else:
        stn = SplineInterpolation_ND_BCXYZ(spacing, spline_order)

    I1_warped = stn(I0, phi)

    return I1_warped


def _compute_warped_image_multiNC_3d(
    I0, phi, spacing, spline_order, zero_boundary=False, use_01_input=True
):

    if spline_order not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        raise ValueError("Currently only orders 0 to 9 are supported")

    if spline_order == 0:
        # return get_warped_label_map(I0,phi,spacing)
        stn = STN_ND_BCXYZ(
            spacing, zero_boundary, use_bilinear=False, use_01_input=use_01_input
        )
    elif spline_order == 1:
        stn = STN_ND_BCXYZ(
            spacing, zero_boundary, use_bilinear=True, use_01_input=use_01_input
        )
    else:
        stn = SplineInterpolation_ND_BCXYZ(spacing, spline_order)

    I1_warped = stn(I0, phi)

    return I1_warped


def identity_map_multiN(sz, spacing, dtype="float32"):
    """
    Create an identity map
    :param sz: size of an image in BxCxXxYxZ format
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map
    """
    dim = len(sz) - 2
    nrOfI = int(sz[0])

    if dim == 1:
        id = np.zeros([nrOfI, 1, sz[2]], dtype=dtype)
    elif dim == 2:
        id = np.zeros([nrOfI, 2, sz[2], sz[3]], dtype=dtype)
    elif dim == 3:
        id = np.zeros([nrOfI, 3, sz[2], sz[3], sz[4]], dtype=dtype)
    else:
        raise ValueError(
            "Only dimensions 1-3 are currently supported for the identity map"
        )

    for n in range(nrOfI):
        id[n, ...] = identity_map(sz[2::], spacing, dtype=dtype)

    return id


def identity_map(sz, spacing, dtype="float32"):
    """
    Returns an identity map.
    :param sz: just the spatial dimensions, i.e., XxYxZ
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map of dimension dimxXxYxZ
    """
    dim = len(sz)
    if dim == 1:
        id = np.mgrid[0 : sz[0]]
    elif dim == 2:
        id = np.mgrid[0 : sz[0], 0 : sz[1]]
    elif dim == 3:
        id = np.mgrid[0 : sz[0], 0 : sz[1], 0 : sz[2]]
    else:
        raise ValueError(
            "Only dimensions 1-3 are currently supported for the identity map"
        )

    # now get it into range [0,(sz-1)*spacing]^d
    id = np.array(id.astype(dtype))
    if dim == 1:
        id = id.reshape(1, sz[0])  # add a dummy first index

    for d in range(dim):
        id[d] *= spacing[d]

        # id[d]*=2./(sz[d]-1)
        # id[d]-=1.

    # and now store it in a dim+1 array
    if dim == 1:
        idnp = np.zeros([1, sz[0]], dtype=dtype)
        idnp[0, :] = id[0]
    elif dim == 2:
        idnp = np.zeros([2, sz[0], sz[1]], dtype=dtype)
        idnp[0, :, :] = id[0]
        idnp[1, :, :] = id[1]
    elif dim == 3:
        idnp = np.zeros([3, sz[0], sz[1], sz[2]], dtype=dtype)
        idnp[0, :, :, :] = id[0]
        idnp[1, :, :, :] = id[1]
        idnp[2, :, :, :] = id[2]
    else:
        raise ValueError(
            "Only dimensions 1-3 are currently supported for the identity map"
        )

    return idnp

class RegistrationModule(nn.Module):
    r"""Base class for icon modules that perform registration.

    A subclass of RegistrationModule should have a forward method that
    takes as input two images image_A and image_B, and returns a python function
    phi_AB that transforms a tensor of coordinates.

    RegistrationModule provides a method as_function that turns a tensor
    representing an image into a python function mapping a tensor of coordinates
    into a tensor of intensities :math:`\mathbb{R}^N \rightarrow \mathbb{R}` .
    Mathematically, this is what an image is anyway.

    After this class is constructed, but before it is used, you _must_ call
    assign_identity_map on it or on one of its parents to define the coordinate
    system associated with input images.

    The contract that a successful registration fulfils is:
    for a tensor of coordinates X, self.as_function(image_A)(phi_AB(X)) ~= self.as_function(image_B)(X)

    ie

    .. math::
        I^A \circ \Phi^{AB} \simeq I^B

    In particular, self.as_function(image_A)(phi_AB(self.identity_map)) ~= image_B
    """

    def __init__(self):
        super().__init__()
        self.downscale_factor = 1

    def as_function(self, image):
        """image is a tensor with shape self.input_shape.
        Returns a python function that maps a tensor of coordinates [batch x N_dimensions x ...]
        into a tensor of intensities.
        """

        return lambda coordinates: compute_warped_image_multiNC(
            image, coordinates, self.spacing, 1
        )

    def assign_identity_map(self, input_shape, parents_identity_map=None):
        self.input_shape = np.array(input_shape)
        self.input_shape[0] = 1
        self.spacing = 1.0 / (self.input_shape[2::] - 1)

        # if parents_identity_map is not None:
        #    self.identity_map = parents_identity_map
        # else:
        _id = identity_map_multiN(self.input_shape, self.spacing)
        self.register_buffer("identity_map", torch.from_numpy(_id), persistent=False)

        if self.downscale_factor != 1:
            child_shape = np.concatenate(
                [
                    self.input_shape[:2],
                    np.ceil(self.input_shape[2:] / self.downscale_factor).astype(int),
                ]
            )
        else:
            child_shape = self.input_shape
        for child in self.children():
            if isinstance(child, RegistrationModule):
                child.assign_identity_map(
                    child_shape,
                    # None if self.downscale_factor != 1 else self.identity_map,
                )

    def adjust_batch_size(self, size):
        shape = self.input_shape
        shape[0] = size
        self.assign_identity_map(shape)

    def forward(image_A, image_B):
        """Register a pair of images:
        return a python function phi_AB that warps a tensor of coordinates such that

        .. code-block:: python

            self.as_function(image_A)(phi_AB(self.identity_map)) ~= image_B

        .. math::
            I^A \circ \Phi^{AB} \simeq I^B

        :param image_A: the moving image
        :param image_B: the fixed image
        :return: :math:`\Phi^{AB}`
        """
        raise NotImplementedError()


class FunctionFromVectorField(RegistrationModule):
    """
    Wrap an inner neural network 'net' that returns a tensor of displacements
    [B x N x H x W (x D)], into a RegistrationModule that returns a function that
    transforms a tensor of coordinates
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image_A, image_B):
        tensor_of_displacements = self.net(image_A, image_B)
        displacement_field = self.as_function(tensor_of_displacements)

        def transform(coordinates):
            if hasattr(coordinates, "isIdentity") and coordinates.shape == tensor_of_displacements.shape:
                return coordinates + tensor_of_displacements
            return coordinates + displacement_field(coordinates)

        return transform
    
class SquaringVelocityField(RegistrationModule):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.n_steps = 256

    def forward(self, image_A, image_B):
        velocityfield_delta = self.net(image_A, image_B) / self.n_steps

        for _ in range(8):
         velocityfield_delta = velocityfield_delta + self.as_function(
             velocityfield_delta)(velocityfield_delta + self.identity_map)
        def transform(coordinate_tensor):
           coordinate_tensor = coordinate_tensor + self.as_function(velocityfield_delta)(coordinate_tensor)
           return coordinate_tensor
        return transform


def multiply_matrix_vectorfield(matrix, vectorfield):
    dimension = len(vectorfield.shape) - 2
    if dimension == 2:
        batch_matrix_multiply = "ijkl,imj->imkl"
    else:
        batch_matrix_multiply = "ijkln,imj->imkln"
    return torch.einsum(batch_matrix_multiply, vectorfield, matrix)


class FunctionFromMatrix(RegistrationModule):
    """
    wrap an inner neural network `net` that returns an N x N+1 matrix representing
    an affine transform, into a RegistrationModule that returns a function that
    transforms a tensor of coordinates.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image_A, image_B):
        matrix_phi = self.net(image_A, image_B)

        def transform(tensor_of_coordinates):
            shape = list(tensor_of_coordinates.shape)
            shape[1] = 1
            coordinates_homogeneous = torch.cat(
                [tensor_of_coordinates, torch.ones(shape, device=tensor_of_coordinates.device)], axis=1
            )
            return multiply_matrix_vectorfield(matrix_phi, coordinates_homogeneous)[:, :-1]

        return transform


class RandomShift(RegistrationModule):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, image_A, image_B):
        shift_shape = (
            image_A.shape[0],
            len(image_A.shape) - 2,
            *(1 for _ in image_A.shape[2:]),
        )
        # In a real class, the variable that parametrizes the returned transform,
        # in this case shift, would be calculated from image_A and image_B before being captured
        # in the closure as below.
        shift = self.stddev * torch.randn(shift_shape, device=image_A.device)
        return lambda input_: input_ + shift


class TwoStepRegistration(RegistrationModule):
    """Combine two RegistrationModules.

    First netPhi is called on the input images, then image_A is warped with
    the resulting field, and then netPsi is called on warped A and image_B
    in order to find a residual warping. Finally, the composition of the two
    transforms is returned.
    """

    def __init__(self, netPhi, netPsi):
        super().__init__()
        self.netPhi = netPhi
        self.netPsi = netPsi

    def forward(self, image_A, image_B):
        
        # Tag for shortcutting hack. Must be set at the beginning of 
        # forward because it is not preserved by .to(config.device)
        self.identity_map.isIdentity = True
            
        phi = self.netPhi(image_A, image_B)
        psi = self.netPsi(
            self.as_function(image_A)(phi(self.identity_map)), 
            image_B
        )
        return lambda tensor_of_coordinates: phi(psi(tensor_of_coordinates))
        


class DownsampleRegistration(RegistrationModule):
    """
    Perform registration using the wrapped RegistrationModule `net`
    at half input resolution.
    """

    def __init__(self, net, dimension):
        super().__init__()
        self.net = net
        if dimension == 2:
            self.avg_pool = F.avg_pool2d
            self.interpolate_mode = "bilinear"
        else:
            self.avg_pool = F.avg_pool3d
            self.interpolate_mode = "trilinear"
        self.dimension = dimension
        # This member variable is read by assign_identity_map when
        # walking the network tree and assigning identity_maps
        # to know that all children of this module operate at a lower
        # resolution.
        self.downscale_factor = 2

    def forward(self, image_A, image_B):

        image_A = self.avg_pool(image_A, 2, ceil_mode=True)
        image_B = self.avg_pool(image_B, 2, ceil_mode=True)
        return self.net(image_A, image_B)


### DEPRECATED
def warninfo(message):
    from inspect import getframeinfo, stack
    import warnings

    caller = getframeinfo(stack()[2][0])
    warnings.warn("%s:%d - %s" % (caller.filename, caller.lineno, message))


def assignIdentityMap(net, size):
    warninfo("assignIdentityMap is deprecated. use net.assign_identity_map")
    net.assign_identity_map(size)


def adjust_batch_size(net, N):
    warninfo(
        "adjust_batch_size is deprecated. Batch size is now determined at runtime from input shape"
    )


DoubleNet = TwoStepRegistration
DownsampleNet = DownsampleRegistration



def to_floats(stats):
    out = []
    for v in stats:
        if isinstance(v, torch.Tensor):
            v = torch.mean(v).cpu().item()
        out.append(v)
    return ICONLoss(*out)


ICONLoss = namedtuple(
    "ICONLoss",
    "all_loss inverse_consistency_loss similarity_loss transform_magnitude flips",
)


class InverseConsistentNet(RegistrationModule):
    def __init__(self, network, similarity, lmbda):

        super().__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity

    def __call__(self, image_A, image_B) -> ICONLoss:
        return super().__call__(image_A, image_B)

    def forward(self, image_A, image_B):

        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)

        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map)

        if getattr(self.similarity, "isInterpolated", False):
            # tag images during warping so that the similarity measure
            # can use information about whether a sample is interpolated
            # or extrapolated
            inbounds_tag = torch.zeros([image_A.shape[0]] + [1] + list(image_A.shape[2:]), device=image_A.device)
            if len(self.input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(self.input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0
        else:
            inbounds_tag = None

        self.warped_image_A = compute_warped_image_multiNC(
            torch.cat([image_A, inbounds_tag], axis=1) if inbounds_tag is not None else image_A,
            self.phi_AB_vectorfield,
            self.spacing,
            1,
        )
        self.warped_image_B = compute_warped_image_multiNC(
            torch.cat([image_B, inbounds_tag], axis=1) if inbounds_tag is not None else image_B,
            self.phi_BA_vectorfield,
            self.spacing,
            1,
        )

        similarity_loss = self.similarity(
            self.warped_image_A, image_B
        ) + self.similarity(self.warped_image_B, image_A)

        Iepsilon = (
            self.identity_map
            + torch.randn(*self.identity_map.shape).to(image_A.device)
            * 1
            / self.identity_map.shape[-1]
        )

        # inverse consistency one way

        approximate_Iepsilon1 = self.phi_AB(self.phi_BA(Iepsilon))

        approximate_Iepsilon2 = self.phi_BA(self.phi_AB(Iepsilon))

        inverse_consistency_loss = torch.mean(
            (Iepsilon - approximate_Iepsilon1) ** 2
        ) + torch.mean((Iepsilon - approximate_Iepsilon2) ** 2)

        transform_magnitude = torch.mean(
            (self.identity_map - self.phi_AB_vectorfield) ** 2
        )

        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss

        return ICONLoss(
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            flips(self.phi_BA_vectorfield),
        )


class GradientICON(RegistrationModule):
    def __init__(self, network, similarity, lmbda):

        super().__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity

    def compute_gradient_icon_loss(self, phi_AB, phi_BA):
        Iepsilon = (
            self.identity_map
            + torch.randn(*self.identity_map.shape).to(self.identity_map.device)
            * 1
            / self.identity_map.shape[-1]
        )

        # compute squared Frobenius of Jacobian of icon error

        direction_losses = []

        approximate_Iepsilon = phi_AB(phi_BA(Iepsilon))

        inverse_consistency_error = Iepsilon - approximate_Iepsilon

        delta = 0.001

        if len(self.identity_map.shape) == 4:
            dx = torch.Tensor([[[[delta]], [[0.0]]]]).to(self.identity_map.device)
            dy = torch.Tensor([[[[0.0]], [[delta]]]]).to(self.identity_map.device)
            direction_vectors = (dx, dy)

        elif len(self.identity_map.shape) == 5:
            dx = torch.Tensor([[[[[delta]]], [[[0.0]]], [[[0.0]]]]]).to(
                self.identity_map.device
            )
            dy = torch.Tensor([[[[[0.0]]], [[[delta]]], [[[0.0]]]]]).to(
                self.identity_map.device
            )
            dz = torch.Tensor([[[[0.0]]], [[[0.0]]], [[[delta]]]]).to(
                self.identity_map.device
            )
            direction_vectors = (dx, dy, dz)
        elif len(self.identity_map.shape) == 3:
            dx = torch.Tensor([[[delta]]]).to(self.identity_map.device)
            direction_vectors = (dx,)

        for d in direction_vectors:
            approximate_Iepsilon_d = phi_AB(phi_BA(Iepsilon + d))
            inverse_consistency_error_d = Iepsilon + d - approximate_Iepsilon_d
            grad_d_icon_error = (
                inverse_consistency_error - inverse_consistency_error_d
            ) / delta
            direction_losses.append(torch.mean(grad_d_icon_error**2))

        inverse_consistency_loss = sum(direction_losses)

        return inverse_consistency_loss

    def compute_similarity_measure(self, phi_AB, phi_BA, image_A, image_B):
        self.phi_AB_vectorfield = phi_AB(self.identity_map)
        self.phi_BA_vectorfield = phi_BA(self.identity_map)

        if getattr(self.similarity, "isInterpolated", False):
            # tag images during warping so that the similarity measure
            # can use information about whether a sample is interpolated
            # or extrapolated
            inbounds_tag = torch.zeros([image_A.shape[0]] + [1] + list(image_A.shape[2:]), device=image_A.device)
            if len(self.input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(self.input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0
        else:
            inbounds_tag = None

        self.warped_image_A = self.as_function(
            torch.cat([image_A, inbounds_tag], axis=1) if inbounds_tag is not None else image_A
        )(self.phi_AB_vectorfield)
        self.warped_image_B = self.as_function(
            torch.cat([image_B, inbounds_tag], axis=1) if inbounds_tag is not None else image_B
        )(self.phi_BA_vectorfield)
        similarity_loss = self.similarity(
            self.warped_image_A, image_B
        ) + self.similarity(self.warped_image_B, image_A)
        return similarity_loss

    def forward(self, image_A, image_B) -> ICONLoss:

        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)

        similarity_loss = self.compute_similarity_measure(
            self.phi_AB, self.phi_BA, image_A, image_B
        )

        inverse_consistency_loss = self.compute_gradient_icon_loss(
            self.phi_AB, self.phi_BA
        )

        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss

        transform_magnitude = torch.mean(
            (self.identity_map - self.phi_AB_vectorfield) ** 2
        )
        return ICONLoss(
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            flips(self.phi_BA_vectorfield),
        )
    

class GradientICONSparse(RegistrationModule):
    def __init__(self, network, similarity, lmbda):

        super().__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity

    def forward(self, image_A, image_B):

        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)

        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map)

        # tag images during warping so that the similarity measure
        # can use information about whether a sample is interpolated
        # or extrapolated

        if getattr(self.similarity, "isInterpolated", False):
            # tag images during warping so that the similarity measure
            # can use information about whether a sample is interpolated
            # or extrapolated
            inbounds_tag = torch.zeros([image_A.shape[0]] + [1] + list(image_A.shape[2:]), device=image_A.device)
            if len(self.input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(self.input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0
        else:
            inbounds_tag = None

        self.warped_image_A = compute_warped_image_multiNC(
            torch.cat([image_A, inbounds_tag], axis=1) if inbounds_tag is not None else image_A,
            self.phi_AB_vectorfield,
            self.spacing,
            1,
        )
        self.warped_image_B = compute_warped_image_multiNC(
            torch.cat([image_B, inbounds_tag], axis=1) if inbounds_tag is not None else image_B,
            self.phi_BA_vectorfield,
            self.spacing,
            1,
        )

        similarity_loss = self.similarity(
            self.warped_image_A, image_B
        ) + self.similarity(self.warped_image_B, image_A)

        if len(self.input_shape) - 2 == 3:
            Iepsilon = (
                self.identity_map
                + 2 * torch.randn(*self.identity_map.shape).to(image_A.device)
                / self.identity_map.shape[-1]
            )[:, :, ::2, ::2, ::2]
        elif len(self.input_shape) - 2 == 2:
            Iepsilon = (
                self.identity_map
                + 2 * torch.randn(*self.identity_map.shape).to(image_A.device)
                / self.identity_map.shape[-1]
            )[:, :, ::2, ::2]

        # compute squared Frobenius of Jacobian of icon error

        direction_losses = []

        approximate_Iepsilon = self.phi_AB(self.phi_BA(Iepsilon))

        inverse_consistency_error = Iepsilon - approximate_Iepsilon

        delta = 0.001

        if len(self.identity_map.shape) == 4:
            dx = torch.Tensor([[[[delta]], [[0.0]]]]).to(image_A.device)
            dy = torch.Tensor([[[[0.0]], [[delta]]]]).to(image_A.device)
            direction_vectors = (dx, dy)

        elif len(self.identity_map.shape) == 5:
            dx = torch.Tensor([[[[[delta]]], [[[0.0]]], [[[0.0]]]]]).to(image_A.device)
            dy = torch.Tensor([[[[[0.0]]], [[[delta]]], [[[0.0]]]]]).to(image_A.device)
            dz = torch.Tensor([[[[0.0]]], [[[0.0]]], [[[delta]]]]).to(image_A.device)
            direction_vectors = (dx, dy, dz)
        elif len(self.identity_map.shape) == 3:
            dx = torch.Tensor([[[delta]]]).to(image_A.device)
            direction_vectors = (dx,)

        for d in direction_vectors:
            approximate_Iepsilon_d = self.phi_AB(self.phi_BA(Iepsilon + d))
            inverse_consistency_error_d = Iepsilon + d - approximate_Iepsilon_d
            grad_d_icon_error = (
                inverse_consistency_error - inverse_consistency_error_d
            ) / delta
            direction_losses.append(torch.mean(grad_d_icon_error**2))

        inverse_consistency_loss = sum(direction_losses)

        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss

        transform_magnitude = torch.mean(
            (self.identity_map - self.phi_AB_vectorfield) ** 2
        )
        return ICONLoss(
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            flips(self.phi_BA_vectorfield),
        )
    
BendingLoss = namedtuple(
    "BendingLoss",
    "all_loss bending_energy_loss similarity_loss transform_magnitude flips",
)
    
class BendingEnergyNet(RegistrationModule):
    def __init__(self, network, similarity, lmbda):
        super().__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity

    def compute_bending_energy_loss(self, phi_AB_vectorfield):
        # dxdx = [f[x+h, y] + f[x-h, y] - 2 * f[x, y]]/(h**2)
        # dxdy = [f[x+h, y+h] + f[x-h, y-h] - f[x+h, y-h] - f[x-h, y+h]]/(4*h**2)
        # BE_2d = |dxdx| + |dydy| + 2 * |dxdy|
        # psudo code: BE_2d = [torch.mean(dxdx**2) + torch.mean(dydy**2) + 2 * torch.mean(dxdy**2)]/4.0  
        # BE_3d = |dxdx| + |dydy| + |dzdz| + 2 * |dxdy| + 2 * |dydz| + 2 * |dxdz|
        
        if len(self.identity_map.shape) == 3:
            dxdx = (phi_AB_vectorfield[:, :, 2:] 
                - 2*phi_AB_vectorfield[:, :, 1:-1]
                + phi_AB_vectorfield[:, :, :-2]) / self.spacing[0]**2
            bending_energy = torch.mean((dxdx)**2)
            
        elif len(self.identity_map.shape) == 4:
            dxdx = (phi_AB_vectorfield[:, :, 2:] 
                - 2*phi_AB_vectorfield[:, :, 1:-1]
                + phi_AB_vectorfield[:, :, :-2]) / self.spacing[0]**2
            dydy = (phi_AB_vectorfield[:, :, :, 2:] 
                - 2*phi_AB_vectorfield[:, :, :, 1:-1]
                + phi_AB_vectorfield[:, :, :, :-2]) / self.spacing[1]**2
            dxdy = (phi_AB_vectorfield[:, :, 2:, 2:] 
                + phi_AB_vectorfield[:, :, :-2, :-2] 
                - phi_AB_vectorfield[:, :, 2:, :-2]
                - phi_AB_vectorfield[:, :, :-2, 2:]) / (4.0*self.spacing[0]*self.spacing[1])
            bending_energy = (torch.mean(dxdx**2) + torch.mean(dydy**2) + 2*torch.mean(dxdy**2)) / 4.0
        elif len(self.identity_map.shape) == 5:
            dxdx = (phi_AB_vectorfield[:, :, 2:] 
                - 2*phi_AB_vectorfield[:, :, 1:-1]
                + phi_AB_vectorfield[:, :, :-2]) / self.spacing[0]**2
            dydy = (phi_AB_vectorfield[:, :, :, 2:] 
                - 2*phi_AB_vectorfield[:, :, :, 1:-1]
                + phi_AB_vectorfield[:, :, :, :-2]) / self.spacing[1]**2
            dzdz = (phi_AB_vectorfield[:, :, :, :, 2:] 
                - 2*phi_AB_vectorfield[:, :, :, :, 1:-1]
                + phi_AB_vectorfield[:, :, :, :, :-2]) / self.spacing[2]**2
            dxdy = (phi_AB_vectorfield[:, :, 2:, 2:] 
                + phi_AB_vectorfield[:, :, :-2, :-2] 
                - phi_AB_vectorfield[:, :, 2:, :-2]
                - phi_AB_vectorfield[:, :, :-2, 2:]) / (4.0*self.spacing[0]*self.spacing[1])
            dydz = (phi_AB_vectorfield[:, :, :, 2:, 2:] 
                + phi_AB_vectorfield[:, :, :, :-2, :-2] 
                - phi_AB_vectorfield[:, :, :, 2:, :-2]
                - phi_AB_vectorfield[:, :, :, :-2, 2:]) / (4.0*self.spacing[1]*self.spacing[2])
            dxdz = (phi_AB_vectorfield[:, :, 2:, :, 2:] 
                + phi_AB_vectorfield[:, :, :-2, :, :-2] 
                - phi_AB_vectorfield[:, :, 2:, :, :-2]
                - phi_AB_vectorfield[:, :, :-2, :, 2:]) / (4.0*self.spacing[0]*self.spacing[2])

            bending_energy = ((dxdx**2).mean() + (dydy**2).mean() + (dzdz**2).mean() 
                    + 2.*(dxdy**2).mean() + 2.*(dydz**2).mean() + 2.*(dxdz**2).mean()) / 9.0
        

        return bending_energy

    def compute_similarity_measure(self, phi_AB_vectorfield, image_A, image_B):

        if getattr(self.similarity, "isInterpolated", False):
            # tag images during warping so that the similarity measure
            # can use information about whether a sample is interpolated
            # or extrapolated
            inbounds_tag = torch.zeros([image_A.shape[0]] + [1] + list(image_A.shape[2:]), device=image_A.device)
            if len(self.input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(self.input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0
        else:
            inbounds_tag = None

        self.warped_image_A = self.as_function(
            torch.cat([image_A, inbounds_tag], axis=1) if inbounds_tag is not None else image_A
        )(phi_AB_vectorfield)
        
        similarity_loss = self.similarity(
            self.warped_image_A, image_B
        )
        return similarity_loss

    def forward(self, image_A, image_B) -> ICONLoss:

        #assert self.identity_map.shape[2:] == image_A.shape[2:]
        #assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        
        similarity_loss = self.compute_similarity_measure(
            self.phi_AB_vectorfield, image_A, image_B
        )

        bending_energy_loss = self.compute_bending_energy_loss(
            self.phi_AB_vectorfield
        )

        all_loss = self.lmbda * bending_energy_loss + similarity_loss

        transform_magnitude = torch.mean(
            (self.identity_map - self.phi_AB_vectorfield) ** 2
        )
        return BendingLoss(
            all_loss,
            bending_energy_loss,
            similarity_loss,
            transform_magnitude,
            flips(self.phi_AB_vectorfield),
        )

    def prepare_for_viz(self, image_A, image_B):
        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        self.phi_BA = self.regis_net(image_B, image_A)
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map)

        self.warped_image_A = self.as_function(image_A)(self.phi_AB_vectorfield)
        self.warped_image_B = self.as_function(image_B)(self.phi_BA_vectorfield)

class DiffusionRegularizedNet(BendingEnergyNet):
    def compute_bending_energy_loss(self, phi_AB_vectorfield):
        phi_AB_vectorfield = self.identity_map - phi_AB_vectorfield
        if len(self.identity_map.shape) == 3:
            bending_energy = torch.mean((
                - phi_AB_vectorfield[:, :, 1:]
                + phi_AB_vectorfield[:, :, 1:-1]
            )**2)

        elif len(self.identity_map.shape) == 4:
            bending_energy = torch.mean((
                - phi_AB_vectorfield[:, :, 1:]
                + phi_AB_vectorfield[:, :, :-1]
            )**2) + torch.mean((
                - phi_AB_vectorfield[:, :, :, 1:]
                + phi_AB_vectorfield[:, :, :, :-1]
            )**2)
        elif len(self.identity_map.shape) == 5:
            bending_energy = torch.mean((
                - phi_AB_vectorfield[:, :, 1:]
                + phi_AB_vectorfield[:, :, :-1]
            )**2) + torch.mean((
                - phi_AB_vectorfield[:, :, :, 1:]
                + phi_AB_vectorfield[:, :, :, :-1]
            )**2) + torch.mean((
                - phi_AB_vectorfield[:, :, :, :, 1:]
                + phi_AB_vectorfield[:, :, :, :, :-1]
            )**2)


        return bending_energy * self.identity_map.shape[2] **2


def flips(phi, in_percentage=False):
    if len(phi.size()) == 5:
        a = (phi[:, :, 1:, 1:, 1:] - phi[:, :, :-1, 1:, 1:]).detach()
        b = (phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, :-1, 1:]).detach()
        c = (phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, 1:, :-1]).detach()

        dV = torch.sum(torch.cross(a, b, 1) * c, axis=1, keepdims=True)
        if in_percentage:
            return torch.mean((dV < 0).float()) * 100.
        else:
            return torch.sum(dV < 0) / phi.shape[0]
    elif len(phi.size()) == 4:
        du = (phi[:, :, 1:, :-1] - phi[:, :, :-1, :-1]).detach()
        dv = (phi[:, :, :-1, 1:] - phi[:, :, :-1, :-1]).detach()
        dA = du[:, 0] * dv[:, 1] - du[:, 1] * dv[:, 0]
        if in_percentage:
            return torch.mean((dA < 0).float()) * 100.
        else:
            return torch.sum(dA < 0) / phi.shape[0]
    elif len(phi.size()) == 3:
        du = (phi[:, :, 1:] - phi[:, :, :-1]).detach()
        if in_percentage:
            return torch.mean((du < 0).float()) * 100.
        else:
            return torch.sum(du < 0) / phi.shape[0]
    else:
        raise ValueError()


def register_pair(
    model, image_A, image_B, finetune_steps=None, return_artifacts=False, device='cuda'
) -> "(itk.CompositeTransform, itk.CompositeTransform)":

    assert isinstance(image_A, itk.Image)
    assert isinstance(image_B, itk.Image)

    # send model to cpu or gpu depending on config- auto detects capability
    model.to(device)

    A_npy = np.array(image_A)
    B_npy = np.array(image_B)

    assert(np.max(A_npy) != np.min(A_npy))
    assert(np.max(B_npy) != np.min(B_npy))
    # turn images into torch Tensors: add feature and batch dimensions (each of length 1)
    A_trch = torch.Tensor(A_npy).to(device)[None, None]
    B_trch = torch.Tensor(B_npy).to(device)[None, None]

    shape = model.identity_map.shape

    # Here we resize the input images to the shape expected by the neural network. This affects the
    # pixel stride as well as the magnitude of the displacement vectors of the resulting
    # displacement field, which create_itk_transform will have to compensate for.
    A_resized = F.interpolate(
        A_trch, size=shape[2:], mode="trilinear", align_corners=False
    )
    B_resized = F.interpolate(
        B_trch, size=shape[2:], mode="trilinear", align_corners=False
    )
    if finetune_steps == 0:
        raise Exception("To indicate no finetune_steps, pass finetune_steps=None")

    if finetune_steps == None:
        with torch.no_grad():
            loss = model(A_resized, B_resized)
    else:
        loss = finetune_execute(model, A_resized, B_resized, finetune_steps)

    # phi_AB and phi_BA are [1, 3, H, W, D] pytorch tensors representing the forward and backward
    # maps computed by the model
    if hasattr(model, "prepare_for_viz"):
        with torch.no_grad():
            model.prepare_for_viz(A_resized, B_resized)
    phi_AB = model.phi_AB(model.identity_map)

    # the parameters ident, image_A, and image_B are used for their metadata
    itk_transforms = (
        create_itk_transform(phi_AB, model.identity_map, image_A, image_B),
    )
    if not return_artifacts:
        return itk_transforms
    else:
        return itk_transforms + (to_floats(loss),)
    
    
    
def create_itk_transform(phi, ident, image_A, image_B) -> "itk.CompositeTransform":

    # itk.DeformationFieldTransform expects a displacement field, so we subtract off the identity map.
    disp = (phi - ident)[0].cpu()

    network_shape_list = list(ident.shape[2:])

    dimension = len(network_shape_list)

    tr = itk.DisplacementFieldTransform[(itk.D, dimension)].New()

    # We convert the displacement field into an itk Vector Image.
    scale = torch.Tensor(network_shape_list)

    for _ in network_shape_list:
        scale = scale[:, None]
    disp *= scale - 1

    # disp is a shape [3, H, W, D] tensor with vector components in the order [vi, vj, vk]
    disp_itk_format = (
        disp.double()
        .numpy()[list(reversed(range(dimension)))]
        .transpose(list(range(1, dimension + 1)) + [0])
    )
    # disp_itk_format is a shape [H, W, D, 3] array with vector components in the order [vk, vj, vi]
    # as expected by itk.

    itk_disp_field = itk.image_from_array(disp_itk_format, is_vector=True)

    tr.SetDisplacementField(itk_disp_field)

    to_network_space = resampling_transform(image_A, list(reversed(network_shape_list)))

    from_network_space = resampling_transform(
        image_B, list(reversed(network_shape_list))
    ).GetInverseTransform()

    phi_AB_itk = itk.CompositeTransform[itk.D, dimension].New()

    phi_AB_itk.PrependTransform(from_network_space)
    phi_AB_itk.PrependTransform(tr)
    phi_AB_itk.PrependTransform(to_network_space)

    # warp(image_A, phi_AB_itk) is close to image_B

    return phi_AB_itk


def resampling_transform(image, shape):

    imageType = itk.template(image)[0][itk.template(image)[1]]

    dummy_image = itk.image_from_array(
        np.zeros(tuple(reversed(shape)), dtype=itk.array_from_image(image).dtype)
    )
    if len(shape) == 2:
        transformType = itk.MatrixOffsetTransformBase[itk.D, 2, 2]
    else:
        transformType = itk.VersorRigid3DTransform[itk.D]
    initType = itk.CenteredTransformInitializer[transformType, imageType, imageType]
    initializer = initType.New()
    initializer.SetFixedImage(dummy_image)
    initializer.SetMovingImage(image)
    transform = transformType.New()

    initializer.SetTransform(transform)
    initializer.InitializeTransform()

    if len(shape) == 3:
        transformType = itk.CenteredAffineTransform[itk.D, 3]
        t2 = transformType.New()
        t2.SetCenter(transform.GetCenter())
        t2.SetOffset(transform.GetOffset())
        transform = t2
    m = transform.GetMatrix()
    m_a = itk.array_from_matrix(m)

    input_shape = image.GetLargestPossibleRegion().GetSize()

    for i in range(len(shape)):

        m_a[i, i] = image.GetSpacing()[i] * (input_shape[i] / shape[i])

    m_a = itk.array_from_matrix(image.GetDirection()) @ m_a

    transform.SetMatrix(itk.matrix_from_array(m_a))

    return transform



def finetune_execute(model, image_A, image_B, steps):
    state_dict = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00002)
    for _ in range(steps):
        optimizer.zero_grad()
        loss_tuple = model(image_A, image_B)
        print(loss_tuple)
        loss_tuple[0].backward()
        optimizer.step()
    with torch.no_grad():
        loss = model(image_A, image_B)
    model.load_state_dict(state_dict)
    return loss
