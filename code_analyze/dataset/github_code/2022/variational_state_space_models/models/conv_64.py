import haiku as hk
import jax.numpy as jnp
import jax
from .batchnorm import MultiBatchNorm

class Reduce(hk.Module):
    def __init__(self, nout):
        super().__init__()
        self.nout = nout

    def __call__(self, x, is_training, cross_axes):
        c = hk.Conv2D(
            output_channels=self.nout,
            kernel_shape=4,
            stride=2,
            padding=((1, 1), (1, 1))
        )(x)
        bn = MultiBatchNorm(True, True, 0.9)(c, is_training, cross_axes)
        #bn = hk.LayerNorm(-1, True, True)(c)
        return jax.nn.leaky_relu(bn, 0.3)

class Expand(hk.Module):
    def __init__(self, oc):
        super().__init__()
        self.oc = oc

    def __call__(self, input, is_training, cross_axes):
        c = hk.Conv2DTranspose(
            output_channels=self.oc, 
            kernel_shape=4,
            stride=2,
            padding=((2, 2), (2, 2))
        )(input)
        bn = MultiBatchNorm(True, True, 0.9)(c, is_training, cross_axes)
        return jax.nn.leaky_relu(bn, 0.3)

def conv_64_encoder(input, odim, is_training, cross_axes=[], nf=32):
    # nc x 64 x 64
    h1 = Reduce(nf)(input, is_training, cross_axes)
    # nf x 32 x 32
    h2 = Reduce(nf*2)(h1, is_training, cross_axes)
    # 2*nf x 16 x 16
    h3 = Reduce(nf*4)(h2, is_training, cross_axes)
    # 4*nf x 8 x 8
    h4 = Reduce(nf*8)(h3, is_training, cross_axes)
    # 8*nf x 4 x 4
    h5 = hk.Conv2D(output_channels=odim, kernel_shape=4,
        stride=1, padding=((0, 0), (0,0)))(h4)
    h5 = MultiBatchNorm(True, True, 0.9)(h5, is_training, cross_axes)
    # print(input.shape)
    # print(h1.shape)
    # print(h2.shape)
    # print(h3.shape)
    # print(h4.shape)
    # print(h5.shape)
    return h5.flatten()

def conv_64_decoder(input, nc, is_training, cross_axes=[], nf=32):
    i = jnp.reshape(input, input.shape[:-1] + (1, 1) + (input.shape[-1],))
    d1 = hk.Conv2DTranspose(
        output_channels=nf*8,
        kernel_shape=4,
        stride=1,
        padding=((3, 3), (3, 3))
    )(i)
    #d1 = hk.BatchNorm(True, True,0.9, cross_replica_axis='batch')(d1, is_training)
    d1 = MultiBatchNorm(True, True, 0.9)(d1, is_training, cross_axes)
    d1 = jax.nn.leaky_relu(d1, 0.3)
    d2 = Expand(nf*4)(d1, is_training, cross_axes)
    d3 = Expand(nf*2)(d2, is_training, cross_axes)
    d4 = Expand(nf)(d3, is_training, cross_axes)
    d5 = hk.Conv2DTranspose(
        output_channels=nc,
        kernel_shape=4,
        stride=2,
        padding=((2, 2), (2, 2))
    )(d4)
    # print(i.shape)
    # print(d1.shape)
    # print(d2.shape)
    # print(d3.shape)
    # print(d4.shape)
    # print(d5.shape)
    # 64 x 64 x nc
    return d5