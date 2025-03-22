import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
# from common.instance_norm import InstanceNormalization
from chainer import link_hooks, Chain
from chainermn.links import MultiNodeBatchNormalization
# from SSD_for_vehicle_detection import SA_module

class GRL(chainer.function_node.FunctionNode):
    def forward(self, inputs):
        x, = inputs
        return x,

    def backward(self, inputs, grad_outputs):
        gy, = grad_outputs
        return - gy,

def gradient_reversal_layer(x):
    return GRL().apply((x,))

class init_BN(object):
    def __init__(self,comm=None):
        self.comm = comm

    def __call__(self,ch):
        if self.comm:
            return MultiNodeBatchNormalization(ch,comm=self.comm)
        else:
            return L.BatchNormalization(ch)

class SA_module(Chain):
    def __init__(self, out_channel, SN = True, f_g_ch_div =8, f_g_ch_specify = None, h_ch_div = 2, h_ch_specify = None, f_h_pool_size=2,g_pool_size= 1):
        super(SA_module, self).__init__()
        # self.original = original
        self.out_channel = out_channel
        # if self.original:
        if f_g_ch_specify:
            self.f_g_channel = f_g_ch_specify
        else:
            self.f_g_channel = self.out_channel // f_g_ch_div
        if h_ch_specify:
            self.h_channel = h_ch_specify
        else:
            self.h_channel = self.out_channel // h_ch_div
        self.f_h_pool_size = f_h_pool_size
        self.g_pool_size = g_pool_size
        # else:
        #     self.f_g_channel = self.out_channel
        #     self.h_channel = self.out_channel
        with self.init_scope():
            self.f_conv = L.Convolution2D(self.f_g_channel, 1)
            self.g_conv = L.Convolution2D(self.f_g_channel, 1)
            self.h_conv = L.Convolution2D(self.h_channel, 1)
            self.out_conv = L.Convolution2D(self.out_channel, 1)
            self.sigma = chainer.Parameter(0)
        self.sigma.initialize(())
        if SN:
            for key in self.__dict__.keys():
                if type(self[key]) in [L.Convolution2D]:
                    self[key].add_hook(link_hooks.SpectralNormalization())

    def __call__(self, x, attn_only=False):
        batchsize, c, h, w = x.shape
        # location_num = h * w
        # downsampled_num = location_num // 4

        f_x = self.f_conv(x)
        # if self.original:
        if self.f_h_pool_size > 1:
            f_x = F.max_pooling_2d(f_x, self.f_h_pool_size, self.f_h_pool_size)
        f_x = f_x.reshape((batchsize,self.f_g_channel,-1)).transpose((0,2,1))

        g_x = self.g_conv(x)
        if self.g_pool_size > 1:
            g_x = F.max_pooling_2d(g_x, self.g_pool_size, self.g_pool_size)
        g_h, g_w = g_x.shape[2:]
        g_x = g_x.reshape((batchsize, self.f_g_channel, -1)).transpose((0, 2, 1))

        attn = F.matmul(g_x,f_x,transb=True)
        attn = F.softmax(attn,axis=-1)

        h_x = self.h_conv(x)
        if self.f_h_pool_size>1:
            h_x = F.max_pooling_2d(h_x,self.f_h_pool_size,self.f_h_pool_size)
        h_x = h_x.reshape((batchsize,self.h_channel,-1)).transpose((0,2,1))

        attn = F.matmul(attn, h_x).reshape((batchsize,g_h,g_w,self.h_channel)).transpose((0,3,1,2))
        if self.g_pool_size > 1:
            attn = F.unpooling_2d(attn, self.g_pool_size, self.g_pool_size)
        attn = self.out_conv(attn)

        if attn_only:
            return attn * self.sigma

        return x + attn * self.sigma

def reflectPad(x, pad):
    if pad < 0:
        print("Pad width has to be 0 or larger")
        raise ValueError
    if pad == 0:
        return x
    else:
        width, height = x.shape[2:]
        w_pad = h_pad = pad
        if width == 1:
            x = F.concat((x,)*(1+pad*2),axis=2)
        else:
            while w_pad > 0:
                pad = min(w_pad, width-1)
                w_pad -= pad
                x = _pad_along_axis(x, pad, 2)
                width, height = x.shape[2:]
        if height == 1:
            x = F.concat((x,)*(1+pad*2),axis=3)
        else:
            while h_pad > 0:
                pad = min(h_pad, height-1)
                h_pad -= pad
                x = _pad_along_axis(x, pad, 3)
                width, height = x.shape[2:]
        return x

def _pad_along_axis(x, pad, axis):
    dim = x.ndim
    head = F.get_item(x,(slice(None),) * axis + (slice(1, 1 + pad),) + (slice(None),)*(dim-1-axis))
    head = F.concat(
        [F.get_item(head, (slice(None),) * axis + (slice(i, i + 1),) + (slice(None),)*(dim-1-axis)) for i in range(pad)][::-1], axis=axis)
    tail = F.get_item(x, (slice(None),) * axis + (slice(-1-pad, -1),) + (slice(None),)*(dim-1-axis))
    tail = F.concat(
        [F.get_item(tail, (slice(None),) * axis + (slice(i, i + 1),) + (slice(None),)*(dim-1-axis)) for i in range(pad)][::-1],
        axis=axis)
    x = F.concat((head, x, tail), axis=axis)
    return x

class ResBlock(chainer.Chain):
    def __init__(self, ch, norm='bn', activation=F.relu, k_size=3, w_init=None, reflect = 0, norm_learnable = True, normalize_grad = False,
                 comm = None):
        if w_init == None:
            w = chainer.initializers.Normal(0.02)
        else:
            w = w_init
        if norm in ['instance','bn']:
            use_norm = True
        else:
            use_norm = False
        self.use_norm = use_norm
        self.activation = activation
        layers = {}

        self.reflect = reflect

        if self.reflect in [1,2]:
            pad = 0
        else:
            pad = k_size//2

        BN = init_BN(comm)

        layers['c0'] = L.Convolution2D(ch, ch, 3, 1, pad, initialW=w)
        layers['c1'] = L.Convolution2D(ch, ch, 3, 1, pad, initialW=w)

        if self.use_norm:
            if norm == 'instance':
                pass
                # layers['norm0'] = InstanceNormalization(ch, use_gamma=norm_learnable, use_beta=norm_learnable, norm_grad=normalize_grad)
                # layers['norm1'] = InstanceNormalization(ch, use_gamma=norm_learnable, use_beta=norm_learnable, norm_grad=normalize_grad)
            elif norm == 'bn':
                layers['norm0'] = BN(ch)
                # layers['norm1'] = L.BatchNormalization(ch, use_gamma=norm_learnable, use_beta=norm_learnable)

        super(ResBlock, self).__init__(**layers)
        self.register_persistent('reflect')
        self.register_persistent('use_norm')
        self.register_persistent('activation')

    # override serialize() to support serializing function object
    def serialize(self, serializer):
        """Serializes the link object.

        Args:
            serializer (~chainer.AbstractSerializer): Serializer object.

        """
        d = self.__dict__
        for name in self._params:
            param = d[name]
            data = serializer(name, param.data)
            if param.data is None and data is not None:
                # Initialize the parameter here
                param.initialize(data.shape)
                if isinstance(param.data, np.ndarray):
                    np.copyto(param.data, data)
                else:
                    param.data.set(np.asarray(data))
        for name in self._persistent:
            if isinstance(serializer,chainer.serializer.Deserializer) and name == "activation":
                d[name] = None
            d[name] = serializer(name, d[name])
            if isinstance(serializer, chainer.serializer.Deserializer) and name == "activation":
                if isinstance(d[name],np.ndarray):
                    d[name] = d[name][()]
        d = self.__dict__
        for name in self._children:
            d[name].serialize(serializer[name])

    def __call__(self, x):
        if self.reflect == 0:
            _pad = self.c0.W.shape[2] // 2
            self.c0.pad = (_pad, _pad)
            self.c1.pad = (_pad, _pad)
        else:
            self.c0.pad = (0,0)
            self.c1.pad = (0,0)
        if self.reflect == 2:
            # h = F.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='reflect') << 'reflect' is not currently supported(v3)
            h = reflectPad(x, 1)
            h = self.c0(h)
        else:
            h = self.c0(x)
        if self.use_norm:
            h = self.norm0(h)
        h = self.activation(h)
        if self.reflect == 2:
            # h = F.pad(h, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='reflect')  << 'reflect' is not currently supported(v3)
            h = reflectPad(h, 1)
        h = self.c1(h)
        # if self.use_norm:
        #     h = self.norm1(h)
        if self.reflect == 1:
            x = F.get_item(x,(slice(None),slice(None),slice(2,-2),slice(2,-2)))
        return h + x

class CNABlock(chainer.Chain):
    def __init__(self, ch0, ch1, \
                nn='conv', \
                norm=None, \
                activation=F.relu, \
                w_init=None, \
                k_size = 3, \
                pad = None, \
                norm_learnable = True,\
                normalize_grad = False,
                SN = False,
                Attention = False,
                attn_args = None,
                 comm=None):

        self.norm = norm
        self.activation = activation
        self.nn = nn
        self.norm_learnable = norm_learnable
        self.normalize_grad = normalize_grad
        self.Attention = Attention
        layers = {}
        BN = init_BN(comm)

        if w_init == None:
            w = chainer.initializers.Normal(0.02)
        else:
            w = w_init

        if nn == 'down_conv':
            layers['c'] = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)

        elif nn == 'down_conv_2':
            layers['c'] = L.Convolution2D(ch0, ch1, 4, 1, 1, initialW=w)

        elif nn == 'g_down_conv':
            layers['c'] = L.Convolution2D(ch0, ch1, 3, 2, 1, initialW=w)

        elif nn=='conv':
            if pad == None:
                pad = k_size//2
            layers['c'] = L.Convolution2D(ch0, ch1, k_size, 1, pad, initialW=w)

        elif nn == 'deconvolution':
            layers['c'] = L.Deconvolution2D(ch0, ch1, 3, 2, 1, initialW=w)

        else:
            raise NotImplementedError("Cannot find method {0} in {1}".format(nn,self.__class__.__name__))

        if self.Attention:
            layers['attn'] = SA_module(ch1,SN,**attn_args)

        if self.norm == 'bn':
            layers['n'] = BN(ch1)
        # elif self.norm == 'instance':
        #     layers['n'] = InstanceNormalization(ch1, use_gamma=self.norm_learnable, use_beta=self.norm_learnable,\
        #                                         norm_grad = self.normalize_grad)

        if SN:
            layers['c'] = layers['c'].add_hook(link_hooks.SpectralNormalization())

        super(CNABlock, self).__init__(**layers)
        self.register_persistent('norm')
        # self.register_persistent('activation')
        self.register_persistent('nn')

    # override serialize() to support serializing function object
    def serialize(self, serializer):
        """Serializes the link object.

        Args:
            serializer (~chainer.AbstractSerializer): Serializer object.

        """
        d = self.__dict__
        for name in self._params:
            param = d[name]
            data = serializer(name, param.data)
            if param.data is None and data is not None:
                # Initialize the parameter here
                param.initialize(data.shape)
                if isinstance(param.data, np.ndarray):
                    np.copyto(param.data, data)
                else:
                    param.data.set(np.asarray(data))
        for name in self._persistent:
            if isinstance(serializer,chainer.serializer.Deserializer) and name == "activation":
                d[name] = None
            d[name] = serializer(name, d[name])
            if isinstance(serializer, chainer.serializer.Deserializer) and name == "activation":
                if isinstance(d[name],np.ndarray):
                    d[name] = d[name][()]
        d = self.__dict__
        for name in self._children:
            d[name].serialize(serializer[name])

    def __call__(self, x):
        if self.nn == 'deconvolution':
            x = F.pad(x, ((0, 0), (0, 0), (0, 1), (0, 1)), mode='constant')
        x = self.c(x)
        if self.nn == 'deconvolution':
            x = F.get_item(x, (slice(None), slice(None), slice(0, -1), slice(0, -1)))
        if self.Attention:
            x = self.attn(x)
        if self.norm != None:
            x = self.n(x)
        if self.activation != None:
            x = self.activation(x)
        return x
