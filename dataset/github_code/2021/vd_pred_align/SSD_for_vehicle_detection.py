# coding: utf-8
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv.links.model.ssd.ssd_vgg16 import VGG16Extractor512, VGG16Extractor300,_imagenet_mean #_check_pretrained_model, _load_npz,
from chainercv.links.model.ssd import Multibox
import chainer
from chainer import Chain, initializers
import chainer.links as L
import chainer.functions as F
from chainercv import utils
from chainer import Variable, link_hooks
from chainercv.links.model.ssd import Normalize
from chainercv import transforms
# from chainercv.links.model.ssd.multibox_coder import _unravel_index
from ops import reflectPad, SA_module, init_BN, CNABlock, ResBlock
from chainercv.links.model.ssd.multibox_loss import _hard_negative
from chainermn.links import MultiNodeBatchNormalization


defaultbox_size_300 = {
    0.15: (30, 48.0, 103.5, 159, 214.5, 270, 325.5),
    0.16: (30, 48.0, 103.5, 159, 214.5, 270, 325.5),
    0.3: (24, 30, 90, 150, 210, 270, 330),
}
defaultbox_size_512 = {
    0.15: (30.72, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72),
    0.16: (30.72, 46.08, 129.02, 211.97, 294.91, 377.87, 460.8, 543.74),
    0.3: (25.6, 30.72, 116.74, 202.75, 288.79, 374.78, 460.8, 546.82),
}  # defaultbox size corresponding to the image resolution

class Mbox_custom_SA(Multibox):
    def __init__(
            self, n_class, aspect_ratios,
            initialW=None, initial_bias=None):
        super(Mbox_custom_SA, self).__init__(n_class, aspect_ratios,
            initialW, initial_bias)
        with self.init_scope():
            self.c_reduce_SA = chainer.ChainList()
        ch_reduced = [32,64,32,16,16,16]
        for ch in ch_reduced:
            # n = (len(ar) + 1) * 2
            self.c_reduce_SA.add_link(SA_module(out_channel=ch,SN=False,f_h_pool_size=1,f_g_ch_specify=512//8,h_ch_specify=512//8))

    def forward(self, xs):
        xs_ = xs[:]
        for i in range(len(xs_)):
            xs_[i] = self.c_reduce_SA[i](xs_[i],attn_only=True)
        return super(Mbox_custom_SA, self).forward(xs_)

class Multibox_custom(chainer.Chain):
    """Multibox head of Single Shot Multibox Detector.

    This is a head part of Single Shot Multibox Detector [#]_.
    This link computes :obj:`mb_locs` and :obj:`mb_confs` from feature maps.
    :obj:`mb_locs` contains information of the coordinates of bounding boxes
    and :obj:`mb_confs` contains confidence scores of each classes.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
       Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Args:
        n_class (int): The number of classes possibly including the background.
        aspect_ratios (iterable of tuple or int): The aspect ratios of
            default bounding boxes for each feature map.
        initialW: An initializer used in
            :meth:`chainer.links.Convolution2d.__init__`.
            The default value is :class:`chainer.initializers.LeCunUniform`.
        initial_bias: An initializer used in
            :meth:`chainer.links.Convolution2d.__init__`.
            The default value is :class:`chainer.initializers.Zero`.

    """

    def __init__(
            self, n_class, aspect_ratios,
            initialW=None, initial_bias=None):
        self.n_class = n_class
        self.aspect_ratios = aspect_ratios

        super(Multibox_custom, self).__init__()
        with self.init_scope():
            self.loc = chainer.ChainList()
            self.conf = chainer.ChainList()

        if initialW is None:
            initialW = initializers.LeCunUniform()
        if initial_bias is None:
            initial_bias = initializers.Zero()
        init = {'initialW': initialW, 'initial_bias': initial_bias}

        for ar in aspect_ratios:
            n = (len(ar) + 1) * 2
            self.loc.add_link(L.Convolution2D(n * 4, 3, pad=1, **init))
            self.conf.add_link(L.Convolution2D(
                n * self.n_class, 3, pad=1, **init))

    def forward(self, xs):
        """Compute loc and conf from feature maps

        This method computes :obj:`mb_locs` and :obj:`mb_confs`
        from given feature maps.

        Args:
            xs (iterable of chainer.Variable): An iterable of feature maps.
                The number of feature maps must be same as the number of
                :obj:`aspect_ratios`.

        Returns:
            tuple of chainer.Variable:
            This method returns two :obj:`chainer.Variable`: :obj:`mb_locs` and
            :obj:`mb_confs`.

            * **mb_locs**: A variable of float arrays of shape \
                :math:`(B, K, 4)`, \
                where :math:`B` is the number of samples in the batch and \
                :math:`K` is the number of default bounding boxes.
            * **mb_confs**: A variable of float arrays of shape \
                :math:`(B, K, n\_fg\_class + 1)`.

        """

        mb_locs = []
        mb_confs = []
        for i, x in enumerate(xs):
            if len(x.shape) == 5:
                loc_pad_org = self.loc[i].pad
                conf_pad_org = self.conf[i].pad
                self.loc[i].pad = (0,0)
                self.conf[i].pad = (0,0)

                x_shape = x.shape
                x = x.reshape((-1,) + x_shape[-3:])

                mb_loc = self.loc[i](x)
                mb_loc = mb_loc.transpose(0,2,3,1)
                # mb_loc = mb_loc.reshape(-1,4)
                mb_loc = mb_loc.reshape(x_shape[0],-1, 4)
                mb_locs.append(mb_loc)

                mb_conf = self.conf[i](x)
                mb_conf = mb_conf.transpose(0, 2, 3, 1)
                # mb_loc = mb_loc.reshape(-1,4)
                mb_conf = mb_conf.reshape(x_shape[0], -1, self.n_class)
                mb_confs.append(mb_conf)

                self.loc[i].pad = loc_pad_org
                self.conf[i].pad = conf_pad_org
            else:
                mb_loc = self.loc[i](x)
                mb_loc = F.transpose(mb_loc, (0, 2, 3, 1))
                mb_loc = F.reshape(mb_loc, (mb_loc.shape[0], -1, 4))
                mb_locs.append(mb_loc)

                mb_conf = self.conf[i](x)
                mb_conf = F.transpose(mb_conf, (0, 2, 3, 1))
                mb_conf = F.reshape(
                    mb_conf, (mb_conf.shape[0], -1, self.n_class))
                mb_confs.append(mb_conf)

        mb_locs = F.concat(mb_locs, axis=1)
        mb_confs = F.concat(mb_confs, axis=1)

        return mb_locs, mb_confs

class Mbox_custom(Multibox_custom):
    def __init__(
            self, n_class, aspect_ratios,
            initialW=None, initial_bias=None,ch_num=None):
        super(Mbox_custom, self).__init__(n_class, aspect_ratios,
            initialW, initial_bias)
        with self.init_scope():
            self.c_reduce = chainer.ChainList()
        self.ch_reduced = ch_num if ch_num else [32,64,32,16,16,16]
        for ch in self.ch_reduced:
            # n = (len(ar) + 1) * 2
            self.c_reduce.add_link(L.Convolution2D(ch, 1))

    def forward(self, xs):
        xs_ = xs[:]
        for i in range(len(xs_)):
            xs_shape = list(xs_[i].shape)
            if len(xs_shape) == 5:
                xs_[i] = xs_[i].reshape([-1,]+xs_shape[-3:])
            xs_[i] = self.c_reduce[i](xs_[i])
            if len(xs_shape) == 5:
                xs_shape[2] = self.ch_reduced[i]
                xs_[i] = xs_[i].reshape(xs_shape)
        return super(Mbox_custom, self).forward(xs_)

class SSD512_vd(SSD512):
    def __init__(self, n_fg_class=None, pretrained_model=None,defaultbox_size=(35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6),mean = _imagenet_mean):
        # n_fg_class, path = _check_pretrained_model(
        #     n_fg_class, pretrained_model, self._models)
        param, path = utils.prepare_pretrained_model(
            {'n_fg_class': n_fg_class}, pretrained_model, self._models)

        super(SSD512, self).__init__(
            extractor=VGG16Extractor512(),
            multibox=Multibox(
                # n_class=n_fg_class + 1,
                n_class=param['n_fg_class'] + 1,
                aspect_ratios=(
                    (2,), (2, 3), (2, 3), (2, 3), (2, 3), (2,), (2,))),
            steps=(8, 16, 32, 64, 128, 256, 512),
            sizes=defaultbox_size,
            mean=mean)

        if path:
            # _load_npz(path, self)
            chainer.serializers.load_npz(path, self, strict=False)

class SSD300_vd(SSD300):
    def __init__(self, n_fg_class=None, pretrained_model=None,defaultbox_size=(30, 60, 111, 162, 213, 264, 315),mean = _imagenet_mean,SA=False,**attn_args):
        # n_fg_class, path = _check_pretrained_model(
        #     n_fg_class, pretrained_model, self._models)
        param, path = utils.prepare_pretrained_model(
            {'n_fg_class': n_fg_class}, pretrained_model, self._models)

        if SA:
            extractor_class = VGG16Extractor300_SA
            ext_args = attn_args
        else:
            extractor_class = VGG16Extractor300
            ext_args = {}

        super(SSD300, self).__init__(
            extractor=extractor_class(**ext_args),
            multibox=Multibox_custom(
                # n_class=n_fg_class + 1,
                param['n_fg_class'] + 1,
                aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))),
            steps=(8, 16, 32, 64, 100, 300),
            sizes=defaultbox_size,
            mean=mean)

        if path:
            # _load_npz(path, self)
            chainer.serializers.load_npz(path, self, strict=False)

        self.RTN = None
        self.RTN_cls = None
        self.mbox_delta = None

    def set_RTN(self, RTN):
        self.RTN = RTN

    def set_RTN_cls(self, RTN_cls):
        self.RTN_cls = RTN_cls

    def set_mbox_delta(self, mbox_delta):
        self.mbox_delta = mbox_delta

    def _prepare(self, img):
        img = img.astype(np.float32)
        img = transforms.resize(img, (self.insize, self.insize))
        img -= self.mean
        if chainer.get_dtype() == chainer.mixed16:
            img = img.astype(np.float16)
        return img

    def predict(self, imgs,calc_entropy=False):
        x = []
        sizes = []
        for img in imgs:
            _, H, W = img.shape
            img = self._prepare(img)
            x.append(self.xp.array(img))
            sizes.append((H, W))

        with chainer.using_config('train', False), \
                chainer.function.no_backprop_mode():
            x = chainer.Variable(self.xp.stack(x))

            fmaps = self.extractor(x)
            if self.RTN:
                fmaps[0] = self.RTN(fmaps[0])
            mb_locs, mb_confs = self.multibox(fmaps)
            if self.RTN_cls:
                mb_locs, mb_confs = self.RTN_cls(mb_locs, mb_confs)
            if self.mbox_delta:
                mb_locs_delta, mb_confs_delta = self.mbox_delta(fmaps)
                mb_locs += mb_locs_delta
                mb_confs += mb_confs_delta
            # mb_locs, mb_confs = self.forward(x)

        #----->> code for analysis
        if calc_entropy:
            mb_confs_d = mb_confs[:, 0:4 * 38 * 38, :]  # extract the part # of the first feature map
            mb_confs_p = F.softmax(mb_confs_d, axis=-1)

            mb_confs_argmax = self.xp.argmax(mb_confs_p.array,axis=-1)
            mask_pos = mb_confs_argmax > 0
            mask_neg = mb_confs_argmax == 0
            # if self.min_entropy == 1:
            norm_coef = np.prod(mb_confs_p.shape[:2])
            norm_coef_pos = mask_pos.astype("f").sum()
            norm_coef_neg = mask_neg.astype("f").sum()
            # else:
            #     norm_coef = mb_confs_p.shape[0]
            entropy_raw = - F.sum(mb_confs_p * F.log2(
                mb_confs_p + (mb_confs_p.array < 1e-30) * 1e-30),axis=-1)

            entropy = F.sum(entropy_raw) / norm_coef
            entropy_pos = F.sum(entropy_raw * mask_pos) / norm_coef_pos
            entropy_neg = F.sum(entropy_raw * mask_neg) / norm_coef_neg

            # entropy = - F.sum(mb_confs_p * F.log2(
            #     mb_confs_p + (mb_confs_p.array < 1e-30) * 1e-30)) / norm_coef
            # print("(output for analysis) prediction entropy:{}".format(entropy))
            # print(entropy.data)
            entropy = entropy.data
            entropy_pos = entropy_pos.data
            entropy_neg = entropy_neg.data
            if not isinstance(entropy,np.ndarray):
                entropy = self.xp.asnumpy(entropy)
            if not isinstance(entropy_pos, np.ndarray):
                entropy_pos = self.xp.asnumpy(entropy_pos)
            if not isinstance(entropy_neg,np.ndarray):
                entropy_neg = self.xp.asnumpy(entropy_neg)

        # -------<<

        mb_locs, mb_confs = mb_locs.array, mb_confs.array

        bboxes = []
        labels = []
        scores = []
        for mb_loc, mb_conf, size in zip(mb_locs, mb_confs, sizes):
            bbox, label, score = self.coder.decode(
                mb_loc, mb_conf, self.nms_thresh, self.score_thresh)
            bbox = transforms.resize_bbox(
                bbox, (self.insize, self.insize), size)
            bboxes.append(chainer.backends.cuda.to_cpu(bbox))
            labels.append(chainer.backends.cuda.to_cpu(label))
            scores.append(chainer.backends.cuda.to_cpu(score))

        if calc_entropy:
            return bboxes, labels, scores, [entropy, entropy_pos, entropy_neg]

        return bboxes, labels, scores



class DA4_discriminator(Chain):
    def __init__(self):
        #w = chainer.initializers.Normal(wscale)
        super(DA4_discriminator, self).__init__()
        # init = {
        #     'initialW': initializers.LeCunUniform(),
        #     'initial_bias': initializers.Zero(),
        # }
        with self.init_scope():
            self.conv1 = L.Convolution2D(1024, 3, pad=1)
            self.conv2 = L.Convolution2D(512, 1)
            self.conv3 = L.Convolution2D(256, 1)
            self.conv4 = L.Convolution2D(1, 1)

    def __call__(self, x):
        h = F.leaky_relu(self.conv1(x[0]))
        h = F.leaky_relu(self.conv2(h))
        h = F.leaky_relu(self.conv3(h))
        h = F.leaky_relu(self.conv4(h))
        return h

class Cls_discriminator(Chain):
    def __init__(self):
        #w = chainer.initializers.Normal(wscale)
        super(Cls_discriminator, self).__init__()
        # init = {
        #     'initialW': initializers.LeCunUniform(),
        #     'initial_bias': initializers.Zero(),
        # }
        with self.init_scope():
            self.conv1 = L.Convolution2D(64, 1)
            self.conv2 = L.Convolution2D(32, 1)
            self.conv3 = L.Convolution2D(1, 1)

    def __call__(self, x):
        h = F.leaky_relu(self.conv1(x))
        h = F.leaky_relu(self.conv2(h))
        h = F.leaky_relu(self.conv3(h))
        return h

class DA4_discriminator_bn(Chain):
    def __init__(self):
        #w = chainer.initializers.Normal(wscale)
        super(DA4_discriminator_bn, self).__init__()
        # init = {
        #     'initialW': initializers.LeCunUniform(),
        #     'initial_bias': initializers.Zero(),
        # }
        with self.init_scope():
            self.conv1 = L.Convolution2D(1024, 3, pad=1)
            self.bn1 = L.BatchNormalization(1024)
            self.conv2 = L.Convolution2D(512, 1)
            self.bn2 = L.BatchNormalization(512)
            self.conv3 = L.Convolution2D(256, 1)
            self.bn3 = L.BatchNormalization(256)
            self.conv4 = L.Convolution2D(1, 1)

    def __call__(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x[0])))
        h = F.leaky_relu(self.bn2(self.conv2(h)))
        h = F.leaky_relu(self.bn3(self.conv3(h)))
        h = F.leaky_relu(self.conv4(h))
        return h

class Generator_coGAN(chainer.Chain):
    def __init__(self,norm="bn",res_num=5,comm=None):
        self.res_num = res_num
        # norm_option = {"bn": init_BN(comm),None: self.identity}
        layers = {}
        # with self.init_scope():
            # self.c_0 = CNABlock(3, 64, nn='conv', norm=norm)
        layers['c_1'] = CNABlock(3, 32, nn='g_down_conv', norm=norm,comm=comm)
        layers['c_2'] = CNABlock(32, 64, nn='g_down_conv', norm=norm,comm=comm)
        layers['c_3'] = CNABlock(64, 128, nn='g_down_conv', norm=norm,comm=comm)

        for i in range(self.res_num):
            layers['c_res_' + str(i)] = ResBlock(128, norm=norm, reflect=0,comm=comm)

        layers['c_4'] = CNABlock(128, 256, nn='conv', norm=norm,comm=comm)
        layers['c_5'] = CNABlock(256, 512, nn='conv', norm=norm,activation=None,comm=comm)
        super(Generator_coGAN, self).__init__(**layers)

    def forward(self, x):
        h = self.c_1(x)
        h = self.c_2(h)
        h = self.c_3(h)
        for i in range(self.res_num):
            h = getattr(self, 'c_res_' + str(i))(h)
        h = self.c_4(h)
        h = self.c_5(h)

        return h

class Generator_VGG16_simple(chainer.Chain):
    def __init__(self,norm=None,residual=False,comm=None):
        super(Generator_VGG16_simple, self).__init__()
        norm_option = {"bn": init_BN(comm),None: self.identity}
        self.residual = residual
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(64, 3, pad=1)
            self.conv1_2 = L.Convolution2D(64, 3, pad=1)

            self.conv2_1 = L.Convolution2D(128, 3, pad=1)
            self.conv2_2 = L.Convolution2D(128, 3, pad=1)

            self.conv3_1 = L.Convolution2D(256, 3, pad=1)
            self.conv3_2 = L.Convolution2D(256, 3, pad=1)
            self.conv3_3 = L.Convolution2D(256, 3, pad=1)

            self.conv4_1 = L.Convolution2D(512, 3, pad=1)
            self.conv4_2 = L.Convolution2D(512, 3, pad=1)
            self.conv4_3 = L.Convolution2D(512, 3, pad=1)
            # self.norm4 = Normalize(512, initial=initializers.Constant(20))


            self.norm1_1 = norm_option[norm](64)
            self.norm1_2 = norm_option[norm](64)

            self.norm2_1 = norm_option[norm](128)
            self.norm2_2 = norm_option[norm](128)

            self.norm3_1 = norm_option[norm](256)
            self.norm3_2 = norm_option[norm](256)
            self.norm3_3 = norm_option[norm](256)

            self.norm4_1 = norm_option[norm](512)
            self.norm4_2 = norm_option[norm](512)

            if residual:
                self.c_r_1 = L.Convolution2D(128,1)
                self.c_r_2 = L.Convolution2D(256, 1)
                self.c_r_3 = L.Convolution2D(512, 1)

            # self.conv5_1 = L.DilatedConvolution2D(512, 3, pad=1)
            # self.conv5_2 = L.DilatedConvolution2D(512, 3, pad=1)
            # self.conv5_3 = L.DilatedConvolution2D(512, 3, pad=1)
            #
            # self.conv6 = L.DilatedConvolution2D(1024, 3, pad=6, dilate=6)
            # self.conv7 = L.Convolution2D(1024, 1)

    def identity(self,x): #ignore x
        return self._identity

    def _identity(self,x):
        return x

    def forward(self, x):
        # ys = []

        h = F.relu(self.norm1_1(self.conv1_1(x)))
        h = F.relu(self.norm1_2(self.conv1_2(h)))
        h1 = F.max_pooling_2d(h, 2)

        h = F.relu(self.norm2_1(self.conv2_1(h1)))
        h = F.relu(self.norm2_2(self.conv2_2(h)))
        h2 = F.max_pooling_2d(h, 2)
        if self.residual:
            h2 = h2 + self.c_r_1(F.resize_images(h1,h2.shape[-2:]))

        h = F.relu(self.norm3_1(self.conv3_1(h2)))
        h = F.relu(self.norm3_2(self.conv3_2(h)))
        h = F.relu(self.norm3_3(self.conv3_3(h)))
        h3 = F.max_pooling_2d(h, 2)
        if self.residual:
            h3 = h3 + self.c_r_2(F.resize_images(h2,h3.shape[-2:]))

        h = F.relu(self.norm4_1(self.conv4_1(h3)))
        h = F.relu(self.norm4_2(self.conv4_2(h)))
        if self.residual:
            h = h + self.c_r_3(F.resize_images(h3,h.shape[-2:]))

        # h = F.relu(self.conv4_3(h))
        h = self.conv4_3(h)
        # ys.append(self.norm4(h))

        # return self.norm4(h)

        return h

        # h = F.max_pooling_2d(h, 2)
        #
        # h = F.relu(self.conv5_1(h))
        # h = F.relu(self.conv5_2(h))
        # h = F.relu(self.conv5_3(h))
        # h = F.max_pooling_2d(h, 3, stride=1, pad=1)
        #
        # h = F.relu(self.conv6(h))
        # h = F.relu(self.conv7(h))
        # ys.append(h)
        #
        # return ys

import random
import numpy as np
import math

class fmapBuffer(object):
    def __init__(self,bufsize,mode =0,discriminator=None,batchsize = 32,gpu = -1):  #mode 0:align src and tgt, 1:not align, 2:sort by loss value
        self.bufsize = bufsize
        self.buffer_src = []
        self.buffer_tgt = []
        self.mode = mode
        self.discriminator = discriminator
        self.loss_src = None
        self.loss_tgt = None
        self.batchsize = batchsize
        self.gpu = gpu
        self.fmap_num = None
        # self.buffer_src_data = None
        # self.buffer_tgt_data = None

    def get_examples(self, n_samples):
        if self.buffer_src == []:
            n_return, src_samples, tgt_samples = 0, None, None
        elif n_samples >= len(self.buffer_src):
            n_return = len(self.buffer_src)
            n_fmap = len(self.buffer_src[0])
            src_samples = []
            tgt_samples = []
            for i in range(n_fmap):
                src_samples.append(np.stack([x[i] for x in self.buffer_src]))
                tgt_samples.append(np.stack([x[i] for x in self.buffer_tgt]))
        else:
            n_return = n_samples
            if self.mode == 2:
                indices_src = [x for x in range(n_samples)]
                indices_tgt = indices_src
            else:
                indices_src = random.sample(range(len(self.buffer_src)), n_samples)
                if self.mode == 0:
                    indices_tgt = indices_src
                else:
                    indices_tgt = random.sample(range(len(self.buffer_tgt)), n_samples)
            n_fmap = len(self.buffer_src[0])
            src_samples = []
            tgt_samples = []
            for i in range(n_fmap):
                src_samples.append(np.stack([self.buffer_src[x][i] for x in indices_src]))
                tgt_samples.append(np.stack([self.buffer_tgt[x][i] for x in indices_tgt]))
        return n_return, src_samples, tgt_samples

    def set_examples(self, src_samples, tgt_samples):
        n_samples = src_samples[0].shape[0]
        if self.bufsize < n_samples:
            print(self.__class__.__name__+"- set_example(): n_examples must not be larger than bufsize ")
            raise ValueError
        n_fmap = len(src_samples)
        n_fmap_elements = src_samples[0].shape[2] * src_samples[0].shape[3]
        if self.mode == 2:
            for i in range(n_samples):
                self.buffer_src.append([src_samples[x][i] for x in range(n_fmap)])
                self.buffer_tgt.append([tgt_samples[x][i] for x in range(n_fmap)])
            del src_samples
            del tgt_samples
            transfer_array = lambda x: chainer.cuda.to_gpu(x, device=self.gpu) if self.gpu >= 0 else lambda x: x
            for i in range(int(math.ceil(len(self.buffer_src)/self.batchsize))):
                src_examples_ = []
                tgt_examples_ = []
                for j in range(n_fmap):
                    src_examples_.append(transfer_array(np.stack([x[j] for x in self.buffer_src[i*self.batchsize:(i+1)*self.batchsize]])))
                    tgt_examples_.append(transfer_array(np.stack([x[j] for x in self.buffer_tgt[i * self.batchsize:(i + 1) * self.batchsize]])))
                with chainer.no_backprop_mode():
                    src_loss_ = chainer.cuda.to_cpu(F.sum(F.softplus(-self.discriminator(src_examples_)),axis=(1,2,3)).data) / n_fmap_elements
                    tgt_loss_ = chainer.cuda.to_cpu(
                        F.sum(F.softplus(self.discriminator(tgt_examples_)), axis=(1, 2, 3)).data) / n_fmap_elements
                if i == 0:
                    self.loss_src = src_loss_
                    self.loss_tgt = tgt_loss_
                else:
                    self.loss_src = np.hstack((self.loss_src, src_loss_))
                    self.loss_tgt = np.hstack((self.loss_tgt, tgt_loss_))
            # self.buffer_src = sorted(self.buffer_src, key=lambda x: self.loss_src[self.buffer_src.index(x)],reverse=True)
            # self.buffer_tgt = sorted(self.buffer_tgt, key=lambda x: self.loss_tgt[self.buffer_tgt.index(x)],reverse=True)
            index_sorted_src = np.argsort(self.loss_src)[::-1]
            index_sorted_tgt = np.argsort(self.loss_tgt)[::-1]
            self.buffer_src = [self.buffer_src[x] for x in index_sorted_src]
            self.buffer_tgt = [self.buffer_tgt[x] for x in index_sorted_tgt]
            self.loss_src = np.sort(self.loss_src)[::-1]
            self.loss_tgt = np.sort(self.loss_tgt)[::-1]
            self.buffer_src = self.buffer_src[:self.bufsize]
            self.buffer_tgt = self.buffer_tgt[:self.bufsize]
            self.loss_src = self.loss_src[:self.bufsize]
            self.loss_tgt = self.loss_tgt[:self.bufsize]
        else:
            n_room = self.bufsize - len(self.buffer_src)
            if n_room >= n_samples:
                for i in range(n_samples):
                    self.buffer_src.append([src_samples[x][i] for x in range(n_fmap)])
                    self.buffer_tgt.append([tgt_samples[x][i] for x in range(n_fmap)])
            else:
                indices_buf_src = random.sample(range(len(self.buffer_src)), n_samples - n_room)
                if self.mode == 0:
                    indices_tgt_src = indices_buf_src
                else:
                    indices_tgt_src = random.sample(range(len(self.buffer_tgt)), n_samples - n_room)
                indices_samples = range(n_room,n_samples)
                for i,j,k in zip(indices_buf_src,indices_tgt_src, indices_samples):
                    self.buffer_src[i] = [src_samples[x][k] for x in range(n_fmap)]
                    self.buffer_tgt[j] = [tgt_samples[x][k] for x in range(n_fmap)]
                for i in range(n_room):
                    self.buffer_src.append([src_samples[x][i] for x in range(n_fmap)])
                    self.buffer_tgt.append([tgt_samples[x][i] for x in range(n_fmap)])

    def encode(self):
        if len(self.buffer_src) > 0:
            for i in range(len(self.buffer_src[0])):
                src_fmaps = [self.buffer_src[x][i] for x in range(len(self.buffer_src))]
                setattr(self,"buffer_src_data_"+str(i),np.array(src_fmaps))
                tgt_fmaps = [self.buffer_tgt[x][i] for x in range(len(self.buffer_tgt))]
                setattr(self,"buffer_tgt_data_" + str(i), np.array(tgt_fmaps))
            self.fmap_num = len(self.buffer_src[0])
        else:
            self.fmap_num = 0

    #     self.buffer_src_data = np.array(self.buffer_src)
    #     self.buffer_tgt_data = np.array(self.buffer_tgt)
    #
    def decode(self):
        if self.fmap_num > 0:
            buffer_src_data = []
            buffer_tgt_data = []
            # i = 0
            # src_temp = getattr(self,"buffer_src_data_"+str(i),None)
            # tgt_temp = getattr(self, "buffer_tgt_data_" + str(i),None)
            # while src_temp != None:
            for i in range(self.fmap_num):
                src_temp = getattr(self, "buffer_src_data_" + str(i), None)
                tgt_temp = getattr(self, "buffer_tgt_data_" + str(i),None)
                buffer_src_data.append(src_temp)
                buffer_tgt_data.append(tgt_temp)
            # i += 1
            # src_temp = getattr(self, "buffer_src_data_" + str(i), None)
            # tgt_temp = getattr(self, "buffer_tgt_data_" + str(i), None)
        # if len(buffer_src_data) > 0:
            self.buffer_src = [[buffer_src_data[x][y] for x in range(len(buffer_src_data))] for y in range(len(buffer_src_data[0]))]
            self.buffer_tgt = [[buffer_tgt_data[x][y] for x in range(len(buffer_tgt_data))] for y in
                               range(len(buffer_tgt_data[0]))]

    def serialize(self, serializer):
        if isinstance(serializer, chainer.serializer.Serializer):
            self.encode()
        self.fmap_num = serializer("fmap_num", self.fmap_num)
        # i = 0
        # src_temp = getattr(self, "buffer_src_data_" + str(i), None)
        # tgt_temp = getattr(self, "buffer_tgt_data_" + str(i), None)
        # while src_temp != None:
        # if isinstance(serializer, chainer.serializer.Deserializer):
        #     for i in range(self.fmap_num):
        #         setattr(self, "buffer_src_data_" + str(i), None)
        #         setattr(self, "buffer_tgt_data_" + str(i), None)
        for i in range(self.fmap_num):
            src_temp = getattr(self, "buffer_src_data_" + str(i), None)
            tgt_temp = getattr(self, "buffer_tgt_data_" + str(i), None)
            src_temp = serializer("buffer_src_data_" + str(i), src_temp)
            tgt_temp = serializer("buffer_tgt_data_" + str(i), tgt_temp)
            if isinstance(serializer, chainer.serializer.Deserializer):
                setattr(self, "buffer_src_data_" + str(i), src_temp)
                setattr(self, "buffer_tgt_data_" + str(i), tgt_temp)
            # buffer_src_data.append(src_temp)
            # buffer_tgt_data.append(tgt_temp)
            # i += 1
            # src_temp = getattr(self, "buffer_src_data_" + str(i), None)
            # tgt_temp = getattr(self, "buffer_tgt_data_" + str(i), None)
        if isinstance(serializer, chainer.serializer.Deserializer):
            self.decode()
        # self.rank_map_data = serializer('rank_map', self.rank_map_data)
        # self.rank_F1_data = serializer('rank_F1', self.rank_F1_data)
        # self.rank_mean_data = serializer('rank_mean', self.rank_mean_data)
        # if isinstance(serializer, chainer.serializer.Deserializer):
        #     self.rank_map = self.decode(self.rank_map_data)
        #     self.rank_F1 = self.decode(self.rank_F1_data)
        #     self.rank_mean = self.decode(self.rank_mean_data)
    #     buf = []
    #     for i in range(len(data)):
    #         buf.append(data[i])
    #         ranking_list.append([data[i][j] for j in range(data.shape[1])])
    #         ranking_list[i][0] = int(ranking_list[i][0]) #iter number must be int
    #     return ranking_list