import torch.nn as nn
from . import element
import collections
from itertools import repeat


class SeqName:
    def __init__(self, module_name):
        ''' define the name of sequence, to be used by class LinearSeq & Conv2dSeq '''
        self.moduleName = module_name
    
    def seq_name(self):
        self.moduleName = str(eval(self.moduleName)+1)
        return self.moduleName

class BatchNorm(object):
    ''' define Batch Normalization, to be used by class LinearSeq & Conv2dSeq '''
    def _batchnorm1d(self, name, n_output):
        self.seq.add_module(name, nn.BatchNorm1d(n_output, eps=self.eps, momentum=self.momentum))
    
    def _batchnorm2d(self, name, out_channel):
        self.seq.add_module(name, nn.BatchNorm2d(out_channel, eps=self.eps, momentum=self.momentum))
    
    def _batchnorm3d(self, name, out_channel):
        self.seq.add_module(name, nn.BatchNorm3d(out_channel, eps=self.eps, momentum=self.momentum))

class Activation(object):
    ''' define activation functions, to be used by class LinearSeq & Conv2dSeq '''
    def _activation(self, module_name, active_name):
        self.seq.add_module(module_name, element.activation(active_name=active_name))

class Pooling(object):
    ''' define pooling, to be used by class LinearSeq & Conv2dSeq '''
    def _pooling(self, module_name, pool_name):
        self.seq.add_module(module_name, element.pooling(pool_name=pool_name))


def _ntuple(x, n=2):
    '''
    return a tuple
    
    :param x: an integer or a tuple with more than two elements
    :param n: the number to be repeated for an integer, it only works for an integer
    '''
    if isinstance(x, collections.Iterable):
        return x
    else:
        return tuple(repeat(x, n))

def multi_ntuple(x, n=2):
    '''
    return a tuple or a list that contain tuples
    
    :param x: an integer, a tuple or a list whose element is tuple (with more than two elements)
    :param n: the number to be repeated for an integer, it only works for an integer
    '''
    if type(x) is list:
        for i in range(len(x)):
            x[i] = _ntuple(x[i], n=n)
    else:
        x = _ntuple(x, n=n)
    return x


class Conv2dSeq(SeqName,BatchNorm,Activation,Pooling):
    def __init__(self, channels, kernels_size=None, strides=None, extra_pads=None, mainBN=True, finalBN=True,
                 mainActive='relu', finalActive='relu', mainPool='None', finalPool='None', eps=1e-05, momentum=0.1, transConv2d=False,
                 upSample=False, upSample_mode='nearest', unPool=False, in_side=512):
        ''' sequence of Conv2d or ConvTranspose2d '''
        
        super(Conv2dSeq, self).__init__('-1') #or SeqName.__init__(self, '-1')
        self.channels = channels
        self.layers = len(channels) - 1
        if kernels_size is None:
            self.kernels_size = [(3,3) for i in range(self.layers)]
        else:
            self.kernels_size = multi_ntuple(kernels_size)
        if strides is None:
            self.strides = [(2,2) for i in range(self.layers)]
        else:
            self.strides = multi_ntuple(strides)
        if extra_pads is None:
            self.extra_pads = [(0,0) for i in range(self.layers)]
        else:
            self.extra_pads = multi_ntuple(extra_pads)
        self.dilation = 1
        self.mainBN = mainBN
        self.finalBN = finalBN
        self.mainActive = mainActive
        self.finalActive = finalActive
        self.mainPool = mainPool
        self.finalPool = finalPool
        self.eps = eps
        self.momentum = momentum
        self.bias = True
        self.transConv2d = transConv2d
        self.upSample = upSample
        self.upSample_mode = upSample_mode
        self.unPool = unPool
        self.sides = [_ntuple(in_side)]
        self.seq = nn.Sequential()
    
    def getPadding(self, extra_pad, kernel_size, stride, dilation):
        '''
        obtain the padding or output_padding for Conv2d and ConvTranspose2d when giving kernel size, stride, and extra_pad
        '''
        pads = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        #note that output padding must be smaller than either stride or dilation
        out_pads = [i for i in range(min(stride, dilation)+1)]
        if self.transConv2d:
            for out_pad in out_pads:
                for pad in pads:
                    if kernel_size-stride-2*pad+out_pad == extra_pad:
                        padding = pad
                        output_padding = out_pad
                        return padding, output_padding
        else:
            for pad in pads:
                # print((2*pad - dilation*(kernel_size-1) -1)//stride + 1, '!!!')
                if (2*pad - dilation*(kernel_size-1) -1)//stride + 1 == extra_pad:
                    padding = pad
                    return padding
            print ('Warning: No padding matching !!!')
    
    def __conv2d(self, name, in_channel, out_channel, kernel_size, stride, extra_pad):
        '''
        The default settings are 'extra_pad=0' and 'dilation=1', so, the size of channel will be reduced by half when using 'Conv2d',
        and will be increase to two times of the original size when using 'ConvTranspose2d'.
        
        For Conv2d, the output size is: H_out = H_in/stride + (2*padding - dilation*(kernel_size-1) -1)/stride + 1,
        for transConv2d, the output size is: H_out = H_in*stride + kernel_size-stride-2*padding + output_padding
        
        :param extra_pad: extra_pad is defined as "(2*padding - dilation*(kernel_size-1) -1)/stride + 1" for Conv2d,
                          and "kernel_size-stride-2*padding + output_padding" for transConv2d
        '''
        kernel_size = _ntuple(kernel_size)
        stride = _ntuple(stride)
        extra_pad = _ntuple(extra_pad)
        if self.transConv2d:
            padding_0, output_panding_0 = self.getPadding(extra_pad[0], kernel_size[0], stride[0], _ntuple(self.dilation)[0])
            padding_1, output_panding_1 = self.getPadding(extra_pad[1], kernel_size[1], stride[1], _ntuple(self.dilation)[1])
            padding = (padding_0, padding_1)
            output_panding = (output_panding_0, output_panding_1)
            
            side_H = self.sides[-1][0]*stride[0] + kernel_size[0]-stride[0]-2*padding[0] + output_panding[0]
            side_W = self.sides[-1][1]*stride[1] + kernel_size[1]-stride[1]-2*padding[1] + output_panding[1]
            self.sides.append((side_H, side_W))
            self.seq.add_module(name, nn.ConvTranspose2d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size,stride=stride,padding=padding,output_padding=output_panding,bias=self.bias,dilation=self.dilation))
        else:
            padding_0 = self.getPadding(extra_pad[0], kernel_size[0], stride[0], _ntuple(self.dilation)[0])
            padding_1 = self.getPadding(extra_pad[1], kernel_size[1], stride[1], _ntuple(self.dilation)[1])
            padding = (padding_0, padding_1)
            
            side_H = self.sides[-1][0]//stride[0] + (2*padding[0] - _ntuple(self.dilation)[0]*(kernel_size[0]-1)-1)//stride[0] + 1
            side_W = self.sides[-1][1]//stride[1] + (2*padding[1] - _ntuple(self.dilation)[1]*(kernel_size[1]-1)-1)//stride[1] + 1
            self.sides.append((side_H, side_W))
            self.seq.add_module(name, nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size,stride=stride,padding=padding,bias=self.bias,dilation=self.dilation))
    
    def __upSample(self, name):
        self.seq.add_module(name, nn.Upsample(scale_factor=2,mode=self.upSample_mode))
    
    def __unPool(self, name):
        self.seq.add_module(name, nn.MaxUnpool2d(kernel_size=2))
    
    def _stride(self):
        self.kernels_size = multi_ntuple(self.kernels_size)
        self.strides = multi_ntuple(self.strides)
        stride_list = [list(self.strides[i]) for i in range(len(self.strides))]
        for i in range(len(self.kernels_size)):
            for j in range(2):
                if self.kernels_size[i][j]==1 and self.strides[i][j]!=1:
                    print ('The stride "%s" that not match kernel size was enforced to be "1" !!!'%self.strides[i][j])
                    stride_list[i][j] = 1
        self.strides = [tuple(stride_list[i]) for i in range(len(self.strides))]
    
    def get_seq(self):
        self._stride()
        
        for i in range(self.layers-1):
            if self.upSample:
                self.__upSample(self.seq_name())
                self.sides[-1] = (self.sides[-1][0]*2, self.sides[-1][1]*2)
            if self.unPool:
                self.__unPool(self.seq_name())
                self.sides[-1] = (self.sides[-1][0]*2, self.sides[-1][1]*2)
            self.__conv2d(self.seq_name(), self.channels[i], self.channels[i+1], self.kernels_size[i], self.strides[i], self.extra_pads[i])
            if self.mainBN:
                self._batchnorm2d(self.seq_name(), self.channels[i+1])
            if self.mainActive!='None':
                self._activation(self.seq_name(), self.mainActive)
            if self.mainPool!='None':
                self._pooling(self.seq_name(), self.mainPool)
                self.sides[-1] = (self.sides[-1][0]//2, self.sides[-1][1]//2)
        
        if self.upSample:
            self.__upSample(self.seq_name())
            self.sides[-1] = (self.sides[-1][0]*2, self.sides[-1][1]*2)
        if self.unPool:
            self.__unPool(self.seq_name())
            self.sides[-1] = (self.sides[-1][0]*2, self.sides[-1][1]*2)
        self.__conv2d(self.seq_name(), self.channels[-2], self.channels[-1], self.kernels_size[-1], self.strides[-1], self.extra_pads[-1])
        if self.finalBN:
            self._batchnorm2d(self.seq_name(), self.channels[-1])
        if self.finalActive!='None':
            self._activation(self.seq_name(), self.finalActive)
        if self.finalPool!='None':
            self._pooling(self.seq_name(), self.finalPool)
            self.sides[-1] = (self.sides[-1][0]//2, self.sides[-1][1]//2)
        print ('sides:', self.sides)
        return self.seq


#%% test Conv2dSeq
if __name__ == '__main__':
#    fc = LinearSeq([10,7,5,3], mainBN=False, finalBN=True, mainDropout='dropout')
#    print fc.get_seq()
    
    conv = Conv2dSeq([1,2,3,4,5,6,7,8,9],extra_pads=[0 for i in range(8)],transConv2d=False)
#    conv.kernels_size = [3,3,3,3,3,3,3,3]
    conv.kernels_size = [(1,4),4,4,4,4,4,4,4]
#    conv.kernels_size = [(3,4),(3,4),(3,4),(3,4),(3,4),(3,4),(3,4),(3,4)]
#    conv.strides = [2,2,2,2,2,2,2,2]
    conv.strides = [(2,2),(2,2),(2,2),(2,2),(2,2),(2,2),(2,2),(2,2)]
    conv.finalPool = 'maxPool2d'
    print(conv.get_seq())#, conv.sides


