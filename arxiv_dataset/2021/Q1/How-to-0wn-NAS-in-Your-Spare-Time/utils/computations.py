"""
    Characteristics of architecture attributes:
    used for the reconstruction of real architecture.
"""

# ------------------------------------------------------------------------------
#   Unary operations that require an input from the preceding layer
# ------------------------------------------------------------------------------
_COMPUTE_UNARY_PT = [
    # Tensor operations
    'transpose',
    'narrow',
    'view',
    'squeeze',
    'repeat',
    # computational layers
    'Embedding',
    'Convolution',
    'Conv2d',
    'Conv3d',
    'ConvTrans3d',
    'FC',
    'MaxPool1d',
    'ReLU',
    'ReLU(in)',
    'Sigmoid',
    'Softmax',
    'AvgPool2d',
    'BatchNorm',
    'Upsample (trilinear)',
]

_COMPUTE_UNARY_TF = [
    # computational layers
    'Conv2D',
    'Conv2D(bias)',
    'FC',
    'FC(bias)',
    'DepthwiseConv',
    'BatchNorm',
    'BatchNorm(bias)',
    'Relu',
    'Relu6',
    'AvgPooling',
]


# ------------------------------------------------------------------------------
#   Binary operations that require an input from the preceding layer and far above
# ------------------------------------------------------------------------------
_COMPUTE_BINARY_PT = [
    # Tensor operations
    'add',
    'sum',
    '* (multiplication)',
]

_COMPUTE_BINARY_TF = [
    # Tensor operations
    'TensorAdd',
]

# ------------------------------------------------------------------------------
# Misc. functions
# ------------------------------------------------------------------------------
def _check_computation(aname, alist):
    for each_name in alist:
        if each_name in aname: return True
    return False
