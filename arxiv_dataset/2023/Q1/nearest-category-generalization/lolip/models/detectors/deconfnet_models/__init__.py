from .wideresnet import WideResNet
from .cnns import CNN002

class WRN_40_10(WideResNet):
    def __init__(self, depth=40, n_classes=10, widen_factor=10, dropRate=0.0):
        super().__init__(depth=depth, num_classes=n_classes,
                widen_factor=widen_factor, dropRate=dropRate)
