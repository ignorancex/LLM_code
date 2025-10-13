from .wrappers import CrossEntropyLoss
from .ssc_loss import SSCLoss
from .l1loss import L1Loss
from .l2loss import L2Loss, L2MarginLoss

__all__ = [
    'CrossEntropyLoss', 'L1Loss', 'L2Loss', 'L2MarginLoss', 'SSCLoss'
]
