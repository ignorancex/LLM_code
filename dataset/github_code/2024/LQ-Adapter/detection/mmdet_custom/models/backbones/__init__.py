# Copyright (c) Shanghai AI Lab. All rights reserved.
from .beit_adapter import BEiTAdapter
from .uniperceiver_adapter import UniPerceiverAdapter
from .uniperceiver_adapter_classification import UniPerceiverAdapterClassifier
from .vit_adapter import ViTAdapter
from .vit_baseline import ViTBaseline
from .swin_adapter import SwinAdapter

__all__ = ['UniPerceiverAdapter', 'UniPerceiverAdapterClassifier', 'ViTAdapter', 'ViTBaseline', 'BEiTAdapter', 'SwinAdapter']
