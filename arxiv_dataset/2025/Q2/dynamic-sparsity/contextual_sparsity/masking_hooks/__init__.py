# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from .dip import build_dip_masking_hooks, build_optimized_dip_masking_hooks
from .glu_pruning import build_glu_pruning_masking_hooks
from .partial_glu_pruning import build_partial_glu_pruning_masking_hooks
from .trained import build_original_turbosparse_hooks, build_predictor_masking_hooks
