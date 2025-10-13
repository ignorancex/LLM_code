"""
Integrated Gradient Correlation (IGC) is a Python/PyTorch package that provides
a unique dataset-wise attribution method.

It is designed to improve the interpretability of deep neural networks at a
task-level, rather than an instance-level (as available attribution methods
generally do). For more theoretical details, please refer to the original
paper (http://arxiv.org/abs/2404.13910).

This package primarily focuses on a class computing IGC attributions for PyTorch
modules. Nonetheless, it also offers utilities to calculate simple gradients,
Integrated Gradients (IG), and some naive dataset-wise attribution methods.

Main Attribution Methods
------------------------

- igc (igc.igc)
    - Gradients
    - Integrated Gradients
    - Integrated Gradient Correlation

Other attribution methods
-------------------------

- igc.igac
    - Integrated Gradient Auto-Correlation

- igc.naive
    - IG mean and std
    - naive correlation
    - naive ttest

- igc.bsc
    - Baseline Shapley
    - Baseline Shapley Correlation

Utilities
---------

- igc.base
    - AbstractAttributionMethod
    - DataManager
"""

from .igc import Gradients, IntegratedGradients, IntGradCorr
