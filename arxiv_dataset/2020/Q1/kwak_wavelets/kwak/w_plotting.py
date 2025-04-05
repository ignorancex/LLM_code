"""
Plotting Functions
Functions that generate the bin scalograms of the coefficeints
and nsigma values.
"""
from __future__ import absolute_import
from . import plotting

__all__ = ['wScalogram', 'wScalogram_nsig',
           'nsigScalogram', 'nsigFixedRes']
           
wScalogram = plotting.wScalogram
wScalogram_nsig = plotting.wScalogram_nsig

nsigScalogram = plotting.nsigScalogram
nsigFixedRes = plotting.nsigFixedRes

