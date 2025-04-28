import numpy as np
import torch
import skimage as sk
from skimage import morphology as morph, measure
import os
import pandas as pd
import pickle
import PIL.Image as Image
import matplotlib.pyplot as plt


def extract_bounds(mask):
    '''
    Given a binary mask, return the top and bottom boundaries, 
    assuming the segmentation is fully-connected.
    '''
    # Stack of indexes where mask has predicted 1
    where_ones = np.vstack(np.where(mask.T)).T
    
    # Sort along horizontal axis and extract indexes where differences are
    sort_idxs = np.argwhere(np.diff(where_ones[:,0]))
    
    # Top and bottom bounds are either at these indexes or consecutive locations.
    bot_bounds = np.concatenate([where_ones[sort_idxs].squeeze(),
                                 where_ones[-1,np.newaxis]], axis=0)
    top_bounds = np.concatenate([where_ones[0,np.newaxis],
                                 where_ones[sort_idxs+1].squeeze()], axis=0)
    
    return (top_bounds, bot_bounds)


def interp_trace(traces, align=True):
    '''
    Quick helper function to make sure every trace is evaluated 
    across every x-value that it's length covers.
    '''
    new_traces = []
    for i in range(2):
        tr = traces[i]  
        min_x, max_x = (tr[:,0].min(), tr[:,0].max())
        x_grid = np.arange(min_x, max_x)
        y_interp = np.interp(x_grid, tr[:,0], tr[:,1]).astype(int)
        interp_trace = np.concatenate([x_grid.reshape(-1,1), y_interp.reshape(-1,1)], axis=1)
        new_traces.append(interp_trace)

    # Crop traces to make sure they are aligned
    if align:
        top, bot = new_traces
        h_idx=0
        top_stx, bot_stx = top[0,h_idx], bot[0,h_idx]
        common_st_idx = max(top[0,h_idx], bot[0,h_idx])
        common_en_idx = min(top[-1,h_idx], bot[-1,h_idx])
        shifted_top = top[common_st_idx-top_stx:common_en_idx-top_stx]
        shifted_bot = bot[common_st_idx-bot_stx:common_en_idx-bot_stx]
        new_traces = (shifted_top, shifted_bot)

    return tuple(new_traces)


def smart_crop(traces, check_idx=20, ythresh=1, align=True):
    '''
    Instead of defining an offset to check for and crop in utils.crop_trace(), which
    may depend on the size of the choroid itself, this checks to make sure that adjacent
    changes in the y-values of each trace are small, defined by ythresh.
    '''
    cropped_tr = []
    for i in range(2):
        _chor = traces[i]
        ends_l = np.argwhere(np.abs(np.diff(_chor[:check_idx,1])) > ythresh)
        ends_r = np.argwhere(np.abs(np.diff(_chor[-check_idx:,1])) > ythresh)
        if ends_r.shape[0] != 0:
            _chor = _chor[:-(check_idx-ends_r.min())]
        if ends_l.shape[0] != 0:
            _chor = _chor[ends_l.max()+1:]
        cropped_tr.append(_chor)

    return interp_trace(cropped_tr, align=align)


def select_largest_mask(binmask):
    '''
    Enforce connectivity of region segmentation
    '''
    # Look at which of the region has the largest area, and set all other regions to 0
    labels_mask = sk.measure.label(binmask)                       
    regions = sk.measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1

    return labels_mask


def get_trace(pred_mask, seg_thresh=0.5, crop_thresh=1, align=False):
    '''
    Helper function to extract traces from a prediction mask. 
    This thresholds the mask, selects the largest mask, extracts upper
    and lower bounds of the mask and crops any endpoints which aren't continuous.
    '''
    binmask = (pred_mask > seg_thresh).astype(int)
    binmask = select_largest_mask(binmask)
    traces = extract_bounds(binmask)
    traces = smart_crop(traces, ythresh=crop_thresh, align=align)
    return traces
        

def generate_imgmask(mask, thresh=None, cmap=0):
    '''
    Given a prediction mask Returns a plottable mask
    '''
    # Threshold
    pred_mask = mask.copy()
    if thresh is not None:
        pred_mask[pred_mask < thresh] = 0
        pred_mask[pred_mask >= thresh] = 1
    max_val = pred_mask.max()
    
    # Compute plottable cmap using transparency RGBA image.
    trans = max_val*((pred_mask > 0).astype(int)[...,np.newaxis])
    if cmap is not None:
        rgbmap = np.zeros((*mask.shape,3))
        rgbmap[...,cmap] = pred_mask
    else:
        rgbmap = np.transpose(3*[pred_mask], (1,2,0))
    pred_mask_plot = np.concatenate([rgbmap,trans], axis=-1)
    
    return pred_mask_plot