import os
import pandas as pd
import PIL.Image as Image
import numpy as np
import torch
import shutil
from pathlib import PosixPath, WindowsPath
from skimage import measure, segmentation, morphology, exposure
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import SimpleITK as sitk

import eyepy
from eyepy.core import utils as eyepy_utils
from eyepy.io.he import vol_reader

from octolyzer.measure.bscan.thickness_maps import grid
from octolyzer.measure.slo import feature_measurement
from octolyzer.segment.octseg import choroidalyzer_inference
from octolyzer.segment.sloseg import avo_inference


def load_volfile(vol_path, preprocess=False, custom_maps=[], logging=[], verbose=True):
    """
    Load and extract pixel and meta data from .vol file

    Returns OCT B-scan data, all relevant metadata and three
    versions of the corresponding IR-SLO image: A pain version,
    one with the fovea-centred B-scan acquisition location superimposed,
    and another with all B-scan acquisition locations superimposed.
    """
    fname = os.path.split(vol_path)[1]
    msg = f"Reading file {fname}..."
    logging.append(msg)
    if verbose:
        print(msg)
    
    # Catch whether .vol file is a peripapillary or macular scan. Other locations, i.e. radial "star-shaped" scans
    # currently not supported.
    try: 
        voldata = eyepy.import_heyex_vol(vol_path)
        scan_type = "Macular"

        # pixel data
        bscan_data = voldata.data / 255
        N_scans, M, N = bscan_data.shape
        fovea_slice_num = N_scans // 2 + 1
        
    except ValueError as msg:
        if len(msg.args) > 0 and msg.args[0] == "The EyeVolume object does not support scan pattern 2 (one Circular B-scan).":
            voldata = vol_reader.HeVolReader(vol_path)
            scan_type = "Peripapillary"

            # pixel data
            pixel_data = voldata.parsed_file.bscans[0].data
            bscan_data = (eyepy_utils.from_vol_intensity(pixel_data.copy()) / 255).reshape(1,*pixel_data.shape)
            N_scans, M, N = bscan_data.shape
            fovea_slice_num = None
        else:
            logging.append(msg)
            raise msg

    # slo data and metadata
    slo = voldata.localizer.data.astype(float) / 255
    slo_N = slo.shape[0]
    slo_metadict = voldata.localizer.meta.as_dict()
    slo_metadict["slo_resolution_px"] = slo_N
    slo_metadict["field_of_view_mm"] = slo_metadict["scale_x"] * slo_N
    
    # bscan metadata
    vol_metadata = voldata.meta.as_dict()
    eye = vol_metadata["laterality"]
    scale_z, scale_x, scale_y = vol_metadata["scale_z"], vol_metadata["scale_x"], vol_metadata["scale_y"]
    bscan_meta = vol_metadata["bscan_meta"]
    
    # Detect type of scan
    if scan_type == "Peripapillary":
        bscan_type = scan_type
        msg = f"Loaded a peripapillary (circular) B-scan."
        logging.append(msg)
        if verbose:
            print(msg)
    elif scan_type == "Macular" and scale_z != 0:
        bscan_type = "Ppole"
        msg = f"Loaded a posterior pole scan with {N_scans} B-scans."
        logging.append(msg)
        if verbose:
            print(msg)
    else:
        stp = bscan_meta[0]["start_pos"][0]
        enp = bscan_meta[0]["end_pos"][1]
        if np.allclose(stp,0,atol=1e-3):
            bscan_type = "H-line"
        elif np.allclose(enp,0,atol=1e-3):
            bscan_type = "V-line"
        else:
            bscan_type = "AV-line"
        msg = f"Loaded a single {bscan_type} B-scan."
        logging.append(msg)
        if verbose:
            print(msg)

    # Optional to try compensate for vessel shadows and brighten B-scans for improved
    # choroid visualisation. 
    # When bscan_type is "AV-line", we do not compensate for shadows
    if preprocess:
        if bscan_type != "AV-line":
            msg = "Preprocessing by compensating for vessel shadows and brightening choroid..."
            bscan_data = np.array([normalise_brightness(shadow_compensate(img)) for img in bscan_data])
        elif bscan_type == "AV-line":
            msg = "AV-line scans are not shadow-compensated and left raw for further processing."
        logging.append(msg)
        if verbose:
            print(msg)

    # retinal layers
    retinal_layers = voldata.layers
    N_rlayers = len(retinal_layers)
    if N_rlayers == 2:
        msg = ".vol file only has inner and outer retinal layer segmentations."
    elif N_rlayers == 3:
        msg = "peripapillary .vol file contains ILM, RNFL and BM layer segmentations."
    else:
        msg = ".vol file contains all retinal layer segmentations."
    logging.append(msg)
    if verbose:
        print(msg)

    # Collect all available retinal layer keys
    try:
        msg = "Processing retinal layer segmentations..."
        logging.append(msg)
        if verbose:
            print(msg)
        layer_keys = list()
        x_grid_all = np.repeat(np.arange(N).reshape(-1,1), N_scans, axis=1).T

        # Dealing with retinal layer segmentations from  peripapillary scans
        if bscan_type == "Peripapillary":
            layer_keys = ["ILM", "BM", "RNFL"]
            N_rlayers = len(layer_keys)
            layer_key_pairwise = ["ILM_RNFL", "ILM_BM", "RNFL_BM"]
            retinal_layers = {key:val for key,val in zip(layer_keys, retinal_layers)}
            #return retinal_layers

        # Dealing with retinal layer segmentations from macular scans
        else:
            retinal_layers = {key:val.data for key,val in retinal_layers.items()}
            for key in retinal_layers.keys():
                if not np.all(np.isnan(retinal_layers[key])):
                    layer_keys.append(key) 
            layer_keys = layer_keys[:1] + layer_keys[2:] + ["BM"]
            N_rlayers = len(layer_keys)
            layer_key_pairwise = [f"{key1}_{key2}" for key1,key2 in zip(layer_keys[:-1], layer_keys[1:])]
        
            # Add custom thickness maps
            # by default we will always provide whole retina, inner and outer retinal layers, removing any duplicates
            custom_maps += ["ILM_ELM", "ELM_BM"]
            custom_maps = list(set(custom_maps))
            if N_rlayers > 2:
                layer_key_pairwise.append("ILM_BM")
                if len(custom_maps) > 0:
                    for key_pair in custom_maps:
                        layer_key_pairwise.append(key_pair)

        # Collect retinal layer segmentations
        layer_pairwise = {}
        for key in layer_key_pairwise:
            key1, key2 = key.split("_")
            lyr1 = np.concatenate([x_grid_all[...,np.newaxis],
                                retinal_layers[key1][...,np.newaxis]], axis=-1)
            lyr2 = np.concatenate([x_grid_all[...,np.newaxis],
                                retinal_layers[key2][...,np.newaxis]], axis=-1)
            lyr12_xy_all = np.concatenate([lyr1[:,np.newaxis], lyr2[:,np.newaxis]], axis=1)
            layer_pairwise[key] = [remove_nans(tr) for tr in lyr12_xy_all]   

        # Check to make sure all B-scans have inner retinal layer segmentations. If not,
        # only return ILM and BM layers - this does not apply for peripapillary scans, where
        # only three layers are segmented.
        if N_rlayers > 2 and bscan_type != "Peripapillary":    
            check_layer = "ILM_RNFL"
            check_segs = [trace.shape[1]==0 for trace in layer_pairwise[check_layer]]
            N_missing = int(np.sum(check_segs))
            if N_missing > 0:
                msg = f"WARNING: Found {N_rlayers} retinal layer segmentations, but {N_missing}/{N_scans} B-scans have not been fully segmented for all retinal layers."
                logging.append(msg)
                if verbose:
                    print(msg)    
            if N_missing > N_scans//2:
                msg = f"""Over half of the B-scans are missing inner retinal layer segmentations.
    Falling back to default state of only analysing whole retina, and removing inner retinal layers in output.
    Please segment inner retinal layers for remaining scans to analyse all retinal layers."""
                logging.append(msg)
                if verbose:
                    print(msg)
                newlayer_pairwise = {}
                newlayer_pairwise["ILM_BM"] = layer_pairwise["ILM_BM"]
                layer_pairwise = newlayer_pairwise
                N_rlayers = 2        
        msg = f"Found {N_rlayers} valid retinal layer segmentations for all B-scans."
        logging.append(msg)
        if verbose:
            print(msg)
    except Exception as e:
        message = f"An exception of type {type(e).__name__} occurred. Error description:\n{e}"
        tell_user = "Unexpected error locating retinal layer segmentations. Please check B-scan for quality or metadata. Ignoring retinal layers."
        logging.extend([message, tell_user])
        if verbose:
            print(message)
            print(tell_user)
        N_rlayers = 0
        layer_key_pairwise = []
        layer_pairwise = {}

    # Construct slo-acquisition image and extract quality of B-scan    
    msg = "Accessing IR-SLO and organising metadata..."
    logging.append(msg)
    if verbose:
        print(msg)
    all_mm_points = []
    all_quality = []
    for m in bscan_meta:
        all_quality.append(m["quality"])
        st = m["start_pos"]
        en = m["end_pos"]
        point = np.array([st, en])
        all_mm_points.append(point)

    # Only relevant for Ppole data
    quality_mu = np.mean(all_quality)
    quality_sig = np.std(all_quality)
    
    # Convert start and end B-scan locations from mm to pixel
    all_px_points = []
    for point in all_mm_points:
        all_px_points.append(slo_N * point / slo_metadict["field_of_view_mm"])

    # return all_px_points

    # Draw the acquisition locations onto the SLO
    slo_at_fovea = np.concatenate(3*[slo[...,np.newaxis]], axis=-1)
    slo_acq = slo_at_fovea.copy()

    # For peripapillary scans, we draw a circular ROI
    if bscan_type == "Peripapillary":
        peripapillary_coords = all_px_points[0].astype(int)

        OD_edge, OD_center = peripapillary_coords
        circular_radius = np.abs(OD_center[0] - OD_edge[0])
        circular_mask = grid.create_circular_mask(img_shape=(slo_N,slo_N), 
                                     center=OD_center, 
                                     radius=circular_radius)
        circular_bnd_mask = segmentation.find_boundaries(circular_mask)
        slo_acq[circular_bnd_mask,:] = 0
        slo_acq[circular_bnd_mask,1] = 1
        slo_at_fovea = slo_acq.copy()
        slo_metadict["stxy_coord"] = f"{OD_edge[0]},{OD_edge[1]}"
        slo_metadict["acquisition_radius_px"] = circular_radius
        slo_metadict["acquisition_radius_mm"] = np.round(circular_radius*slo_metadict["scale_x"],2)
        slo_metadict["acquisition_optic_disc_center_x"] = OD_center[0]
        slo_metadict["acquisition_optic_disc_center_y"] = OD_center[1]

    else:
        # For macular scans, we generate line for each B-scan location and 
        # superimpose acquisition line onto copied SLO. Create one with all
        #  acquisition lines, and one with only
        # the fovea. slo_at_fov only used when N_scans > 1
        for idx, point in enumerate(all_px_points):
            x_idx, y_idx = [[1,0], [0,1]][bscan_type != "V-line"]
            X, y = point[:,x_idx].reshape(-1,1), point[:,y_idx]
            linmod = LinearRegression().fit(X, y)
            x_grid = np.linspace(X[0,0], X[1,0], 800).astype(int)
            x_grid = x_grid[(x_grid < slo_N) & (x_grid >= 0)]
            y_grid = linmod.predict(x_grid.reshape(-1,1)).astype(int)
            x_grid = x_grid[y_grid < slo_N]
            y_grid = y_grid[y_grid < slo_N]
            for (x,y) in zip(x_grid, y_grid):
                x_idx, y_idx = [[y,x], [x,y]][bscan_type != "V-line"]
                slo_acq[y_idx, x_idx, :] = 0
                slo_acq[y_idx, x_idx, 1] = 1
                if (idx+1) == fovea_slice_num:
                    slo_at_fovea[y_idx, x_idx, :] = 0
                    slo_at_fovea[y_idx, x_idx, 1] = 1
    # Create DataFrame of metadata
    bscan_metadict = {}
    bscan_metadict["Filename"] = fname
    if eye  == 'OD':
        bscan_metadict["eye"] = 'Right'
    elif eye == 'OS':
        bscan_metadict["eye"] = 'Left'
    bscan_metadict["bscan_type"] = bscan_type
    bscan_metadict["bscan_resolution_x"] = N
    bscan_metadict["bscan_resolution_y"] = M
    bscan_metadict["bscan_scale_z"] = 1e3*scale_z
    bscan_metadict["bscan_scale_x"] = 1e3*scale_x
    bscan_metadict["bscan_scale_y"] = 1e3*scale_y
    bscan_metadict["scale_units"] = "microns_per_pixel"
    bscan_metadict["avg_quality"] = quality_mu
    bscan_metadict["retinal_layers_N"] = N_rlayers

    # Remove duplicates: store scales as microns-per-pixel, laterality=eye
    slo_metadict["slo_scale_xy"] = 1e3*slo_metadict["scale_x"]
    for key in ["laterality", "scale_x", "scale_y", "scale_unit"]:
        del slo_metadict[key]
    slo_metadict["location"] = scan_type.lower()
    slo_metadict["field_size_degrees"] = slo_metadict.pop("field_size")
    slo_metadict["slo_modality"] = slo_metadict.pop("modality")
        
    # Combine metadata and return with data
    metadata = {**bscan_metadict, **slo_metadict}
    msg = "Done!"
    logging.append(msg)
    if verbose:
        print(msg)
        
    return bscan_data, metadata, (slo, slo_acq, slo_at_fovea), layer_pairwise, logging


def load_img(path, ycutoff=0, xcutoff=0, pad=False, pad_factor=32):
    '''
    Helper function to load and normalize an image, and optionally
    pad to have dimensions divisible by pad_factor
    '''
    img = np.array(Image.open(path))[ycutoff:, xcutoff:]/255.0
    if pad:
        ndim = img.ndim
        M, N = img.shape[:2]
        pad_M = (pad_factor - M%pad_factor) % pad_factor
        pad_N = (pad_factor - N%pad_factor) % pad_factor
    
        # Assuming third color channel is last axis
        if ndim == 2:
            return np.pad(img, ((0, pad_M), (0, pad_N)))
        else: 
            return np.pad(img, ((0, pad_M), (0, pad_N), (0,0)))
    else:
        return img


# Plotting data


def plot_img(img_data, traces=None, cmap=None, fovea=None, save_path=None, 
             fname=None, sidebyside=False, rnfl=False, close=False, 
             trace_kwds={'c':"r", 'linestyle':"--", 'linewidth':2}):
    '''
    Helper function to plot the result - plot the image, traces, colourmap, etc.
    '''
    img = img_data.copy().astype(np.float64)
    img -= img.min()
    img /= img.max()
    M, N = img.shape
    
    if rnfl:
        figsize=(15,6)
    else:
        figsize=(6,6)

    if sidebyside:
        if rnfl:
            figsize = (2*figsize[0], figsize[1])
        else:
            figsize = (figsize[0], 2*figsize[1])
    
    if sidebyside:
        if rnfl:
            fig, (ax0, ax) = plt.subplots(1,2,figsize=figsize)
        else:
            fig, (ax0, ax) = plt.subplots(2,1,figsize=figsize)
        ax0.imshow(img, cmap="gray", zorder=1, vmin=0, vmax=1)
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])
        ax0.tick_params(axis='both', which='both', bottom=False,left=False, labelbottom=False)
    else:
        fig, ax = plt.subplots(1,1,figsize=figsize)
        
    ax.imshow(img, cmap="gray", zorder=1, vmin=0, vmax=1)
    fontsize=16
    if traces is not None:
        if len(traces) == 2:
            for tr in traces:
                 ax.plot(tr[:,0], tr[:,1], zorder=3, **trace_kwds)
        else:
            ax.plot(traces[:,0], traces[:,1], zorder=3, **trace_kwds)

    if cmap is not None:
        cmap_data = cmap.copy().astype(np.float64)
        cmap_data -= cmap_data.min()
        cmap_data /= cmap_data.max()
        ax.imshow(cmap_data, alpha=0.5, zorder=2)
    if fname is not None:
        ax.set_title(fname, fontsize=15)

    if fovea is not None:
        ax.scatter(fovea[0], fovea[1], color="r", edgecolors=(0,0,0), marker="X", s=50, linewidth=1)
            
    ax.set_axis_off()
    fig.tight_layout(pad = 0)
    if save_path is not None and fname is not None:
        ax.set_title(None)
        fig.savefig(os.path.join(save_path, f"{fname}.png"), bbox_inches="tight", pad_inches=0)

    if close:
        plt.close()
    else:
        return fig



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


# Post-processing segmentation

def remove_nans(trace):
    if trace.ndim > 2:
        return trace[:,~np.isnan(trace[...,1]).any(axis=0)].astype(np.int64)
    else:
        return trace[~np.isnan(trace[:,1])].astype(np.int64)


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


def select_largest_mask(binmask):
    '''
    Enforce connectivity of region segmentation
    '''
    # Look at which of the region has the largest area, and set all other regions to 0
    labels_mask = measure.label(binmask)                       
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1

    return labels_mask


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



def get_trace(pred_mask, threshold=0.5, align=False):
    '''
    Helper function to extract traces from a prediction mask. 
    This thresholds the mask, selects the largest mask, extracts upper
    and lower bounds of the mask and crops any endpoints which aren't continuous.
    '''
    binmask = (pred_mask > threshold).astype(int)
    binmask = select_largest_mask(binmask)
    traces = extract_bounds(binmask)
    traces = smart_crop(traces, align=align)
    return traces



def rebuild_mask(traces, img_shape=None):
    '''
    Rebuild binary mask from choroid traces
    '''
    # Work out extremal coordinates of traces
    top_chor, bot_chor = interp_trace(traces)
    common_st_idx = np.maximum(top_chor[0,0], bot_chor[0,0])
    common_en_idx = np.minimum(top_chor[-1,0], bot_chor[-1,0])
    top_idx = top_chor[:,1].min()
    bot_idx = bot_chor[:,1].max()

    if img_shape is not None:
        binmask = np.zeros(img_shape)
    else:
        binmask = np.zeros((bot_idx+100, common_en_idx+100))

    for i in range(common_st_idx, common_en_idx):
        top_i = top_chor[i-common_st_idx,1]
        bot_i = bot_chor[i-common_st_idx,1]
        binmask[top_i:bot_i,i] = 1

    return binmask



def visualise_vessels(masks, vessels_binmap):
    '''
    Using the binary vessel map, create 4d colour map which colours indvidual vessels and adds a fourth map of 
    transparency for superimposing onto a figure.

    INPUTS:
    -----------------
        masks (ndarray) : Array of binary maps of individual vessels.

        vessels_binmap (2darray) : Binary map storing all vessels.
    '''
    img_shape = vessels_binmap.shape
    vessel_colourmap = np.zeros((*img_shape, 4))
    binmap3D = np.zeros((*img_shape, 3))
    for j, binmap in enumerate(masks):
        np.random.seed(j)
        color = (np.random.choice(range(256), size=3)/255)
        binmap3D += np.concatenate([c * binmap[:,:,np.newaxis] for c in color], axis=-1)
    vessel_colourmap[:,:,:-1] = binmap3D
    vessel_colourmap[:,:,-1] = vessels_binmap

    return vessel_colourmap


def generate_vesselmask(binmap):
    '''
    Using skimage.measure, extract individual vessel masks and other
    interesting metrics
    '''
    labels_mask = measure.label(binmap.astype(int))         
    regions = measure.regionprops(labels_mask)
    vessel_masks = []
    for r in regions:
        sing_vessel_mask = np.zeros_like(binmap)
        v_coords = r.coords
        sing_vessel_mask[v_coords[:,0], v_coords[:,1]] = 1
        vessel_masks.append(sing_vessel_mask)
    cmap = visualise_vessels(vessel_masks, binmap)

    return cmap




def normalise(img, 
              minmax_val=(0,1), 
              astyp=np.float64):
    '''
    Normalise image between minmax_val.

    INPUTS:
    ----------------
        img (np.array, dtype=?) : Input image of some data type.

        minmax_val (tuple) : Tuple storing minimum and maximum value to normalise image with.

        astyp (data type) : What data type to store normalised image.
    
    RETURNS:
    ----------------
        img (np.array, dtype=astyp) : Normalised image in minmax_val.
    '''
    # Extract minimum and maximum values
    min_val, max_val = minmax_val

    # Convert to float type to perform [0, 1] normalisation
    img = img.astype(np.float64)

    # Normalise to [0, 1]
    img -= img.min()
    img /= img.max()

    # Rescale to max_val and output as specified data type
    img *= (max_val - min_val)
    img += min_val

    return img.astype(astyp)


def normalise_brightness(img, target_mean=0.2):
    """
    Gamma transform image to improve brightness
    """
    # Adjust image to mean brightness of 0.25
    gamma = np.log(target_mean) / np.log(img.mean())
    img = exposure.adjust_gamma(img, gamma=gamma)
    img = normalise(img)

    return img


def shadow_compensate(img, gamma=1, win_size=75, plot=False):
    """Using moving averages, compensate for vessel shadowing by
    scaling A-scan pixel intensities to average out drop in signal caused by shadowing.
    Gamma is used here as an implicit enhancer too."""
    # If RGB, select first channel on the assumption it is an OCT B-scan 
    # and is therefore grayscale. Channels in last dimension by assumption.
    if img.ndim > 2:
        img = img[...,0]
    
    # Remove any black columns either side of image
    comp_idx_l = img[:,:img.shape[1]//2].mean(axis=0) != 0
    comp_idx_r = img[:,img.shape[1]//2:].mean(axis=0) != 0
    img_crop = img[:,np.concatenate([comp_idx_l, comp_idx_r], axis=0)]

    # Energy of each pixel of the A-line, where gamma > 1 for implicit image enhancement
    # by darkening the image
    E_ij = exposure.adjust_gamma(img_crop, gamma=gamma)

    # Total energy of each A-scan is the sum across their rows
    E_i = E_ij.sum(axis=0)

    # Centred, moving average according to win_size, pad edges of average with original signal
    E_i_smooth = pd.Series(E_i).rolling(win_size, center=True).mean().values
    E_i_smooth[:win_size//2], E_i_smooth[-win_size//2:] = E_i[:win_size//2], E_i[-win_size//2:]

    # Compensation is linear scale made to individual energy levels to match total energy per A-scan
    # with its moving average value
    E_comp = (E_i_smooth / E_i)

    # If plotting energy levels
    if plot:
        fig, (ax, ax1) = plt.subplots(1,2,figsize=(14,7))
        ax.set_xlabel("A-scan (column) index", fontsize=14)
        ax.set_ylabel(rf"Energy level (I$^\gamma$)", fontsize=14)
        ax.plot(np.arange(E_i.shape[0]), E_i, c="b", linewidth=2)
        ax.plot(np.arange(E_i_smooth.shape[0]), E_i_smooth, c="r", linewidth=2)
        ax1.set_xlabel("A-scan (column) index", fontsize=14)
        ax1.set_ylabel(rf"Correction factor (I$^\gamma$ / MA(I$^\gamma$)", fontsize=14)
        ax1.plot(np.arange(E_comp.shape[0]), E_comp, c="r", linewidth=2)
        ax.set_title(rf"Energy per A-scan ($\gamma$ = {gamma})", fontsize=18)
        ax1.set_title(rf"Correction per A-scan ($\gamma$ = {gamma})", fontsize=18)

    # Reshape to apply element-wise to original image and darken
    E_comp_arr = E_comp.reshape(1,-1).repeat(img_crop.shape[0], axis=0)
    output = E_ij*E_comp_arr
    #output = (img_crop*E_comp_arr)**gamma

    # Put back any black columns either side of image
    if (~comp_idx_l).sum() > 0:
        output = np.pad(output, ((0,0),((~comp_idx_l).sum(),0)))
    if (~comp_idx_r).sum() > 0:
        output = np.pad(output, ((0,0),(0,(~comp_idx_r).sum())))

    # Plot original and compensated versions
    if plot:
        fig, (ax, ax1) = plt.subplots(1,2,figsize=(18,7))
        ax.imshow(img, cmap="gray")
        ax1.imshow(output, cmap="gray")
        ax.set_axis_off()
        ax1.set_axis_off()
        fig.tight_layout()

    # Return normalised output
    output = normalise(output)

    return output



def flatten_dict(nested_dict):
    '''
    Recursive flattening of a dictionary of dictionaries.
    '''
    res = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        res[()] = nested_dict
    return res


def nested_dict_to_df(values_dict):
    '''
    Nested dictionary is flattened and converted into an index-wise, multi-level Pandas DataFrame
    '''
    flat_dict = flatten_dict(values_dict)
    df = pd.DataFrame.from_dict(flat_dict, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.unstack(level=-1)
    df.columns = df.columns.map("{0[1]}".format)
    return df



def align_peripapillary_data(metadata, fovea_at_slo, slo_acq, slo_avimout, fname, save_path, save=True):
    """
    In order to measure the thickness profile around the optic nerve head with reference
    to the fovea, we need to align the thickness profile, i.e. the B-scan.
    """
    # Extract Optic disc mask and acquisition line boundary, dilate circ mask for plotting
    od_mask = slo_avimout[...,1]
    N = od_mask.shape[1]
    circ_mask = slo_acq[...,1] == 1
    circ_mask_dilate = morphology.dilation(circ_mask, footprint=morphology.disk(radius=2))
    
    # Extract user-specified OD center, and binary mask-based OD center and setup
    # measuring overlap
    od_user_center = np.array([metadata["acquisition_optic_disc_center_x"], 
                               metadata["acquisition_optic_disc_center_y"]]).astype(int)
    od_mask_center = measure.centroid(od_mask.astype(bool))[[1,0]]

    # Overlap index measures distance between user-specified and model-specified (acting as ground
    # truth) and normalises that according to the radius of the mask's optic disc binary mask. The 
    # radius is deduced by the minor axis lenth of the optic disc mask.
    props = measure.regionprops(measure.label(od_mask))[0]
    od_diameter = int((props.axis_minor_length + props.axis_major_length)/2)
    od_centers_both = np.concatenate([od_mask_center[np.newaxis], od_user_center[np.newaxis]], axis=0)
    center_distance = feature_measurement._curve_length(od_centers_both[:,0], od_centers_both[:,1])
    offset_ratio = np.round(center_distance / od_diameter,4)

    # Work out reference line between fovea and user-specified optic disc center. 
    Xy =  np.concatenate([od_user_center[np.newaxis], fovea_at_slo[np.newaxis]], axis=0)
    linmod = LinearRegression().fit(Xy[:,0].reshape(-1,1), Xy[:,1])

    # Sample line coordinates
    x_grid = np.linspace(min(Xy[0,0], Xy[1,0]), max(Xy[0,0], Xy[1,0]), 1000).astype(int)
    y_grid = linmod.predict(x_grid.reshape(-1,1)).astype(int)

    # Extract coordinates to compare directly
    line_pts = np.array([x_grid,y_grid]).T
    circ_pts = np.argwhere(circ_mask == 1)[:,[1,0]]

    # Elementwise comparison to get pixel coord along line of acquisition colinear with fovea and OD centre
    dist_mins = []
    for pt in line_pts:
        dist_mins.append(((circ_pts - pt)**2).sum(axis=1).min())
    dist_mins = np.array(dist_mins)

    # Intersection of reference line and circular acquisition line is where the temporal
    # midpoint
    temp_mid_idx = ((circ_pts - line_pts[dist_mins.argmin()])**2).sum(axis=1).argmin()
    temporal_mid = circ_pts[temp_mid_idx]

    # Old, simple version before 22/10/2024, replaced by lines 790 -- 807
    # x_grid = np.arange(min(Xy[0,0], Xy[1,0]), max(Xy[0,0], Xy[1,0])).astype(int)
    # temp_mid_idx = np.argmax(circ_mask[(y_grid, x_grid)] == 1)
    # temporal_mid = np.array((x_grid[temp_mid_idx], y_grid[temp_mid_idx])).reshape(-1)

    # For visualising intersection point detection
    # plt.imshow(circ_mask)
    # plt.scatter(od_user_center[0], od_user_center[1])
    # plt.scatter(od_mask_center[0], od_mask_center[1])
    # plt.plot(Xy[:,0], Xy[:,1])
    # plt.plot(x_grid, y_grid, linestyle='--')
    # plt.plot(line_pts[:,0], line_pts[:,1])
    # plt.scatter(circ_pts[:,0], circ_pts[:,1])
    # plt.scatter(temporal_mid[0], temporal_mid[1])

    # Work out where this temporal midpoint is along the peripapillary OCT B-can (A-scan
    # location). We use this value to align our thickness profile to, i.e. shift
    # the peripapillary OCT B-scan in order for the central A-scan to be where the middle
    # of the Temporal subregion is (temporal midpoint)
    od_user_radius = metadata["acquisition_radius_px"]
    circ_coords = np.array([(od_user_radius*np.cos(theta)+od_user_center[0],
                             od_user_radius*np.sin(theta)+od_user_center[1]) 
                            for theta in np.linspace(0, 2*np.pi, N)])
    ascan_idx_temp0 = np.sum(np.square(circ_coords - temporal_mid), axis=1).argmin()

    # Plot the result onto SLO 
    od_user_edge = np.array(metadata["stxy_coord"].split(",")).astype(int)
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    ax.imshow(slo_acq)
    ax.imshow(slo_avimout)
    ax.imshow(generate_imgmask(circ_mask_dilate,cmap=1))
    ax.scatter(od_mask_center[0], od_mask_center[1], zorder=3, s=200, c="green", edgecolors=(0,0,0), marker="X", label="True OD center")
    ax.scatter(od_user_center[0], od_user_center[1], zorder=3, s=200, c="b", marker="X", edgecolors=(0,0,0), label="User-specified OD center")
    ax.scatter(fovea_at_slo[0], fovea_at_slo[1], label="Fovea", c="orange", 
               marker="X", edgecolors=(0,0,0), s=200, zorder=5)
    ax.plot(x_grid, y_grid, c="m", linewidth=3, label="Reference line (Fovea -> User OD Center)")
    ax.scatter(temporal_mid[0], temporal_mid[1], marker="X", c="r", 
               edgecolors=(0,0,0), s=200, label="Temporal center of peripapillary thickness", zorder=5)
    ax.plot([-1],[-1],c="lime",label="Line of acquisition")
    
    legend_loc = "lower right"
    if metadata["eye"] == "Right":
        legend_loc = "lower left" 
    ax.legend(loc=legend_loc, fontsize=16)
    ax.axis([0,N,N,0])
    ax.set_axis_off()
    ax.set_title(f"Overlap(user,true) of OD center: {np.round(100*offset_ratio,2)}% of OD diameter", fontsize=20)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(save_path, f"{fname}_peripapillary_alignment.png"), bbox_inches="tight")
    plt.close()

    return od_mask_center, offset_ratio, ascan_idx_temp0



def compute_opticdisc_radius(fovea, od_centre, od_mask):
    """
    Work out optic disc radius in pixels, according to it's position relative to the fovea.

    This is deprecated, and a simpler version averages and minor and major axis lengths.
    """
    # Extract Optic disc mask and acquisition line boundary,
    od_mask_props = measure.regionprops(measure.label(od_mask))[0]
    #od_mask_radius = od_mask_props.axis_minor_length/2 # naive radius, without account for orientation with fovea
    #od_mask_radius = od_mask_props.axis_major_length/2
    od_boundary = segmentation.find_boundaries(od_mask)

    # Work out reference line between fovea and  optic disc center. 
    Xy =  np.concatenate([od_centre[np.newaxis], fovea[np.newaxis]], axis=0)
    linmod = LinearRegression().fit(Xy[:,0].reshape(-1,1), Xy[:,1])
    x_grid = np.arange(min(Xy[0,0], Xy[1,0]), max(Xy[0,0], Xy[1,0])).astype(int)
    y_grid = linmod.predict(x_grid.reshape(-1,1)).astype(int)
    
    # Intersection of reference line and optic disc boundary
    intersection_idx = np.argwhere(od_boundary[(y_grid, x_grid)] == 1)[0]
    od_intersection = np.array((x_grid[intersection_idx], 
                                y_grid[intersection_idx])).reshape(1,-1)

    # Now we can work out the optic disc radius, according to it's position with the fovea
    od_bounds = np.concatenate([od_centre.reshape(1,-1), od_intersection], axis=0)
    od_radius = feature_measurement._curve_length(od_bounds[:,0], od_bounds[:,1])

    plot_info = (od_intersection, (x_grid, y_grid), intersection_idx)

    return np.round(od_radius).astype(int), plot_info



def _process_opticdisc(od_mask):
    """
    Work out optic disc radius in pixels, according to it's position relative to the fovea.
    """
    # Extract Optic disc radius and OD boundary if detected
    try:
        od_mask_props = measure.regionprops(measure.label(od_mask))[0]
    except:
        return None, np.zeros_like(od_mask)
    od_radius = int((od_mask_props.axis_minor_length + od_mask_props.axis_major_length)/4)
    od_boundary = segmentation.find_boundaries(od_mask)

    return od_radius, od_boundary



def _get_fovea(rvfmasks, foveas, N_scans=31, scan_type="Ppole", logging=[]):
    '''
    Helper function to resolve fovea coordinate if prediction map below default threshold,
    or if volume scan acquisition not centred at fovea
    '''
    if N_scans == 1:
        fovea_slice_num = 0
        fovea = foveas[0]
        if fovea.sum() == 0:
            fmask = rvfmasks[0][-1]
            if scan_type != "AV-line":
                fovea = choroidalyzer_inference.process_fovea_prediction(torch.tensor(fmask).unsqueeze(0))[0]
                msg = "\n\nPrediction threshold for fovea too high. Detecting from raw probabilities. Please check output for correct fovea alignment."
            else:
                fovea = None
                msg = "\n\nAV-line scan assumed not to be fovea-centred."
            logging.append(msg)
            print(msg)

    else:
        fovea_slice_num = N_scans//2 + 1
        fovea = foveas[fovea_slice_num]
        if fovea.sum() == 0:
            msg = "\n\nPrediction threshold for fovea too high or non-centred Ppole scan acquisition."
            logging.append(msg)
            print(msg)
            foveas_arr = np.array(foveas)
            fov_idx = np.where(foveas_arr[:,0]>0)[0]

            # If default fovea-centred B-scan prediction is at origin, work out highest score from fovea masks
            # which have detected a fovea coordinate
            if fov_idx.shape[0] > 0:
                fov_scores = []
                fov_preds = []
                for idx in fov_idx:
                    fmask = rvfmasks[idx][-1]
                    fov_pred = choroidalyzer_inference.process_fovea_prediction(torch.tensor(fmask).unsqueeze(0))[0]
                    fov_scores.append(fmask[fov_pred[1], fov_pred[0]])
                    fov_preds.append(fov_pred)
                fovea_slice_num = fov_idx[np.argmax(fov_scores)]+1
                #fovea = fov_preds[np.argmax(fov_scores)]
                msg = f"Potentially detected fovea-centred B-scan in Ppole at slice {fovea_slice_num}/{N_scans}."
                logging.append(msg)
                print(msg)
            fmask = rvfmasks[fovea_slice_num][-1]
            fovea = choroidalyzer_inference.process_fovea_prediction(torch.tensor(fmask).unsqueeze(0))[0]
            msg = "Detecting from raw probabilities. Please check output for correct fovea alignment."
            logging.append(msg)
            print(msg)

    return fovea_slice_num, fovea, logging


def load_trace(df, layers=['CHORupper', 'CHORlower']):
    '''
    Load in paired layer segmentations from OCTolyzer's output DataFrame.
    '''
    trace = df[df.layer.isin(layers)]
    trace = pd.melt(trace, id_vars='layer', value_name='y', var_name='x')
    traces = [trace[trace.layer==lyr].iloc[:,1:].values.astype(int) for lyr in layers]
    traces = tuple([tr[tr[:,1]!=0] for tr in traces])
    return interp_trace(traces)



def load_annotation(path, key=None, raw=False, binary=False):
    """Load in .nii.gz file and output region and vessel masks"""
    
    # Read the .nii image containing thevsegmentations
    sitk_t1 = sitk.ReadImage(path)

    if key is None:
        a_i = 191
        od_i = 255
        v_i = 127
    else:
        a_i, v_i, od_i = key
        
    # and access the numpy array, saved as (1, N, N)
    segmentations = sitk.GetArrayFromImage(sitk_t1)[0]

    # returning raw segmentation map
    if raw:
        return segmentations

    # if binary, only return vessels with label 255
    if binary:
        return (segmentations == od_i).astype(int)

    # for artery-vein-optic disc map
    artery = (segmentations == a_i)
    OD = (segmentations == od_i)
    vein = (segmentations == v_i)
    mask = (segmentations > 0)
    cmap = np.concatenate([artery[...,np.newaxis], 
                           OD[...,np.newaxis],
                           vein[...,np.newaxis], 
                           mask[...,np.newaxis]], axis=-1).astype(int)
    
    
    return cmap


def plot_composite_volume(bscan_data, vmasks, fovea_slice_num, layer_pairwise, reshape_idx, analyse_choroid, fname, save_path):
    '''
    Plot high-res, composite image of all volume B-scans (apart from fovea-centred one) stitched together with
    segmentations overlaid.
    '''
    # Get layer names
    pairwise_keys = list(layer_pairwise.keys())
    layer_keys = list(set(pd.DataFrame(pairwise_keys).reset_index(drop=True)[0].str.split("_", expand=True).values.flatten()))
    
    # Organise B-scan data and choroid vessel maps
    img_shape = bscan_data.shape[-2:]
    M, N = img_shape
    bscan_list = list(bscan_data.copy())
    bscan_list.pop(fovea_slice_num)
    bscan_arr = np.array(bscan_list)
    bscan_arr = bscan_arr.reshape(*reshape_idx,*img_shape)
    bscan_hstacks = []

    if analyse_choroid:
        vmasks_list = list(vmasks.copy())
        vmasks_list.pop(fovea_slice_num)
        vmasks_arr = np.asarray(vmasks_list)
        vmasks_arr = vmasks_arr.reshape(*reshape_idx,M,N)
        vmask_hstacks = []

    # Stack B-scans and vessel maps horizontally
    for i in range(reshape_idx[0]):
        bscan_hstacks.append(np.hstack(bscan_arr[i]))
        if analyse_choroid:
            vmask_hstacks.append(np.hstack(vmasks_arr[i]))

    # Stack B-scans and vessel maps vertically
    bscan_stacked = np.vstack(bscan_hstacks)
    if analyse_choroid:
        vmask_stacked = np.vstack(vmask_hstacks)
        all_vcmap = np.concatenate([vmask_stacked[...,np.newaxis]] 
                    + 2*[np.zeros_like(vmask_stacked)[...,np.newaxis]] 
                    + [vmask_stacked[...,np.newaxis] > 0.01], axis=-1)

    # figure to be saved out at same dimensions as stacked array
    h,w = bscan_stacked.shape
    np.random.seed(0)
    COLORS = {key:np.random.randint(255, size=3)/255 for key in layer_keys}
    fig, ax = plt.subplots(1,1,figsize=(w/1000, h/1000), dpi=100)
    ax.set_axis_off()
    ax.imshow(bscan_stacked, cmap='gray')

    # add all traces
    for (i, j) in np.ndindex(reshape_idx):
        layer_keys_copied = layer_keys.copy()
        for key, traces in layer_pairwise.items():
            tr = traces.copy()
            tr.pop(fovea_slice_num)
            for (k, t) in zip(key.split("_"), tr[reshape_idx[1]*i + j]):
                if k in layer_keys_copied:
                    c = COLORS[k]
                    ax.plot(t[:,0]+j*N,t[:,1]+i*M, label='_ignore', color=c, zorder=2, linewidth=0.175)
                    layer_keys_copied.remove(k)

    # add vessel maps  
    if analyse_choroid:
        ax.imshow(all_vcmap, alpha=0.5)
    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(save_path, f"{fname}_volume_octseg.png"), dpi=1000)
    plt.close()


def print_error(e, verbose=True):
    '''
    If robust_run is 1 and an unexpected error occurs, this will be printed out and also saved to the log.

    A detailed explanation of the error found.
    '''
    message = f"\nAn exception of type {type(e).__name__} occurred. Error description:\n{str(e)}\n"
    if verbose:
        print(message)
    trace = ["Full traceback:\n"]
    if verbose:
        print(trace[0])
    tb = e.__traceback__
    tb_i = 1
    while tb is not None:
        tb_fname = tb.tb_frame.f_code.co_filename
        tb_func = tb.tb_frame.f_code.co_name
        tb_lineno = tb.tb_lineno
        tb_str = f"Traceback {tb_i} to filename\n{tb_fname}\nfor function {tb_func}(...) at line {tb_lineno}.\n"
        if verbose:
            print(tb_str)
        trace.append(tb_str)
        tb = tb.tb_next
        tb_i += 1
    logging_list = [message] + trace
    return logging_list


def superimpose_slo_segmentation(slo, slo_vbinmap, slo_avimout, 
                             od_mask, od_centre, 
                             fovea, location, zonal_masks,
                             save_info):
    '''
    Superimpose the segmentation masks onto the SLO image.
    '''
    fname, save_path, dirpath, save_images, collate_segmentations = save_info
    # binary vessel mask - purple
    N = slo_vbinmap.shape[0]
    slo_vcmap = generate_imgmask(slo_vbinmap, None, 1)
    stacked_img = np.hstack(3*[slo/255])

    # Stacks the colour maps together, binary, then artery-vein-optic disc
    slo_av_cmap = slo_avimout.copy()
    slo_av_cmap[slo_av_cmap[...,1]>0,-1] = 0
    slo_av_cmap[...,1] = 0
    stacked_cmap = np.hstack([np.zeros_like(slo_vcmap), slo_vcmap, slo_av_cmap])
    if od_mask.sum() != 0:
        od_coords = avo_inference._fit_ellipse((255*od_mask).astype(np.uint8), get_contours=True)[:,0]
        od_coords = od_coords[(od_coords[:,0] > 0) & (od_coords[:,0] < N-1)]
        od_coords = od_coords[(od_coords[:,1] > 0) & (od_coords[:,1] < N-1)]

    fig, ax = plt.subplots(1,1,figsize=(18,6))
    ax.imshow(stacked_img, cmap="gray")
    ax.imshow(stacked_cmap, alpha=0.5)
    for i in [N, 2*N]:
        ax.scatter(fovea[0]+i, fovea[1], marker="X", s=100, edgecolors=(0,0,0), c="r")
        if i == 2*N:  
            if od_mask.sum() != 0:
                ax.plot(od_coords[:,0]+i, od_coords[:,1], color='lime', linestyle='--', linewidth=3, zorder=4)
                if location == "Optic disc":
                    ax.scatter(od_centre[0]+i, od_centre[1], marker="X", s=100, edgecolors=(0,0,0), c="lime", zorder=4)
        else:
            if od_mask.sum() != 0:
                ax.plot(od_coords[:,0]+i, od_coords[:,1], color='blue', linestyle='--', linewidth=3, zorder=4)
                if location == "Optic disc":
                    ax.scatter(od_centre[0]+i, od_centre[1], marker="X", s=100, edgecolors=(0,0,0), c="blue", zorder=4)
        
            if location == "Optic disc":
                for mask, colour, z in zip(zonal_masks[1:], [0,2], [3,2]):
                    mask_bnds = segmentation.find_boundaries(mask)
                    mask = np.hstack(2*[np.zeros_like(mask)]+[mask])
                    cmap = generate_imgmask(mask, None, colour)
                    mask_bnds = np.hstack(2*[np.zeros_like(mask_bnds)]+[mask_bnds])
                    mask_bnds = morphology.dilation(mask_bnds, footprint=morphology.disk(radius=2))
                    cmap_bnds = generate_imgmask(mask_bnds, None, colour)
                    ax.imshow(cmap, alpha=0.25, zorder=z)
                    ax.imshow(cmap_bnds, alpha=0.75, zorder=z)

    # ax.imshow(od_cmap, alpha=0.5)
    ax.set_axis_off()
    fig.tight_layout(pad = 0)
    if save_images:
        fig.savefig(os.path.join(save_path, f"{fname}_superimposed.png"), bbox_inches="tight")
    if collate_segmentations:
        segmentation_directory = os.path.join(dirpath, "slo_segmentations")
        if not os.path.exists(segmentation_directory):
            os.mkdir(segmentation_directory)
        if save_images:
            shutil.copy(os.path.join(save_path, f"{fname}_superimposed.png"), 
                        os.path.join(segmentation_directory, f"{fname}.png"))
        else:
            fig.savefig(os.path.join(segmentation_directory, f"{fname}.png"), bbox_inches="tight")
    plt.close()



def _create_circular_mask(center, img_shape, radius):
    """
    Given a center, radius and image shape, draw a filled circle
    as a binary mask.
    """
    # Circular mask
    h, w = img_shape
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = (dist_from_center <= radius).astype(int)
    
    return mask



def generate_zonal_masks(img_shape, od_radius, od_centre, location='Macular'):

    mask_rois = {}
    if location=='Macula':
        zones = ['whole']
    elif location=='Optic disc':
        zones = ['whole', 'B', 'C']

    for roi_type in zones:
        if roi_type == 'whole':
            mask = np.ones(img_shape)
        else:
            if roi_type == "B":
                od_circ = _create_circular_mask(img_shape=img_shape, 
                                            radius=2*od_radius, 
                                            center=od_centre)
                
                mask  = _create_circular_mask(img_shape=img_shape, 
                                            radius=3*od_radius, 
                                            center=od_centre)
            elif roi_type == "C":
                od_circ = _create_circular_mask(img_shape=img_shape, 
                                            radius=od_radius, 
                                            center=od_centre)
                
                mask = _create_circular_mask(img_shape=img_shape, 
                                                radius=5*od_radius, 
                                                center=od_centre)

            mask -= od_circ
        mask_rois[roi_type] = mask

    return mask_rois


# Description of all the columns found in the metadata sheet
meta_cols = {'Filename':'Filename of the SLO+OCT file analyse.',
             'FAILED':'Boolean flag on whether file unexpectedly failed to be analysed.',
             'eye':'Type of eye, either Right or Left.',
             
             'bscan_type':'Type of OCT scan acquired. One of H(orizontal)-line, V(ertical)-line;A(rtery)V(ein)-line, P(osterior)pole and Peripapillary.',
             'bscan_resolution_x': 'Number of columns of B-scan, typically 768 or 1536 for Heidelberg.',
             'bscan_resolution_y': 'Number of rows of B-scan, typically 768 or 496 for Heidelberg.',
             'bscan_scale_z': 'Micron distance between successive B-scans in a Posterior pole acquisition. Is 0 for all other Bscan_types.',
             'bscan_scale_x': 'Pixel lengthscale in the horizontal direction B-scan/SLO, measured in microns per pixel.',
             'bscan_scale_y': 'Pixel lengthscale in the vertical direction in the B-scan, measured in microns per pixel.',
             'scale_units': 'Units of the lengthscales, this is fixed as microns per pixel.',
             
             'avg_quality': 'Heidelberg-provided signal-to-noise ratio of the B-scan(s).',
             'retinal_layers_N': 'Number of retinal layer segmentations extracted from metadata.',
             'scan_focus': 'Scan focus of the acquisition, in Dioptres. This decides the scaling and is a gross measure of refractive error.',
             'visit_date': 'Date of acquisition.',
             'exam_time': 'Time of acquisition.',
             
             'slo_resolution_px': 'Number of rows/columns in the square-shaped SLO image (typically 768 or 1536).',
             'field_of_view_mm': 'Field of view captured during acquisition, usually between 8 and 9 mm if field size is 30 degrees.',
             'slo_scale_xy': 'Pixel lengthscale of the SLO image, and is typically the same for both directions.',
             'location': 'Whether scan is macula-centred or disc-centred. Is either "macular" or "peripapillary"',
             'field_size_degrees': 'Field of view in degrees, typically 30.',
             'slo_modality': ' Modality used for SLO image capture. OCTolyzer supports grayscale NIR cSLO images currently.',
             
             'bscan_fovea_x': 'Horizontal pixel position of the fovea on the OCT B-scan (if visible in one of the scans, only relevant for macular OCT).',
             'bscan_fovea_y': 'Vertical pixel position of the fovea on the OCT B-scan (if visible in one of the scans, only relevant for macular OCT).',
             'bscan_missing_fovea':'Boolean value flagging whether fovea is missing from OCT data (either due to acquisition or segmentation failure).',
             'slo_fovea_x': 'Horizontal pixel position of the fovea on the SLO image, if visible.',
             'slo_fovea_y': 'Vertical pixel position of the fovea on the SLO image, if visible.',
             'acquisition_angle_degrees': 'Angle of elevation from horizontal image axis of acquisition for Posterior pole scans.',
             'slo_missing_fovea':'Boolean value flagging whether fovea is missing from SLO data (either due to acquisition or segmentation failure).',
             
             'optic_disc_overlap_index_%':'% of the optic disc diameter, defining how off-centre a peripapillary image acquisition is from the optic disc centre.',
             'optic_disc_overlap_warning': 'Boolean value, flagging if the overlap index is greater than 15%, the empirical cut-off to warn end-user of an off-centre scan.',
             'optic_disc_x': 'Horizontal pixel position of the optic disc centre on the SLO image, if visible.',
             'optic_disc_y': 'Vertical pixel position of the optic disc centre on the SLO image, if visible.',
             'optic_disc_radius_px': 'Pixel radius of the optic disc.',
             
             'thickness_units':'Units of measurement for thickness, always in microns (micrometres).',
             'vascular_index_units':'Units of measurement for choroid vascular index, always dimensionless (no units, but is a ratio between 0 and 1).',
             'vessel_density_units':'Units of measurement for choroid vessel density, always in micron2 (square microns)',
             'area_units':'Units of measurements for area, always in mm2 (square millimetres).',
             'volume_units':'Units of measurements for volume, always in mm3 (cubic millimetres).',
             'linescan_area_ROI_microns':'For single-line, macular OCT, this is the micron distance defining the fovea-centred region of interest.',
             'choroid_measure_type':'Whether the choroid is measured column-wise (per A-scan) or perpendicularly. Always per A-scan for peripapillary OCT.',
             'missing_retinal_oct_linescan_measurements':'Whether OCT retinal measurements could not be computed for H-/V-linescans due to too large an ROI or too short a segmentation.',
             'missing_choroid_oct_linescan_measurements':'Whether OCT choroidal measurements could not be computed for H-/V-linescans due to too large an ROI or too short a segmentation.',

             'acquisition_radius_px': 'Pixel radius of the acquisition line around the optic disc for peripapillary OCT.',
             'acquisition_radius_mm': 'Millimetre radius of the acquisition line around the optic disc for peripapillary OCT.',
             'acquisition_optic_disc_center_x': 'Horizontal pixel position of the optic disc centre, as selected by the user during peripapillary OCT acquisition.',
             'acquisition_optic_disc_center_y': 'Vertical pixel position of the optic disc centre, as selected by the user during peripapillary OCT acquisition.'}
metakey_df = pd.DataFrame({'column':meta_cols.keys(), 'description':meta_cols.values()})