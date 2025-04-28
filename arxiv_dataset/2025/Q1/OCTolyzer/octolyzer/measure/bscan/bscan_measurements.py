import logging
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

from octolyzer.measure.bscan.utils import (extract_bounds, curve_length, curve_location, 
                                            get_trace, nearest_coord, construct_line, 
                                            detect_orthogonal_pts, interp_trace)
from octolyzer import utils

def compute_measurement(reg_mask, 
                       vess_mask=None, 
                       fovea: [tuple, np.ndarray] = None,
                       scale: tuple[int, int, int] = (11.49,3.87),
                       macula_rum: int = 3000,
                       N_measures: int = 'all',
                       N_avgs = 0,
                       offset=15,
                       measure_type: str = "perpendicular",
                       img_shape: tuple = (768,768),
                       plottable: bool = False,
                       force_measurement: bool = False,
                       verbose: [int, bool]=0,
                       logging_list: list = []):
    """
    Compute measurements of interest, that is thickness and area using the reg_mask and 
    CVI (optional) using the vess_mask.

    Inputs:
    -------------------------
    reg_mask : binary mask segmentation of the choroidal space.
    vess_mask : segmentation of the choroidal space. Need not be binary.
    fovea : Fovea coordinate to define fovea-centred ROI. Default is center column,row of mask
    scale: Microns-per-pixel in x and z directions. Default setting is Heidelberg scaling 
        for emmetropic individual.
    macula_rum : Micron radius either side of fovea to measure. Default is the largest region in 
        ETDRS grid (3mm).
    N_measures : Number of thickness measurements to make across choroid. Default is three: subfoveal and 
        a single temporal/nasal location.
    N_avgs : Number of adjecent thicknesses to average at each location to enforce robustness. Default is
        one column, set as 0.
    offset : Number of pixel columns to define tangent line around upper boundary reference points, for 
        accurate, perpendicular detection of lower boundary points.
    measure_type : Whether to measure locally perpendicular to the upper boundary ("perpendicular") or measure
        columnwise, i.e. per A-scan ("vertical").
    plottable : If flagged, returnboundary points defining where thicknesses have been measured, and binary masks
        where choroid area and vascular index have been measured.
    force_measurement : If segmentation isn't long enough for macula_rum-N_avgs-offset selection, this forces
        a measurement to be made by under-measuring. Default: False.
    verbose : Log to user regarding segmentation length.
    logging_list : List of log information to save out.

    Outputs:
    -------------------------
    ct : choroid thickness, an integer value per location measured (N_measures, the average of N_avgs adjacent thickness values)
    ca : choroid area in a macula_rum microns, fovea-centred region of interest.
    cvi : choroid vascular index in a macula_rum microns, fovea-centred region of interest.
    """
    measurements = []

    # Constants
    micron_pixel_x, micron_pixel_y = scale
    pixels_per_micron  = 1 / micron_pixel_x 
    if N_measures == 'all':
        measure_all = True
        N_measures = int(2*macula_rum*pixels_per_micron)
        N_avgs = 0
    else:
        measure_all = False
        N_avgs = N_avgs+1 if N_avgs%2 != 0 else N_avgs
    N_measures = max(N_measures + 1,3) if N_measures % 2 == 0 else max(N_measures,3)  

    # Organise region mask
    if isinstance(reg_mask, np.ndarray):
        if reg_mask.ndim == 2:
            rmask = reg_mask.copy()
            traces = get_trace(reg_mask, None, align=True)
        else:
            rmask = utils.rebuild_mask(reg_mask, img_shape=img_shape)
            traces = reg_mask.copy()
    elif isinstance(reg_mask, tuple):
        rmask = utils.rebuild_mask(reg_mask, img_shape=img_shape)
        traces = interp_trace(reg_mask)

    # If fovea is known - if not, take as halfway along region
    # segmentation
    if fovea is not None:
        if isinstance(fovea, tuple):
            fovea = np.array(fovea)
        ref_pt = fovea
    else:
        x_N = int(0.5 * (traces[0].shape[0] + traces[1].shape[0]))
        # x_st = int(0.5*(traces[0,0] + traces[1,0]))
        x_st = int(0.5 * (traces[0][0, 0] + traces[1][0, 0]))
        x_half = x_N // 2 + x_st
        y_half = traces[0][:,1].mean()
        ref_pt = np.array([x_half, y_half])

    # Work out reference point along upper boundary closest to fovea
    # and re-adjust reference point on upper boundary to align with indexing
    top_chor, bot_chor = traces
    rpechor_stx, chorscl_stx = int(top_chor[0, 0]), int(bot_chor[0, 0])
    rpechor_refpt, offset_pts = nearest_coord(top_chor, ref_pt, offset, columnwise=False)
    ref_idx = rpechor_refpt[0] - rpechor_stx

    # Set up list of micron distances either side of reference point, dictated by N_measures
    delta_micron = 2 * macula_rum // (N_measures - 1)
    delta_i = [i for i in range((N_measures - 1) // 2 + 1)]
    micron_measures = np.array([i * delta_micron for i in delta_i])

    # Locate coordinates along the upper boundary at equal spaces away from foveal pit until macula_rum
    curve_indexes = [curve_location(top_chor, distance=d, ref_idx=ref_idx, scale=scale) for d in micron_measures]

    # Catch if this completely fails, then resort to segmentation failure (either due to fovea/layer detection)
    if (None, np.nan) in curve_indexes or (np.nan, None) in curve_indexes or (None, None) in  curve_indexes:

        msg = f"""        Segmentation failure for {macula_rum} micron ROI. 
        Please check fovea detection or segmentation. 
        Perhaps reducing the region of interest might prevent this from happening.
        Returning -1s."""
        logging_list.append(msg)
        if verbose:
            print(msg)

        # Empty outputs depending on input
        plotinfo = None
        avg_thick = [np.array(N_measures*[-1], dtype=np.int64), -1][measure_all]
        measurements = [(-1, avg_thick, -1, -1, -1), (-1, avg_thick, -1)][vess_mask is None]
        if plottable:
            return measurements, plotinfo, logging_list
        else:
            return measurements, logging_list

    # If we can, make sure that the curve_indexes at each end of the choroid are within offset+N_avgs//2 of the 
    # last index of each end of  the trace. Only relevant if measuring perpendicularly.
    if measure_type == "perpendicular":
        rpechor_endpts = np.array([top_chor[idx, 0] for idx in curve_indexes[-1]])
        x_endpts = top_chor[[0,-1],0]
        new_curve_indexes = list(curve_indexes[-1])
        st_diff = (rpechor_endpts[0]-(offset+N_avgs//2)) - x_endpts[0]
        en_diff = (rpechor_endpts[1]+(offset+N_avgs//2)) - x_endpts[1]
        st_flag = 0
        en_flag = 0
        if st_diff <= 0:
            st_flag = 1
            new_curve_indexes[0] = curve_indexes[-1][0] - st_diff
        if en_diff >= 0:
            en_flag = 1
            new_curve_indexes[1] = curve_indexes[-1][1] - en_diff
        curve_indexes[-1] = tuple(new_curve_indexes)
        
        # Logging to user about consequence of forcing measurement if segmentation isn#t long enough
        if st_flag + en_flag > 0 and not force_measurement:
            msg = f"""        Segmentation not long enough for {macula_rum} microns.
        Extend segmentation or reduce region of interest to prevent this from happening.
        Returning -1s."""
            logging_list.append(msg)
            if verbose:
                print(msg)
            
            # Empty outputs depending on input
            plotinfo = None
            avg_thick = [np.array(N_measures*[-1], dtype=np.int64), -1][measure_all]
            measurements = [(-1, avg_thick, -1, -1, -1), (-1, avg_thick, -1)][vess_mask is None]
            if plottable:
                return measurements, plotinfo, logging_list
            else:
                return measurements, logging_list

        elif force_measurement and st_flag==1:
            msg = f"""        Segmentation not segmented long enough for {macula_rum} microns.
        Reducing left-endpoint reference point by {-st_diff} pixels.
        Extend segmentation or reduce region of interest to prevent under-measurement."""
            logging_list.append(msg)
            if verbose:
                print(msg)
        elif force_measurement and en_flag==1:
            msg = f"""        Segmentation not long enough for {macula_rum} microns.
        Reducing right-endpoint reference point by {en_diff} pixels.
        Extend segmentation or reduce region of interest to prevent under-measurement."""
            logging_list.append(msg)
            if verbose:
                print(msg)

    # Collect reference points along upper boundary - N_avgs allows us to make more robust thickness measurements by taking average value of advacent positions
    rpechor_pts = np.array([top_chor[[idx + np.arange(-N_avgs//2, N_avgs//2+1)]] for loc in curve_indexes for idx in loc]).reshape(-1,2)[1:]
    rpechor_pts = rpechor_pts[rpechor_pts[:,0].argsort()]

    # Collect reference points along lower boundary, given upper boundary reference points - taken A-scan wise for vertical measure_type and 
    # locally perpendicular to upper boundary reference point if perpendicular measure_type
    if measure_type == "perpendicular":
        chorscl_pts, rpechor_pts, perps, endpoint_errors = detect_orthogonal_pts(rpechor_pts, traces, offset)
        rpechor_pts[~endpoint_errors] = np.nan
        chorscl_pts[~endpoint_errors] = np.nan
    elif measure_type == "vertical":
        st_Bx = bot_chor[0,0]
        chorscl_pts = bot_chor[rpechor_pts[:,0]-st_Bx]

    # Collect reference points along boundaries to make thickness measurement
    chorscl_pts = chorscl_pts.reshape(N_measures, N_avgs+1, 2)
    rpechor_pts = rpechor_pts.reshape(N_measures, N_avgs+1, 2)
    boundary_pts = np.concatenate([rpechor_pts.reshape(*chorscl_pts.shape), chorscl_pts], axis=-1).reshape(*chorscl_pts.shape, 2)
                                       
    # Compute choroid thickness at each reference point.
    delta_xy = np.abs(np.diff(boundary_pts, axis=boundary_pts.ndim-2)) * np.array([micron_pixel_x, micron_pixel_y])

    # If measuring at every point across ROI, return average and subfoveal (averaging across 5 adjacent measurements)
    if measure_all:
        choroid_thickness = np.sqrt(np.square(delta_xy).sum(axis=-1))[:,0]
        subf_thick = int(np.nanmean(choroid_thickness[N_measures//2-3:N_measures//2+2]))
        avg_thick = int(np.nanmean(choroid_thickness))
        measurements.append(subf_thick)
        measurements.append(avg_thick)

    # Otherwise, return all thickness values measured at all N_measures reference points
    else:
        choroid_thickness = np.rint(np.nanmean(np.sqrt(np.square(delta_xy).sum(axis=-1)), axis=1)).astype(int)[:,0]
        measurements.append(choroid_thickness)
    
    # Compute choroid area                               
    area_bnds_arr = np.swapaxes(boundary_pts[[0,-1], N_avgs//2], 0, 1).reshape(-1,2)
    if np.any(np.isnan(area_bnds_arr)):
        msg = f"""        Segmentation not long enough for {macula_rum} microns.
        Extend segmentation or reduce region of interest to prevent under-measurement.
        Returning -1s."""
        logging_list.append(msg)
        if verbose:
            print(msg)

        # Empty outputs depending on input
        plotinfo = None
        avg_thick = [np.array(N_measures*[-1], dtype=np.int64), -1][measure_all]
        measurements = [(-1, avg_thick, -1, -1, -1), (-1, avg_thick, -1)][vess_mask is None]
        if plottable:
            return measurements, plotinfo, logging_list
        else:
            return measurements, logging_list

    # Compute choroidal area
    choroid_area, plot_output = compute_area_enclosed(traces, area_bnds_arr.astype(int), scale=scale, plot=True)
    chor_pixels, (x_range, y_range), (left_x, right_x) = plot_output
    measurements.append(choroid_area)

    # CVI
    if vess_mask is not None:

        # Compute CVI --- choroidal vessel area is whatever pixels in keep_pixel 
        # which are flagged as 1 in vess_mask
        vessel_area = 0
        xchor = np.unique(chor_pixels[:,0]).astype(np.int64)
        chor_pixels = chor_pixels.astype(np.int64)
        for x in xchor:
            col = chor_pixels[chor_pixels[:,0]==x]
            bmap_col = vess_mask[col[:,1]-1, x]
            vessel_area += bmap_col.sum()
    
        # Total pixel-choroid area is the number of pixels contained with the ROI
        # CVI is just # vessel pixels divided by # choroid area pixels
        total_chor_area = chor_pixels.shape[0]
        choroid_cvi = np.round(vessel_area/total_chor_area, 5)
        measurements.append(choroid_cvi)

        # Other metrics like EVI, vessel area and intersitial area (in mm2)
        micron_area = micron_pixel_x * micron_pixel_y
        choroid_vessel_area = np.round(1e-6 * micron_area * vessel_area, 6)
        measurements.append(choroid_vessel_area)

    if plottable:
        ca_mask = np.zeros_like(rmask)
        chor_pixels = chor_pixels.astype(int)
        ca_mask[chor_pixels[:,1], chor_pixels[:,0]] = 1
        if vess_mask is not None:
            return measurements, (boundary_pts, ca_mask, vess_mask), logging_list
        else:
            return measurements, (boundary_pts, ca_mask), logging_list
    else:
        return measurements, logging_list



def compute_area_enclosed(traces, 
                          area_bnds_arr, 
                          scale=(11.49,3.87), 
                          plot=False):
    """
    Function which, given traces and four vertex points defining the smallest irregular quadrilateral to which
    the choroid area is enclosed in, calculate the area to square millimetres.

    INPUTS:
    ---------------------
        traces (3-tuple) : Tuple storing upper and lower boundaries of trace

        area_bnds_arr (np.array) : Four vertex pixel coordnates defining the smallest irregular quadrilateral
            which contains the choroid area of interest.

        scale (3-tuple) : Tuple storing pixel_x-pixel_y-micron scalar constants.

        plot (bool) : If flagged, output information to visualise area calculation, including the points contained
            in the quadrilateral and the smallest rectangular which contains the irregular quadrilateral.

    RETURNS:
    --------------------
        choroid_mm_area (float) : Choroid area in square millimetres.

        plot_outputs (list) : Information to plot choroid area calculation.
    """
    # Extract reference points scale and traces
    top_chor, bot_chor = traces
    rpechor_stx, chorscl_stx = top_chor[0, 0], bot_chor[0, 0]
    rpechor_ref, chorscl_ref = area_bnds_arr[:2], area_bnds_arr[2:]

    # Compute microns-per-pixel and how much micron area a single 1x1 pixel represents.
    micron_pixel_x, micron_pixel_y = scale
    micron_area = micron_pixel_x * micron_pixel_y
    
    # Work out range of x- and y-coordinates bound by the area, building the smallest rectangular
    # region which overlaps the area of interest fully
    x_range = np.arange(area_bnds_arr[:, 0].min(), area_bnds_arr[:, 0].max() + 1)
    y_range = np.arange(min(top_chor[x_range[0] - rpechor_stx: x_range[-1] - rpechor_stx + 1, 1].min(),
                            area_bnds_arr[:, 1].min()),
                        max(bot_chor[x_range[0] - chorscl_stx: x_range[-1] - chorscl_stx + 1, 1].max(),
                            area_bnds_arr[:, 1].max()) + 1)
    N_y = y_range.shape[0]

    # This defines the left-most perpendicular line and right-most perpendicular line
    # for comparing with coordinates from rectangular region
    left_m, left_c = construct_line(rpechor_ref[0], chorscl_ref[0])
    right_m, right_c = construct_line(rpechor_ref[1], chorscl_ref[1])
    if left_m != np.inf:
        left_x = ((y_range - left_c) / left_m).astype(np.int64)
    else:
        left_x = np.array(N_y * [rpechor_ref[0][0]])
    if right_m != np.inf:
        right_x = ((y_range - right_c) / right_m).astype(np.int64)
    else:
        right_x = np.array(N_y * [rpechor_ref[1][0]])
    # The rectangular region above needs reduced to only containing coordinates which lie
    # above the Chor-Sclera boundary, below the RPE-Choroid boundary, to the right of the
    # left-most perpendicular line and to the left of the right-most perpendicular line.
    keep_pixel = []

    # We vectorise check by looping across x_range and figuring out if each coordinate
    # in the column satisfies the four checks described above
    for x in x_range:
        # Extract column
        col = np.concatenate([x * np.ones(N_y)[:, np.newaxis], y_range[:, np.newaxis]], axis=1)

        # Define upper and lower bounds at this x-position
        top, bot = top_chor[x - rpechor_stx], bot_chor[x - chorscl_stx]

        # Check all 4 conditions and make sure they are all satisfied
        cond_t = col[:, 1] >= top[1]
        cond_b = col[:, 1] <= bot[1]
        cond_l = x >= left_x
        cond_r = x < right_x
        col_keep = col[cond_t & cond_b & cond_l & cond_r]
        keep_pixel.append(col_keep)

    # All pixels bound within the area of interest
    keep_pixel = np.concatenate(keep_pixel)

    # Calculate area (in square mm)
    choroid_pixel_area = keep_pixel.shape[0]
    choroid_mm_area = np.round(1e-6 * micron_area * choroid_pixel_area, 6)

    # If plotting, reutrn pixels used to compute  area
    if plot:
        plot_outputs = [keep_pixel, (x_range, y_range), (left_x, right_x)]
        outputs = [choroid_mm_area, plot_outputs]
    else:
        outputs = choroid_mm_area

    return outputs