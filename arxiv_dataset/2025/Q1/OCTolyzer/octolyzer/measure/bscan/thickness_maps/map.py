import numpy as np
import os
import logging
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import rotate, resize
from skimage.morphology import skeletonize
from tqdm.autonotebook import tqdm
from sklearn.linear_model import RANSACRegressor, LinearRegression
from scipy.ndimage import gaussian_filter, median_filter
from octolyzer.measure.bscan.thickness_maps.utils import (extract_bounds, interp_trace, 
                                                          smart_crop, get_trace)
import cv2


def detect_angle(slo_at_fovea, fovea, N_stack=31, horizontal=True, plot=False, inpt=".tif"):
    """
    Using the SLO at the fovea, detect the angle at which the macula scan is from the
    x-axis. Used to rotate SLO for registering ChTh heatmap.
    """
    # Image shape and number of pixels to define ROI to locate line and corresponding angle
    M, N, C = slo_at_fovea.shape
    slo_thresh = 0.05
    n = 250
    
    # Take mean intensities over red and blue channels and extract ROI
    if N_stack==61 and inpt == ".tif":
        rb_slo = slo_at_fovea[...,[0,2]].mean(axis=-1)
    else:
        rb_slo = np.abs(1-slo_at_fovea[...,1])
    roi_slo = rb_slo[M//2-n : M//2+n, N//2-n : N//2+n]

    # Detect and fit RANSAC regressor line
    where0 = np.argwhere(skeletonize(roi_slo < slo_thresh))
    if horizontal:
        where0 = where0[:,[1,0]]
    ransac_line = RANSACRegressor().fit(where0[:,0].reshape(-1,1), where0[:,1])

    # Work out fovea coordinate on SLO image 
    fovea_x = (fovea[0] - N//2+n).reshape(-1, 1)
    fovea_y = int(np.round(ransac_line.predict(fovea_x) + M//2-n)[0])
    if horizontal:
        fovea_in_slo = np.array([fovea_x[0,0]+N//2-n, fovea_y])
    else:
        fovea_in_slo = np.array([fovea_y, N//2-fovea_x[0,0]+n])
    #return ransac_line, roi_

    # Work out angle of elevation from x-axis
    gradient = ransac_line.estimator_.coef_[0]
    angle = np.arctan(gradient)
    angle_degrees = angle*180 / np.pi
    if not horizontal:
        angle_degrees = 90.0

    # If plotting
    if plot:
        #ref_x, ref_y = where0[0,0], where0[0,1]
        x_pts = (np.arange(2*n)).reshape(-1, 1)
        y_pts = ransac_line.predict(x_pts)
        x_pts += N//2-n
        y_pts += M//2-n
        
        fig, ax = plt.subplots(1,1,figsize=(10,10))
        ax.imshow(slo_at_fovea)
        ax.scatter(fovea_in_slo[0], fovea_in_slo[1], c="r", s=100)
        if horizontal:
            #ax.scatter([fovea_x+N//2-n], [fovea_y], c="r", s=100)
            ax.plot(x_pts, y_pts)
        else:
            #ax.scatter([fovea_y], [fovea_x+N//2-n], c="r", s=100)
            ax.plot(y_pts,x_pts)
    
    return angle_degrees, fovea_in_slo



def _check_offset(offset, offsets_lr, N_pts):
    '''
    Quick helper function to check if offset is too large, and deal with it if so
    '''
    (offset_l, offset_r) = offsets_lr
    if offset_l < 0:
        offset_l = 0
        logging.warning(f"Offset {offset} too far to the left, choosing index {offset_l}")
        
    if offset_r >= N_pts:
        offset_r = N_pts-1
        logging.warning(f"Offset {offset} too far to the right, choosing index {offset_r}")

    return offset_l, offset_r



def generate_perp_line(pt1, pt2=None, N=None, ref_pt=None):
    """
    Linear model of tangent line centred at reference
    point along RPE-Choroid boundary, evaluate far enough 
    such that rotating by 90 degrees will ensure its
    intersection with Choroid-Sclera boundary
    """
    # Fit linear model at reference points along tangent
    if pt2 is None:
        X, y = pt1[:,0].reshape(-1,1), pt1[:,1]
    else:
        X, y = np.array([pt1[0], pt2[0]]).reshape(-1,1), np.array([pt1[1], pt2[1]])    
    output = LinearRegression().fit(X, y)

    # Evaluate across tangent, rotate at reference point and return perpendicular
    # linear model
    if N is not None and ref_pt is not None:
        
        ref_x, ref_y = ref_pt
        xtan_grid = np.array([ref_x, X[-1,0]+N])
        ytan_grid = output.predict(xtan_grid.reshape(-1,1)).astype(int)
        
        perp_x = (-(ytan_grid - ref_y) + ref_x).reshape(-1,)
        perp_y = (xtan_grid - ref_x + ref_y).reshape(-1,)
        output = (perp_x, perp_y)

        y_grid = np.arange(perp_y[0], perp_y[1])
        x_grid = np.interp(y_grid, perp_y, perp_x)
        output = (x_grid, y_grid)
        
    return output



def detect_orthogonal_chorscl(reference_pts, traces, offset=15, tol=2):
    """
    Given the lower choroid boundary and reference points along the upper boundary, detect which
    coordinates along the lower choroid boundary which intersect the perpendicular line drawn from
    a tangent line at these reference points.

    INPUTS:
    ------------------
        reference_pts (np.array) : Points along upper boundary to construct tangent and perpendicular lines at. 

        traces (2-tuple) : Tuple storing the upper and lower boundaries of the segmented chorod, in xy-space.

        offset (int) : Value either side of reference point to define tangential line.

        tol (int) : Threshold to detect any perpendicular lines from the upper to lower boundaries
        which are deviate away the lower boundary via Euclidean distance, i.e. it's likely these lines 
        divert away from the segmented region.
    """
    # Extract traces    
    top_chor, bot_chor = traces
    rpechor_stx, chorscl_stx = top_chor[0, 0], bot_chor[0, 0]

    # total number of candidate points at each reference point to compare with 
    # Choroid-Sclera boundary
    N = max([bot_chor[ref_x-chorscl_stx, 1] - ref_y for (ref_x, ref_y) in reference_pts])
    perps = []
    for ref_pt in reference_pts:
    
        # Work out local tangent line for each reference point
        # and rotate orthogonally
        ref_x, ref_y = ref_pt
        ref_xidx = ref_x - rpechor_stx
        #N = bot_chor[ref_x-chorscl_stx, 1] - ref_y
        tan_pt1, tan_pt2 = top_chor[[ref_xidx - offset, ref_xidx + offset]] 
        (perp_x, perp_y) = generate_perp_line(tan_pt1, tan_pt2, N, ref_pt)
        perps.append(np.array([perp_x, perp_y]))
    
    # Vectorised search for points along Choroid-Sclera boundary where orthogonal 
    # lines from RPE-Choroid intersect
    perps = np.array(perps)
    bot_cropped = bot_chor[(perps[:,0].astype(int)-chorscl_stx).clip(0, bot_chor.shape[0]-1)]
    bot_perps_residuals = np.transpose(perps, (0,2,1)) - bot_cropped
    bot_perps_distances = np.sqrt(((bot_perps_residuals)**2).sum(axis=-1))
    endpoint_errors = np.min(bot_perps_distances, axis=-1) <= tol 
    chorscl_indexes = np.argmin(bot_perps_distances, axis=1)
    chorscl_pts = perps[np.arange(chorscl_indexes.shape[0]),:,chorscl_indexes].astype(int)

    return chorscl_pts[endpoint_errors], reference_pts[endpoint_errors], perps[endpoint_errors].astype(int)



def measure_thickness(chorsegs, fovea, scale, offset=15, max_N=768,
                      measure_type="perpendicular", region_thresh=1, 
                      disable_progress=True, logging=[]):
    """
    Measure choroid thickness across all Ppole segs as far as possible for every slice
    """
    N_scans = len(chorsegs)

    ct_data = []
    ct_fovs = []
    ct_topStx = []
    for idx, bmap in enumerate(tqdm(chorsegs, disable=disable_progress, total=N_scans)):

        # Smart crop, unless ndim is 3, then these are retinal layer segmentations
        if bmap.ndim == 3:
            traces = bmap.copy()
        else:
            try:
                traces = get_trace(bmap, seg_thresh=0.5, crop_thresh=region_thresh, align=True)
            except:
                traces = (np.empty((0,2)), np.empty((0,2)))
        top_chor, bot_chor = traces
    
        # Catch exception if a B-scan doesn't have any segmentations,
        N_t = top_chor.shape[0]
        N_b = bot_chor.shape[0]
        if (N_t == 0) or (N_b == 0):
            fail_msg = f"WARNING: B-scan {idx+1}/{N_scans} does not have a valid trace for this layer. This slice will be empty in the map"
            print(fail_msg)
            logging.append(fail_msg)
            ct_topStx.append(0)
            ct_data.append(-1*np.ones((max_N)))
            ct_fovs.append(fovea[0])
            continue

        # Catch any other unexpected problem
        try: 
            # Select every coordinate along upper boundary to compute thickness at
            st_Tx = top_chor[0,0]
            ct_topStx.append(st_Tx)
            rpechor_pts = top_chor if offset is None else top_chor[offset:N_t-offset]

            # If measuring perpendicularly to upper boundary, or vertically
            if measure_type == "perpendicular":
                chorscl_pts, rpechor_pts, _ = detect_orthogonal_chorscl(rpechor_pts, traces, offset=offset, tol=2)
            elif measure_type == "vertical":
                st_Bx = bot_chor[0,0]
                chorscl_pts = bot_chor[rpechor_pts[:,0]-st_Bx]
        
            # Combine upper and lower boundary reference points and remove padding
            boundary_pts = np.swapaxes(np.concatenate([rpechor_pts[...,np.newaxis], 
                                                    chorscl_pts[...,np.newaxis]], axis=-1), 1, 2)

            #M = np.abs(np.diff(boundary_pts, axis=boundary_pts.ndim-2)).max()
            # Compute choroid thickness at each boundary point using scale
            micron_pixel_x, micron_pixel_y = scale
            delta_xy = np.abs(np.diff(boundary_pts, axis=boundary_pts.ndim-2)) * np.array([micron_pixel_x, micron_pixel_y])
            ct_i = np.sqrt(np.square(delta_xy).sum(axis=-1)).mean(axis=1)
            ct_data.append(ct_i)

            # This should only fail if the segmentation doesn't reach as far as the 
            # horizontal fovea pixel location
            try:
                rpe_fovea = np.argwhere(rpechor_pts[:,0] == fovea[0])[0][0]

            # Catch this rare exception
            except:
                # Check if segmentation is mostly in left-portion/right-portion,
                # using the horizontal coordinate of the fovea 
                if boundary_pts[:,0,0].mean() < 2*fovea[0]:
                    rpe_fovea = fovea[0] - boundary_pts[0][0,0]
                else:
                    rpe_fovea = - boundary_pts[0][0,0]
            ct_fovs.append(rpe_fovea)

        except Exception as e:
            msg_error = f"\nUnknown exception of type {type(e).__name__} occurred. Error description:\n{e.args[0]}"
            msg_fail = f"Failure to measure thickness for B-scan {idx+1}/{N_scans} for current layer. Returning as 0s"
            print(msg_error)
            print(msg_fail)
            logging.extend([msg_error, msg_fail])
            ct_topStx.append(0)
            ct_data.append(np.zeros((max_N)))
            ct_fovs.append(fovea[0])
        
    return ct_data, ct_fovs, ct_topStx, logging



def measure_vessels(ves_chorsegs, reg_chorsegs, fovea, scale, offset=15, max_N=768,
                    measure_type="perpendicular", region_thresh=1, disable_progress=True, logging=[]):
    """
    Measure choroid thickness across all Ppole segs as far as possible for every slice
    """
    N_scans = len(ves_chorsegs)
    cv_data = []
    cvi_data = []
    cv_fovs = []
    cv_topStx = []
    for idx, (v_binmap, reg_bmap) in enumerate(tqdm(zip(ves_chorsegs, reg_chorsegs), total=N_scans, disable=disable_progress)):

        # Smart crop
        # Error handling if B-scan clipped so region mask is invalid, making vessel mask invalid
        try:
            traces = get_trace(reg_bmap, seg_thresh=0.5, crop_thresh=region_thresh, align=True)
            binmap = v_binmap * reg_bmap
        except:
            traces = (np.empty((0,2)), np.empty((0,2)))
            binmap = np.zeros_like(reg_bmap)
        top_chor, bot_chor = traces
    
        
        # Catch exception if a B-scan doesn't have any segmentations,
        N_t = top_chor.shape[0]
        N_b = bot_chor.shape[0]
        if (N_t == 0) or (N_b == 0):
            fail_msg = f"WARNING: B-scan {idx+1}/{N_scans} does not have a valid trace for this layer. This slice will be empty in the map"
            print(fail_msg)
            logging.append(fail_msg)
            cv_topStx.append(0)
            cv_data.append(-1*np.ones((max_N)))
            cvi_data.append(-1*np.ones((max_N)))
            cv_fovs.append(fovea[0])
            continue

        # Catch any other unexpected problem
        try: 
            # Select every coordinate along upper boundary to compute cvi at
            st_Tx = top_chor[0,0]
            cv_topStx.append(st_Tx)
            rpechor_pts = top_chor if offset is None else top_chor[offset:-offset]

            # If measuring perpendicularly to upper boundary, or vertically
            if measure_type == "perpendicular":
                vessel_pixels = np.zeros(rpechor_pts.shape[0])
                chorscl_pts, rpechor_pts, perps = detect_orthogonal_chorscl(rpechor_pts, traces, offset=offset, tol=2)
                max_M = binmap.shape[0]
                perps[:,0] = perps[:,0].clip(0, max_N-1)
                perps[:,1] = perps[:,1].clip(0, max_M-1)
                vessel_pixels = binmap[perps[:,1], perps[:,0]].sum(axis=-1)
                region_pixels = reg_bmap[perps[:,1], perps[:,0]].sum(axis=-1)
                cvi_pixels = vessel_pixels / region_pixels
            elif measure_type == "vertical":
                vessel_pixels = binmap[:,rpechor_pts[:,0]].sum(axis=0)
                region_pixels = reg_bmap[:,rpechor_pts[:,0]].sum(axis=0)
                cvi_pixels = vessel_pixels / region_pixels

            # Compute how much vessel area was picked up 
            micron_pixel_x, micron_pixel_y = scale
            pixel_micron_area = micron_pixel_x * micron_pixel_y
            vessel_area = vessel_pixels * pixel_micron_area
            cvi_data.append(cvi_pixels)
            cv_data.append(vessel_area)

            # This should only fail if the segmentation doesn't reach as far as the 
            # horizontal fovea pixel location
            try:
                rpe_fovea = np.argwhere(rpechor_pts[:,0] == fovea[0])[0][0]

            # Catch this rare exception
            except:
                # Check if segmentation is mostly in left-portion/right-portion,
                # using the horizontal coordinate of the fovea 
                if rpechor_pts[:,0].mean() < 2*fovea[0]:
                    rpe_fovea = fovea[0] - rpechor_pts[0,0]
                else:
                    rpe_fovea = - rpechor_pts[0,0]
            cv_fovs.append(rpe_fovea)

        except Exception as e:
            msg_error = f"\nUnknown exception of type {type(e).__name__} occurred. Error description:\n{e.args[0]}"
            msg_fail = f"Failure to measure thickness for B-scan {idx+1}/{N_scans} for current layer. Returning as 0s"
            print(msg_error)
            print(msg_fail)
            logging.extend([msg_error, msg_fail])
            cv_topStx.append(0)
            cv_data.append(np.zeros((max_N)))
            cvi_data.append(np.zeros((max_N)))
            cv_fovs.append(fovea[0])
        
    return cv_data, cvi_data, cv_fovs, cv_topStx, logging



def build_chth_map(ct_data, 
                   ct_fovs,
                   ct_topStx,
                   fovea, 
                   N_stack, 
                   slo_Vs, 
                   max_N=768,
                   line_distance=10,
                   type="bilinear", 
                   verbose=False):
    """
    Build choroid thickness map to scale with SLO
    """    
    # Build map of choroid thicknesses per slice
    slo_V, slo_V_t, slo_V_b = slo_Vs
    ct_lengths = [ct.shape[0] for ct in ct_data]
    max_shape = max(ct_lengths)
    max_ct = np.argmax(ct_lengths)

    # In order to prevent any interpolation artefatcs, create a smooth version of the choroid
    # map which is just a padded original, duplicate edge values
    ct_map = -np.zeros((N_stack, max_N)) # This is new
    ct_smooth = np.zeros((N_stack, max_N)) # This is new
    for i, (ct_i, ct_f, ct_l, stX) in enumerate(zip(ct_data, ct_fovs, ct_lengths, ct_topStx)):
        pad_l = fovea[0] - ct_f
        pad_r = max_N - (pad_l + ct_l)
        ct_map[i] = np.pad(ct_i, (pad_l,pad_r),mode="constant", constant_values=-1)
        ct_smooth[i] = np.pad(ct_i, (pad_l,pad_r),mode="edge")
    ct_mask = (ct_map > -1).astype(float) # This is new
    
    if verbose:
        fig, ax = plt.subplots(1,1,figsize=(9,7))
        hmax = sns.heatmap(ct_map[::-1],
                    cmap = "rainbow",
                    alpha = 0.5,
                    zorder = 2,
                    ax = ax)

    # Vertical interpolation to scale to ChTh map to the SLO
    # Interpolate both the binary map of where the original choroid map was defined
    # and the smoothed version, and then set everywhere not originally measured at to 0
    ct_maskT = torch.tensor(ct_mask).unsqueeze(0).unsqueeze(0)
    ct_mask_scaled = F.interpolate(ct_maskT, size=(slo_V, max_N), mode="nearest").squeeze(0).squeeze(0).numpy().astype(bool)
    ct_mapT = torch.tensor(ct_smooth).unsqueeze(0).unsqueeze(0)
    ct_scaled = F.interpolate(ct_mapT, size=(slo_V, max_N), mode=type).squeeze(0).squeeze(0).numpy()

    # Smooth using a isotropic Gaussian window with standard eviation of 5, and remove
    # smooth padding
    ct_scaled = gaussian_filter(ct_scaled, sigma=2*np.sqrt(line_distance))
    ct_scaled[~ct_mask_scaled] = -1 # This is new
    if verbose:
        fig, ax = plt.subplots(1,1,figsize=(9,7))
        hmax = sns.heatmap(ct_scaled[::-1],
                    cmap = "rainbow",
                    alpha = 0.5,
                    zorder = 2,
                    ax = ax)

    # Pad vertically to align with macular ROI on SLO
    ct_M, ct_N = ct_scaled.shape
    pad_M = (max(0, fovea[1]-slo_V_t),
             max_N - (slo_V+max(0, fovea[1]-slo_V_t)))

    ct_mask_padded = np.pad(ct_mask_scaled, ((pad_M[1], pad_M[0]),(0, 0)), 
                            mode="constant", constant_values=False)[::-1]
    ct_padded = np.pad(ct_scaled, ((pad_M[1], pad_M[0]),(0,0)))[::-1]
    
    return ct_padded, ct_mask_padded



def trim_map(map, mask, length=2):
    """
    Only used when rotate_slo=False, i.e. when we rotate
    the map, we trim any extreme values from the rotation
    interpolation
    """
    new_map = map.copy()
    mask = 255*(mask.astype(np.uint8))
    for i in range(length):
        threshold = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)[1]
        boundary = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0].squeeze()
        mask[boundary[:,1], boundary[:,0]] = 0
    new_map[~mask.astype(bool)] = -1
    return new_map




def construct_map(slo, 
                slo_at_fovea, 
                reg_chorsegs, 
                fovea,
                fovea_slice_num,
                scale=(11.49,3.87), 
                bscan_delta=120, 
                max_N = 768,
                measure_type="vertical",
                rotate_slo=False,
                ves_chorsegs=None,
                inpt = ".vol",
                log_list = []):
    """
    Wrapper function to build a paired SLO, Choroid Thickness map
    for training a model to predict ChTh map from SLO.

    Inputs:
    ---------------------------------
    slo : clean SLO image without any acquisition locations.
    
    slo_at_fovea : Required, SLO scan with fovea B-scan acquisition location superimposed.
    
    reg_chorsegs : Required, choroid region segmentations for generating thickness maps.
    
    fovea : Pixel coordinate of the fovea in the fovea-centred OCT B-scan.
        
    measure_type ("vertical") : Measuring thickness/vessel density
             column-wise or (vertical) or locally "perpendicular" to upper boundary.
                
    scale (11.49, 3.87) : Horizontal and vertical scaling of the B-scan.
    
    bscan_delta (120) : Micron distance between parallel macular Ppole line scans. 

    max_N (int) : So as to align the resolution of the SLO and B-scan together, as sometimes
    the SLO can be of (512,512) dimensions, when the B-scans are 496 x 768 (in Heidelberg, OCT2, for example).
            
    rotate_slo (False) : Whether to rotate SLO to register to map or vice versa.
    
    ves_chorsegs (None) : choroid vessel segmentations of Ppole stack. If provided,
            vessel density map is computed.

    inpt (str): The type of input data, can be either ".tif" or ".vol". ".e2e" and
            ".dicom" yet to be implemented.

    log_list (list) : Process log to alert user of any exceptions caught.
    """
    # Core parameters
    verbose = False
    N_stack = len(reg_chorsegs)
    scale_X, scale_Y = scale
    N, M = (max_N,max_N)
    if (slo.shape[:2] != (N, M)) or (slo_at_fovea.shape[:2] != (N, M)):
        slo = resize(slo, output_shape=(M,N))
        slo_at_fovea = resize(slo_at_fovea, output_shape=(M,N))

    # Scan number in stack which intersects fovea on SLO
    fovea_slice_num = fovea_slice_num or N_stack//2 + 1

    # vertical distance to interpolate and sample ChTh heatmap from
    line_distance = bscan_delta/scale_X
    delta_b, delta_t = np.array([fovea_slice_num-1, N_stack-fovea_slice_num])
    slo_V_t = int(delta_t*line_distance)
    slo_V_b = int(delta_b*line_distance)
    slo_V = slo_V_t+slo_V_b
    slo_Vs = (slo_V, slo_V_t, slo_V_b)

    # Detect angle of rotation for SLO to map to ChTh map
    # slo_at_fovea: SLO scan with fovea B-scan acquisition location superimposed
    # fovea_in_slo: pixel coordinates of the fovea on the SLO image.
    # slo: clean SLO image without any acquisition locations
    # fovea_in_slo_output: 
    angle, fovea_in_slo = detect_angle(slo_at_fovea, fovea, N_stack, inpt=inpt)
    if rotate_slo:
        slo_at_fovea_output = rotate(slo_at_fovea, angle=angle, center=fovea_in_slo, resize=False)
        slo_output = rotate(slo, angle=angle, center=fovea_in_slo, resize=False)
    else:
        slo_at_fovea_output = slo_at_fovea.copy()
        slo_output = slo.copy()
    
    if verbose:
        fig, (ax,ax1) = plt.subplots(1,2,figsize=(14,7))
        ax.imshow(slo_at_fovea)
        ax.scatter([fovea_in_slo[0]], [fovea_in_slo[1]], c="r", s=100)
        ax1.imshow(slo_at_fovea_output)

    # Measure choroid thickneess for every slice
    if ves_chorsegs is None:
        chor_data, chor_fovs, chor_stxs, log = measure_thickness(reg_chorsegs, 
                                                     fovea_in_slo, 
                                                     scale, 
                                                     max_N=max_N,
                                                     logging=[],
                                                     measure_type=measure_type)
        fname_type = "_region"

    # Measure choroid vessel density along every slice
    else:
        chor_data, chor_cvi_data, chor_fovs, chor_stxs, log = measure_vessels(ves_chorsegs, 
                                                          reg_chorsegs, 
                                                          fovea_in_slo, 
                                                          scale, 
                                                          logging=[],
                                                          max_N=max_N,
                                                          measure_type=measure_type)
        fname_type = "_vessel"
    log_list.extend(log)
        

    # Interpolate map to a heatmap aligned with rotate SLO resolution
    ch_map, ch_mask = build_chth_map(chor_data, chor_fovs, chor_stxs, fovea_in_slo, 
                                     N_stack, slo_Vs, max_N=N, line_distance=line_distance, verbose=verbose)
    if ves_chorsegs is not None:
        cvi_map, _ = build_chth_map(chor_cvi_data, chor_fovs, chor_stxs, fovea_in_slo, 
                                     N_stack, slo_Vs, max_N=N, line_distance=line_distance, verbose=verbose)
    
    # If rotating the depth map, not the SLO
    if not rotate_slo:
        chmap_output = rotate(ch_map, angle=-angle, center=fovea_in_slo, resize=False, 
                              preserve_range=True, cval=-1, mode="constant")
        chmask_output = rotate(ch_mask.astype(bool), angle=-angle, center=fovea_in_slo, resize=False, 
                               cval=False, mode="constant", order=0)
        chmap_output = trim_map(chmap_output, chmask_output)
        if ves_chorsegs is not None:
            cvimap_output = rotate(cvi_map, angle=-angle, center=fovea_in_slo, resize=False, 
                              preserve_range=True, cval=-1, mode="constant")
            cvimap_output = trim_map(cvimap_output, chmask_output)

    # If rotating the SLO
    else:
        chmap_output = ch_map.copy()
        chmask_output = ch_mask.copy()
        chmap_output[~chmask_output] = -1
        if ves_chorsegs is not None:
            cvimap_output = cvi_map.copy()
            cvimap_output[~chmask_output] = -1

    if ves_chorsegs is None:
        return slo_output, chmap_output, (angle, fovea_in_slo), log_list
    else:
        return slo_output, chmap_output, (angle, fovea_in_slo), cvimap_output, log_list


def plot_slo_map(slo, map, fname="", save_path="", transparent=False, cbar=True, clip=None):
    """
    Helper function for plotting choroid map over SLO
    """
    # if clipping heatmap
    mask = map < 0
    if clip is None:
        vmax = np.quantile(map[map != -1], q=0.995)
    else:
        vmax = clip
    
    if cbar:
        figsize=(9,7)
    else:
        figsize=(9,9)
    fig, ax = plt.subplots(1,1,figsize=figsize)
    hmax = sns.heatmap(map,
                cmap = "rainbow",
                alpha = 0.5,
                vmax = vmax,
                zorder = 2,
                mask=mask,
                ax = ax,
                cbar=cbar)
    if slo is not None:
        hmax.imshow(slo, cmap="gray",
                aspect = hmax.get_aspect(),
                extent = hmax.get_xlim() + hmax.get_ylim(),
                zorder = 1)
    ax.set_axis_off()
    if fname != "": 
       fig.savefig(os.path.join(save_path, fname), pad_inches=0,
                   bbox_inches="tight", transparent=transparent)
       plt.close()