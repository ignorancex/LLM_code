import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage.morphology import skeletonize
from skimage import measure, graph
import cv2
from scipy.spatial import distance
from sloctolyzer.measure.retina import Retina, Window, detect_vessel_border
from sloctolyzer.measure.tortuosity_measures import (global_cal, distance_measure_tortuosity, 
                                                    _curve_length, squared_curvature_tortuosity, 
                                                    tortuosity_density, Knudtson_cal)
from sloctolyzer.measure import tortuosity_measures 


# Older, slower version of geometrical reordering of vessel segments
# def _reorder_coords_2(coords, imsize = (768,768)):
#     """
#     reorders (x,y)-coordinates to form a continuous line. Can be slow depending on
#     length of coordiantes and size of image, so only do it if necessary, i.e. if the
#     _curve_length(v) of the original vessel coordinates is greater than sqrt(2)*len(v),
#     as v should be discretely continuous, i.e. move one pixel horizontally and/or vertically.
#     """
#     filter = np.ones([3, 3])
#     filter[1, 1] = 10
#     binary_image = np.zeros(imsize)
#     rr = coords[:, 0]
#     cc = coords[:, 1]
#     binary_image[cc, rr] = 1
#     ends = np.zeros(imsize)

#     ends = convolve(binary_image, filter, mode='constant')
#     endpoints = np.rot90(np.array(np.where(ends == 11.)), -1)
#     if len(endpoints) != 2:
#         return(coords)

#     start_point = endpoints[0]
#     ordered_skel = []
#     for i in range(len(coords) - 1):
#         distmat = distance.cdist([start_point], coords)
#         dmat = zip(*distmat, coords)
#         dmat_s = sorted(dmat, key=lambda x: x[0])
#         ordered_skel.extend([dmat_s[0][1]])
#         for point in ordered_skel:
#               and_out = np.logical_and(coords[:, 0] == point[0], coords[:, 1] == point[1])
#               coords = np.delete(coords, np.where(and_out)[0], axis = 0)
#         start_point = ordered_skel[-1]

    
#     distmat = distance.cdist([start_point], coords)
#     dmat = zip(*distmat, coords)
#     dmat_s = sorted(dmat, key=lambda x: x[0])
#     ordered_skel.extend([x for y, x in dmat_s])
        
#     ordered_skel = np.array(ordered_skel)[:, ::-1]
    
#     return ordered_skel


def _reorder_coords(coords, imsize = (768,768)):
    """
    Fast method for sorting vessels
    
    reorders (x,y)-coordinates to form a discretely continuous line. Should only be
    used if tortuosity_measures._curve_length(coords[:,0], coords[:,1]) of the 
    original vessel coordinates is greater than sqrt(2)*len(coords), as v should be 
    discretely continuous, i.e. move one pixel horizontally and/or vertically.

    - We crop the image to a small patch which only includes the vessel segment.
    - Figure out the endpoints using a convolution over the patch
    - Use Dijkstra's shortest path to work out the geometrical ordering from the endpoints.
    """
    # Construct minim binary mask around coords
    M, N = imsize    
    binary_image = np.zeros(imsize)
    binary_image[(coords[:, 0], coords[:, 1])] = 1
    bin_props = measure.regionprops(measure.label(binary_image))[0]
    bnd_coords = bin_props.bbox
    fast_pad = 10
    x_st, x_en = max(0,bnd_coords[1]-fast_pad), min(N,bnd_coords[3]+fast_pad)
    y_st, y_en = max(0,bnd_coords[0]-fast_pad), min(M,bnd_coords[2]+fast_pad)
    binary_image = binary_image[y_st:y_en, x_st:x_en]

    # Detect endpoints by convolving filter whose center is 10 and corners are 1
    # endpoints are those with convolved value of 11
    filter = np.ones([3, 3])
    filter[1, 1] = 10
    ends = convolve(binary_image, filter, mode='constant')
    endpoints = np.rot90(np.array(np.where(ends == 11.)), -1)
    if len(endpoints) != 2:
        return(coords)

    # Given endpoints and mini mask, use Dijkstra's algorithm to trace and get ordering, then add back what was cropped
    endpoints = endpoints[:,[1,0]].T
    output = np.array(graph.route_through_array(1-binary_image, start=endpoints[:,1], end=endpoints[:,0])[0])[:,[1,0]]
    output[:,1] += y_st
    output[:,0] += x_st
    
    return output


# def width_measurement(x, y, retinal, fast_width=True, fast_pad=20, ret_maps=False):    

#     # fast width crops the mask to only the vessel
#     mask = retinal.vessel_image
#     M, N = mask.shape
#     if fast_width:
#         # Crop the image mask to make vectorised search below quicker
#         # We pad the image as we only crop the mask according to the vessel skeleton
#         # not the vessel itself
#         bin_img = np.zeros_like(mask)
#         bin_img[(x, y)] = 1
#         bin_props = measure.regionprops(measure.label(bin_img))[0]
#         bnd_coords = bin_props.bbox
#         y_st, y_en = max(0,bnd_coords[0]-fast_pad), min(N,bnd_coords[2]+fast_pad)
#         y_offset = np.abs(y_st - bnd_coords[0])
#         x_st, x_en = max(0,bnd_coords[1]-fast_pad), min(M,bnd_coords[3]+fast_pad)
#         x_offset = np.abs(x_st - bnd_coords[1])
#         vessel_map = mask[y_st:y_en, x_st:x_en]

#         # Make sure to offset vessel segment accordingly to the padding
#         v = [y, x]
#         varr = np.array(v).copy()
#         #if ret_maps:
#         #    return vessel_map, x, y, y_offset, x_offset, bnd_coords
#         varr[1] -= bnd_coords[0]
#         varr[0] -= bnd_coords[1]
#         x, y = varr[0], varr[1]
#         x += x_offset
#         y += y_offset
#         if ret_maps:
#             return vessel_map, x, y, y_offset, x_offset, bnd_coords
#         x, y = y, x
#     else:
#         vessel_map = mask.copy() 
#         if ret_maps:    
#             return vessel_map, x, y

#     # Carry on as normal
#     width_list = []
    
    
#     for i in range(0, len(x) - 1):
#         width = 0
#         width_matrix = 1
#         width_mask = np.zeros((vessel_map.shape))
#         width_cal = 0
        
#         while width_matrix:
#             width+=1
#             #print("CIRCLE")
#             cv2.circle(width_mask,(y[i],x[i]),radius=width,color=(255,255,255),thickness=-1)
#             #print("TEST1")
#             masked_vessel = vessel_map[width_mask>0]
#             #print("TEST2\n")
#             width_matrix = np.all(masked_vessel>0)
        
#         if np.sum(masked_vessel==0)==1:
#             width_cal = width*2
#         elif np.sum(masked_vessel==0)==2:
#             width_cal = width*2-1
#         elif np.sum(masked_vessel==0)==3:
#             width_cal = width*2-1
#         else:
#             width_cal = width*2

#         width_cal = width_cal*retinal.resolution
        
#         width_list.append(width_cal)
        
#     return width_list


def width_measurement(x, y, retinal, fast_width=True, fast_pad=20, ret_maps=False):    

    # fast width crops the mask to only the vessel
    mask = np.ascontiguousarray(retinal.vessel_image, dtype=np.uint8)
    M, N = mask.shape
    if fast_width:
        # Crop the image mask to make vectorised search below quicker
        # We pad the image as we only crop the mask according to the vessel skeleton
        # not the vessel itself
        bin_img = np.zeros_like(mask)
        bin_img[(x, y)] = 1
        bin_props = measure.regionprops(measure.label(bin_img))[0]
        bnd_coords = bin_props.bbox
        y_st, y_en = max(0,bnd_coords[0]-fast_pad), min(M,bnd_coords[2]+fast_pad)
        y_offset = np.abs(y_st - bnd_coords[0])
        x_st, x_en = max(0,bnd_coords[1]-fast_pad), min(N,bnd_coords[3]+fast_pad)
        x_offset = np.abs(x_st - bnd_coords[1])
        vessel_map = mask[y_st:y_en, x_st:x_en]

        # Make sure to offset vessel segment accordingly to the padding
        v = [y, x]
        varr = np.array(v).copy()
        #if ret_maps:
        #    return vessel_map, x, y, y_offset, x_offset, bnd_coords
        varr[1] -= bnd_coords[0]
        varr[0] -= bnd_coords[1]
        x, y = varr[0], varr[1]
        x += x_offset
        y += y_offset
        if ret_maps:
            return vessel_map, x, y, y_offset, x_offset, bnd_coords
        x, y = y, x
    else:
        vessel_map = mask.copy() 
        if ret_maps:    
            return vessel_map, x, y

    # Carry on as normal
    width_list = []
    
    
    for i in range(0, len(x) - 1):
        width = 0
        width_matrix = 1
        width_mask = np.ascontiguousarray(np.zeros_like(vessel_map), dtype=np.uint8)
        width_cal = 0
        
        while width_matrix:
            width+=1
            #print("CIRCLE")
            cv2.circle(width_mask,(y[i],x[i]),radius=width,color=(255,255,255),thickness=-1)
            #print("TEST1")
            masked_vessel = vessel_map[width_mask>0]
            #print("TEST2\n")
            width_matrix = np.all(masked_vessel>0)
        
        if np.sum(masked_vessel==0)==1:
            width_cal = width*2
        elif np.sum(masked_vessel==0)==2:
            width_cal = width*2-1
        elif np.sum(masked_vessel==0)==3:
            width_cal = width*2-1
        else:
            width_cal = width*2

        width_cal = width_cal*retinal.resolution
        
        width_list.append(width_cal)
        
    return width_list
    

def _create_circular_mask(scalex=11.48, center=None, img_shape=(768,768), grid_size=7000, radius=None, logging=[], verbose=True):
    """
    Given a center, radius and image shape, draw a filled circle
    as a binary mask.
    """
    # Force center
    h, w = img_shape
    if center is None: # use the middle of the image
        center = (int(h/2), int(w/2))

    # Work out pixel diameter
    if radius is not None:
        diameter = 2*radius
    else:
        diameter = grid_size / scalex
        radius = int(diameter//2)

    # Circular mask
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = (dist_from_center <= radius).astype(int)
    
    # Check to make sure square fits
    rprops = measure.regionprops(measure.label(mask))[0]
    if not np.allclose(rprops.axis_minor_length, rprops.axis_major_length):
        msg = "WARNING: Region of interest too large for image."
        logging.append(msg)
        if verbose:
            print(msg)
    
    return mask, logging



def _create_square_mask(scalex=11.48, center=None, img_shape=(768,768), grid_size=7000, width=None, logging=[], verbose=True):
    '''
    Create a squares-shaped binary mask
    '''
    # Force center
    h, w = img_shape
    if center is None: # use the middle of the image
        center = (int(h/2), int(w/2))

    # Size of each cell in mm, and width of grid in pixels
    if width is None:
        width = grid_size / scalex

    # Square mask
    Y, X = np.ogrid[:h, :w]
    upper_mask = Y - center[1] > width//2
    low_mask = Y - center[1] < -width//2
    left_mask = X - center[0] > width//2
    right_mask = X - center[0] < -width//2
    mask = upper_mask | low_mask | left_mask | right_mask

    # Check to make sure square fits
    mask = (~mask).astype(int)
    rprops = measure.regionprops(measure.label(mask))[0]
    if rprops.centroid != center:
        msg = "WARNING: Region of interest too large for image. "
        logging.append(msg)
        if verbose:
            print(msg)

    return mask, logging
    


def vessel_metrics(window: Window, 
                   window_size: tuple[int, int] = (768,768),
                   min_pixels_per_vessel: int = 10, 
                   vessel_type: str = "binary",
                   plot: bool = False,
                   sort_vessels: callable = _reorder_coords,
                   width_method: callable = width_measurement,
                   return_vessels: bool = False,
                   logging_list: list = []):
    """
    Re-write of tortuosity_measures.evaluate_window() to include only necessary code.
    """
    # collect outputs in a dictionary
    slo_dict = {}
    
    # Loop over windows
    all_avg_vessel_widths = []
    for i in range(0, window.shape[0], 1):
        
        # select grayscale window
        bw_window = window.windows[i, 0, :, :]
        
        # count vessel pixels and total number of pixels
        vessel_total_count = np.sum(bw_window==1) # =np.sum(window.vessel_image) for patch_size=image_dims
        pixel_total_count = bw_window.shape[0]*bw_window.shape[1]
    
        # create retina class
        retina = Retina(bw_window, 
                      "window{}" + window.filename, 
                      window.segmentation_path, 
                      window.resolution) # added store_path and resolution
        retina.skeletonization()
        
        # Compute FD, VD and Average width over whole image, equivalent to doing this inside loop
        # if we use the whole image as input
        #print("global")
        fractal_dimension, vessel_density, average_width_all = global_cal(window)
        #print("global DONE")
        slo_dict["fractal_dimension"] = fractal_dimension
        slo_dict["vessel_density"] = vessel_density
        slo_dict["average_global_calibre"] = average_width_all
    
        # detect individual vessels, similar to skelentonisation but detects individual vessels, and
        # splits them at any observed intersection
        vessels = detect_vessel_border(retina, ignored_pixels=1)
        #return vessels
    
        # ALL OF THESE INITIAL VALUES AREN'T ACTUALLY USED
        vessel_count = 0
        t2, t4, td, tcurve = 0, 0, 0, 0 

        #return vessels
    
        # Initialise vessel widths and count lists
        vessel_widths_list = []
        avg_vessel_widths = []
        vessel_count_list = []
    
        for i, vessel in enumerate(vessels):
    
            # If detected vessel is greater than the minimum number of pixels per vessel
            N = len(vessel[0])
            if N > min_pixels_per_vessel:
                vessel_count += 1
                
                # Work out length of current vessel
                #print(i, "CURVLENGTH")
                v_length = _curve_length(vessel[0], vessel[1], distance_measure="euclidean")

                # Check if it is greater than maximum length, and re-order and compute length again
                max_length = np.sqrt(2)*N
                if v_length > max_length:
                    #print(i, "SORT")
                    v_arr = np.array(vessel).T
                    v_arr = sort_vessels(v_arr, window_size)
                    #print(i, "DONE")
                    vessel = [list(v_arr[:,1]), list(v_arr[:,0])]
                    vessels[i] = vessel
                    v_length = _curve_length(vessel[0], vessel[1], distance_measure="euclidean")
                    try:
                        assert len(vessel[0]) == N, f"New size of vessel {len(vessel[0])} unequal to original size {N}."
                    except AssertionError as msg:
                        print(msg)

                    try:
                        assert v_length <= max_length, f"New length of vessel {v_length} greater than maximum length {max_length}."
                    except AssertionError as msg:
                        print(msg)
                    

                # tcurve is simply the pixel length of the vessel
                tcurve += v_length
    
                # t2 measures the curve length of the vessel divided by the chord length.
                # Removed to simplify metric output
                #print(i, "DISTANCE_TORT")
                #t2 += distance_measure_tortuosity(vessel[0], vessel[1])
    
                # Measure curvature based on discrete derivatives to estimate curvature, and integrate
                # along vessels, https://pubmed.ncbi.nlm.nih.gov/10193892/ (1999, 367 citations...)
                # Removed to simplify metric output
                #print(i, "SQCURVE_TORT")
                #t4 += squared_curvature_tortuosity(vessel[0], vessel[1])
    
                # td measures curve_chord_ratio for subvessel segments per inflection point 
                # and cumulatively add them, and scale by number of inflections and overall curve length
                # formula comes from https://ieeexplore.ieee.org/document/1279902 (2003, 16 citations...)
                #print(i, "TORT_DENSE")
                td += tortuosity_density(vessel[0], vessel[1])
                
                # Compute vessel width in window, and also average width
                #print(i, "WIDTH")
                vessel_widths = width_method(vessel[0], vessel[1], retina)
                vessel_widths_list.append(vessel_widths)
                avg_vessel_widths.append(sum(vessel_widths)/len(vessel_widths))
                vessel_count_list.append(vessel_count)
    
        # If detected more than one distinct vessel - this will always pass if we use the whole input image
        if vessel_count > 0:
    
            # Normalise tortuosity indexes by vessel_count
            t2 = t2/vessel_count
            t4 = t4/vessel_count
            td = td/vessel_count
    
            # ASSUMING PATCH SIZE IS THE WHOLE IMAGE:
            # Dimensionless vessel density of ROI window - this is also the same as vessel_density computed in tortuosity_measures.global_cal()
            # vessel_density = vessel_total_count/pixel_total_count
    
            # ASSUMING PATCH SIZE IS THE WHOLE IMAGE:
            # This is measuring the same thing as average_width computed in global_cal, but should be smaller as tcurve > np.sum(retina.np_image) 
            # as t_curve is measuring euclidean distances between pixels, rather than pixel counts. Moreover, this value is dimensionless 
            average_caliber = (vessel_total_count/tcurve)*retina.resolution
    
        # collect outputs
        # slo_dict["tortuosity_curvebychord"] = t2
        # slo_dict["tortuosity_sqcurvature"] = t4
        slo_dict["tortuosity_density"] = td
        slo_dict["average_local_calibre"] = average_caliber
        
        # Compatibility for multuple windows, this takes all individual, average vessel widths
        # and collects them into a single list
        all_avg_vessel_widths.append(avg_vessel_widths)
    all_avg_flatten = np.array([a for l in all_avg_vessel_widths for a in l])
    
    # Plotting the individual vessels on the vessel image
    if plot:
        fig, ax = plt.subplots(1,1,figsize=(6,6))
        ax.imshow(window.vessel_image)
        for vessel in vessels:
            y, x = vessel
            ax.plot(x,y)#,s=1)
        ax.set_axis_off()
        fig.tight_layout()

    # Do not calculate CRAE/CRVE if binary vessels.
    #print('CRAE/CRVE')
    if vessel_type != "binary":   
    
        # calculate the CRAE/CRVE with Knudtson calibre
        vtype = vessel_type[0].upper()
        sorted_vessel_widths_average = sorted(all_avg_flatten)[-6:]
        N_vessels = len(sorted_vessel_widths_average)

        # Error handle if detected less than 6 vessels, must be even number
        if N_vessels < 6:
            msg1 = f'        WARNING: Less than 6 vessels detected in zone. Please check segmentation. Returning -1 for CR{vtype}E.'
            msg2 = f'                 Note that this means AVR cannot be computed for this image'
            slo_dict["CRAE_Knudtson"] = -1
            slo_dict["CRVE_Knudtson"] = -1

            # log to user
            print(msg1)
            print(msg2)
            logging_list.append(msg1)
            logging_list.append(msg2)

        #  Compute calibre, taking into account number of available vessels
        else:
            
            w_first_artery_Knudtson_1, w_first_vein_Knudtson_1 = Knudtson_cal(sorted_vessel_widths_average[0],
                                                                              sorted_vessel_widths_average[5])
            
            w_first_artery_Knudtson_2, w_first_vein_Knudtson_2 = Knudtson_cal(sorted_vessel_widths_average[1],
                                                                              sorted_vessel_widths_average[4])
                
            w_first_artery_Knudtson_3, w_first_vein_Knudtson_3 = Knudtson_cal(sorted_vessel_widths_average[2],
                                                                              sorted_vessel_widths_average[3])
            
            CRAE_first_round = sorted([w_first_artery_Knudtson_1,
                                       w_first_artery_Knudtson_2,
                                       w_first_artery_Knudtson_3])
            CRVE_first_round = sorted([w_first_vein_Knudtson_1,
                                       w_first_vein_Knudtson_2,
                                       w_first_vein_Knudtson_3])
            
            if vessel_type=='artery': 
                w_second_artery_Knudtson_1, w_second_vein_Knudtson_1 = Knudtson_cal(CRAE_first_round[0],
                                                                                    CRAE_first_round[2])  
            
                CRAE_second_round = sorted([w_second_artery_Knudtson_1,CRAE_first_round[1]])
                CRAE_Knudtson,_ = Knudtson_cal(CRAE_second_round[0],CRAE_second_round[1])
                slo_dict["CRAE_Knudtson"] = CRAE_Knudtson
                slo_dict["CRVE_Knudtson"] = -1
            
            else:
                w_second_artery_Knudtson_1, w_second_vein_Knudtson_1 = Knudtson_cal(CRVE_first_round[0],
                                                                                    CRVE_first_round[2])  
            
                CRVE_second_round = sorted([w_second_vein_Knudtson_1,CRVE_first_round[1]])
                _,CRVE_Knudtson = Knudtson_cal(CRVE_second_round[0],CRVE_second_round[1])
                slo_dict["CRAE_Knudtson"] = -1
                slo_dict["CRVE_Knudtson"] = CRVE_Knudtson
        
    else:
        slo_dict["CRAE_Knudtson"] = -1
        slo_dict["CRVE_Knudtson"] = -1
    
    

    if return_vessels:
        return slo_dict, vessels, logging_list
    else:
        return slo_dict, logging_list
    


def measure_sloroi(binmap, fovea, od_centre, od_radius,
                   scalex=None, img_shape=(768,768), vessel_type="binary", 
                   roi_type="whole", distance=-1, method="fast",
                   return_vessels=False, plot=False, verbose=False):

    # od_radius tells us whether SLO is macula or optic disc-centred
    if od_radius is None:
        roi_center = fovea.copy()
    else:
        roi_center = od_centre.copy()

    # Build ROI mask based on distance, or zone
    logging = []
    whole_image = False
    if roi_type not in ["B", "C"]:
        # log = []
        if distance == 0 or roi_type == "whole" or scalex == None:
            whole_image = True
            distance = 0
            mask = np.ones(img_shape)
        else:
            distance *= 1e3
            macula_p = int(distance/scalex)
            if roi_type == "square":
                mask, log = _create_square_mask(scalex=scalex, center=roi_center, 
                                            img_shape=img_shape, grid_size=distance, verbose=verbose)
            elif roi_type == "circle":
                mask, log = _create_circular_mask(scalex=scalex, center=roi_center, 
                                            img_shape=img_shape, grid_size=distance, verbose=verbose)
    else:
        od_diameter = 2*od_radius
        if roi_type == "B":
            od_circ = _create_circular_mask(img_shape=img_shape, 
                                        radius=2*od_radius, 
                                        center=od_centre, verbose=verbose)[0]
            
            mask, log  = _create_circular_mask(img_shape=img_shape, 
                                                radius=3*od_radius, 
                                                center=od_centre, verbose=verbose)
            macula_p = 3*od_diameter
        elif roi_type == "C":
            od_circ = _create_circular_mask(img_shape=img_shape, 
                                        radius=od_radius, 
                                        center=od_centre, verbose=verbose)[0]
            
            mask, log = _create_circular_mask(img_shape=img_shape, 
                                            radius=5*od_radius, 
                                            center=od_centre, verbose=verbose)
            macula_p = 5*od_diameter
        mask -= od_circ
    #logging.extend(log)

    # create ROI
    roi = binmap * mask
    N = img_shape[0]

    # We originally cropped the image to speed up metric calc, but no need after
    # slowest parts (vessel geometrical ordering and width measurements are quicker now)
    # roi_large = False
    # if len(log) > 0:
    #     roi_large = log[-1] == 'WARNING: Region of interest too large for image.'
    # if distance != 0 and not whole_image and scalex is not None and not roi_large:
    #     sty, eny = max(0, roi_center[1]-macula_p//2), min(N-1, roi_center[1]+macula_p//2)
    #     stx, enx = max(0, roi_center[0]-macula_p//2), min(N-1, roi_center[0]+macula_p//2)
    #     roi = roi[sty:eny, stx:enx]

    # If unspecified scale, set to 1
    if scalex is None:
        scalex = 1

    # SET UP FOR AUTOMORPH INPUT
    # WINDOW_SIZE is the entire image.
    WINDOW_SIZE = roi.shape[0]
    WINDOW_SHAPE = (WINDOW_SIZE, WINDOW_SIZE)

    # This is used for ignoring vessels which are very short, or ignoring windows with low vessel pixel count. 
    # The latter is irrelevant when considering the whole image as the window.
    MIN_VESSEL_PIXEL = 10
    MIN_WINDOW_PIXEL = 15

    # return mask, roi
    
    # Instantiate retina class for ROI and create window(s). We also skeletonise the map so that 
    # window.np_image is the skeleton version but window.vessel_image is the original binary map
    roi_Retina = Retina(roi, ".", store_path=vessel_type, scalex=scalex)
    roi_Window = Window(roi_Retina, WINDOW_SIZE, min_pixels=MIN_WINDOW_PIXEL)
    roi_Window.skeletonization()

    # Measure vessel metrics on window(s) and extract with informative variable names
    # if method == "default":
    #     sort_vessels = _reorder_coords_2
    #     width_method = tortuosity_measures.width_measurement
    # elif method == "fast":
    #     sort_vessels = _reorder_coords
    #     width_method = width_measurement
    # elif method == 'automorph':
    #     sort_vessels = lambda x, y: x[:,[1,0]]
    #     width_method = tortuosity_measures.width_measurement

    # By default we use quicker geometrical ordering of vessels and quicker width measurement
    # relative to Automorph
    sort_vessels = _reorder_coords
    width_method = width_measurement
    output = vessel_metrics(roi_Window, WINDOW_SHAPE, MIN_VESSEL_PIXEL, vessel_type,    
                            plot=plot, sort_vessels=sort_vessels, width_method=width_method, 
                            return_vessels=return_vessels)
    logging.extend(output[-1])
    slo_dict = output[0]

    if return_vessels:
        return output[:2], logging, mask, roi
    return slo_dict, logging, mask, roi