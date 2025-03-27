# -*- coding: utf-8 -*-
"""
Created on Sat May  4 00:27:37 2024

@author: Murthy, L.R.D.,
"""

import numpy as np



def convert_to_unit_vector(angles):
    """
    Parameters
    ----------
    angles : Radians 

    Returns
    -------
    x, y, z : Unit vector components

    """
    
    x = -np.cos(angles[:, 0]) * np.sin(angles[:, 1])
    y = -np.sin(angles[:, 0])
    z = -np.cos(angles[:, 1]) * np.cos(angles[:, 0])
    norm = np.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    return x, y, z



def maptodisplay(preds, R_Matrix, Face_centers, mR, mT):
    """
    

    Parameters
    ----------
    preds : gaze angle predictions from the a deep learning model in the form of 
            Nx2x1 where N predictions are available in the format [pitch, yaw]
    R_Matrix : Face Rotational matrix, obtained usually during data normalization process
                Provided as part of the dataset in the name of "FcRotMat"
    Face_centers : Center of the face coordinate system based on the adopted convention. 
                    Since all data normalization is done using face as the reference coordinate system. 
                    Provided as part of the dataset in the name of "FcCenter"
    mR : Extrinsic rotational matrix between camera and the screen coordinate systems
                    Provided as part of the dataset in the name of "Camera_Screen_R"
    mT : Extrinsic translational vector between camera and the screen coordinate systems
                    Provided as part of the dataset in the name of "Camera_Screen_T"

    Returns
    -------
    screen_points_mm : Predicted gaze points on the display in millimeters. 
                        This variable would be of size Nx2x1. Each 2x1 represents 
                        horizontal and vertical coordinates of gaze point on screen.
                        To map it to pixels, one can use Screen_Resolution_X_mm, Screen_Resolution_Y_mm
                        and Screen_Resolution_X_px and Screen_Resolution_Y_px provided as part 
                        of the dataset in "session_metadata"

    """
    preds_vec = convert_to_unit_vector(preds)
    preds_vec = np.array(preds_vec).T
    monitor_norm_vec = np.array([0,0,1])
    monitor_normal = np.dot(mR, monitor_norm_vec)
    monitor_normal = np.reshape(monitor_normal, (3,1))
    mT = np.reshape(mT, (3,1))
    
   
    screen_points_mm = []
    for p, R, origin in zip(preds_vec, R_Matrix, Face_centers):

        gaze_v_cam = np.dot(np.linalg.inv(R), p)
        gaze_v_cam = np.array(gaze_v_cam)
          
        gaze_v_cam = np.reshape(gaze_v_cam, (3,1))
        origin = np.array(origin)
   
        origin = np.reshape(origin, (3,1))
              
        num = np.dot(monitor_normal.T, (mT-origin))
        denom = np.dot(monitor_normal.T, gaze_v_cam)
              
        gaze_len = num/denom
          
        ss = (gaze_len*gaze_v_cam)        
        ss = np.reshape(ss, (3,1))        
        gaze_pos_cam = origin + ss
        zz = np.dot(np.linalg.inv(mR), (gaze_pos_cam-mT))
        screen_points_mm.append(zz[:2])
   
    return screen_points_mm