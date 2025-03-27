import numpy as np
import cv2

# model 1_4
# Roi with padding of the RGB where the flow magnitude is zero
# rgb = single frame
def roi_pad (rgb, flow):
    flow[..., 0] = cv2.normalize(flow[..., 0],None,0,255,cv2.NORM_MINMAX)
    flow[..., 1] = cv2.normalize(flow[..., 1],None,0,255,cv2.NORM_MINMAX)
    flow[:,:,0] -= np.mean(flow[:,:,0])
    flow[:,:,1] -= np.mean(flow[:,:,1])

    
    # Compute the magnitude of the flow
    magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
    thresh = np.mean(magnitude)
    magnitude[magnitude<thresh] = 0
    
    for c in range(3):
        rgb [:,:,c] = np.where(magnitude == 0, 0, rgb[:,:,c])
    return rgb

#model1_5
# Roi with a square padding with the ROI computed as in the paper RWF-2000
# rgb = single frame
def roi_pad_square (rgb, flow):
    np.random.seed(8)
    global last_x
    global last_y
    flow[..., 0] = cv2.normalize(flow[..., 0],None,0,255,cv2.NORM_MINMAX)
    flow[..., 1] = cv2.normalize(flow[..., 1],None,0,255,cv2.NORM_MINMAX)
    flow[:,:,0] -= np.mean(flow[:,:,0])
    flow[:,:,1] -= np.mean(flow[:,:,1])

    
    # Compute the magnitude of the flow
    magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
    thresh = np.mean(magnitude)
    magnitude[magnitude<thresh] = 0
    
    # If we are not in the last frame, compute the center of gravity as usually
    if (not np.all((magnitude == 0))):
        # calculate center of gravity of magnitude map and adding 0.001 to avoid empty value
        x_pdf = np.sum(magnitude, axis=1) + 0.001
        y_pdf = np.sum(magnitude, axis=0) + 0.001
        # normalize PDF of x and y so that the sum of probs = 1
        x_pdf /= np.sum(x_pdf)
        y_pdf /= np.sum(y_pdf)
        # randomly choose some candidates for x and y 
        x_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=x_pdf)
        y_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=y_pdf)
        # get the mean of x and y coordinates for better robustness
        x = int(np.mean(x_points))
        y = int(np.mean(y_points))
        x = max(56,min(x,167))
        y = max(56,min(y,167))
    # Otherwise, use the last_x and last_y from the previous frame
    else:
        x = last_x 
        y = last_y
    
    rgb_new = np.zeros([224,224,3],dtype=np.uint8)
    roi = rgb [x-56:x+56,y-56:y+56] 
    
    rgb_new [x-56:x+56,y-56:y+56] = roi
    
    last_x = x
    last_y = y
    #visualize(magnitude)
    #visualize(rgb_new)
    return rgb_new

#model1_6
# Roi as in the paper RWF-2000: zoom and rescaling of the RGB
# rgb = single frame
def roi (rgb, flow):
    np.random.seed(8)
    global last_x
    global last_y
    flow[..., 0] = cv2.normalize(flow[..., 0],None,0,255,cv2.NORM_MINMAX)
    flow[..., 1] = cv2.normalize(flow[..., 1],None,0,255,cv2.NORM_MINMAX)
    flow[:,:,0] -= np.mean(flow[:,:,0])
    flow[:,:,1] -= np.mean(flow[:,:,1])

    
    # Compute the magnitude of the flow
    magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
    thresh = np.mean(magnitude)
    magnitude[magnitude<thresh] = 0
    
    # If we are not in the last frame, compute the center of gravity as usually
    if (not np.all((magnitude == 0))):
        # calculate center of gravity of magnitude map and adding 0.001 to avoid empty value
        x_pdf = np.sum(magnitude, axis=1) + 0.001
        y_pdf = np.sum(magnitude, axis=0) + 0.001
        # normalize PDF of x and y so that the sum of probs = 1
        x_pdf /= np.sum(x_pdf)
        y_pdf /= np.sum(y_pdf)
        # randomly choose some candidates for x and y 
        x_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=x_pdf)
        y_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=y_pdf)
        # get the mean of x and y coordinates for better robustness
        x = int(np.mean(x_points))
        y = int(np.mean(y_points))
        x = max(56,min(x,167))
        y = max(56,min(y,167))
    # Otherwise, use the last_x and last_y from the previous frame
    else:
        x = last_x 
        y = last_y
    
    roi = rgb [x-56:x+56,y-56:y+56] 
    
    rgb_new = cv2.resize(roi, (224,224), interpolation=cv2.INTER_CUBIC)
    
    last_x = x
    last_y = y
    #visualize(magnitude)
    #visualize(rgb_new)
    return rgb_new

#model1_7
# Roi "exactly" as in the paper RWF-2000: zoom and rescaling of the RGB 
# rgb = multiple frames (segments)
def roi_video (rgb_segment, flow_segment, seed=True):
    
    if (seed):
        np.random.seed(8)
    
    magnitude = []
    for flow in flow_segment:
            flow[..., 0] = cv2.normalize(flow[..., 0],None,0,255,cv2.NORM_MINMAX)
            flow[..., 1] = cv2.normalize(flow[..., 1],None,0,255,cv2.NORM_MINMAX)
            flow[:,:,0] -= np.mean(flow[:,:,0])
            flow[:,:,1] -= np.mean(flow[:,:,1])
            magnitude.append(np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2))
            
    # Compute the magnitude of the flow
    magnitude = np.sum(magnitude, axis=0)
    #magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
    thresh = np.mean(magnitude)
    magnitude[magnitude<thresh] = 0

    # calculate center of gravity of magnitude map and adding 0.001 to avoid empty value
    x_pdf = np.sum(magnitude, axis=1) + 0.001
    y_pdf = np.sum(magnitude, axis=0) + 0.001
    # normalize PDF of x and y so that the sum of probs = 1
    x_pdf /= np.sum(x_pdf)
    y_pdf /= np.sum(y_pdf)

    # randomly choose some candidates for x and y 
    x_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=x_pdf)
    y_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=y_pdf)

    # get the mean of x and y coordinates for better robustness
    x = int(np.mean(x_points))
    y = int(np.mean(y_points))

    x = max(56,min(x,167))
    y = max(56,min(y,167))

    rgb_segment = np.array(rgb_segment)
    roi = rgb_segment [:,x-56:x+56,y-56:y+56,:] 
    rgb_segment_new = []
    for r in roi:
        rgb_segment_new.append(cv2.resize(r, (224,224), interpolation=cv2.INTER_CUBIC))

    #visualize(magnitude)
    #visualize(rgb_segment_new[0])
    return rgb_segment_new



#model1_8
# Same as before, but with the scale and zoom also on the flow
# rgb = multiple frames (segments)
def roi_video_and_flow (rgb_segment, flow_segment):
    np.random.seed(8)
    
    magnitude = []
    for flow in flow_segment:
            flow[..., 0] = cv2.normalize(flow[..., 0],None,0,255,cv2.NORM_MINMAX)
            flow[..., 1] = cv2.normalize(flow[..., 1],None,0,255,cv2.NORM_MINMAX)
            flow[:,:,0] -= np.mean(flow[:,:,0])
            flow[:,:,1] -= np.mean(flow[:,:,1])
            magnitude.append(np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2))
            
    # Compute the magnitude of the flow
    magnitude = np.sum(magnitude, axis=0)
    #magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
    thresh = np.mean(magnitude)
    magnitude[magnitude<thresh] = 0

    # calculate center of gravity of magnitude map and adding 0.001 to avoid empty value
    x_pdf = np.sum(magnitude, axis=1) + 0.001
    y_pdf = np.sum(magnitude, axis=0) + 0.001
    # normalize PDF of x and y so that the sum of probs = 1
    x_pdf /= np.sum(x_pdf)
    y_pdf /= np.sum(y_pdf)
        
    # randomly choose some candidates for x and y 
    x_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=x_pdf)
    y_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=y_pdf)

    # get the mean of x and y coordinates for better robustness
    x = int(np.mean(x_points))
    y = int(np.mean(y_points))

    x = max(56,min(x,167))
    y = max(56,min(y,167))

    rgb_segment = np.array(rgb_segment)
    roi = rgb_segment [:,x-56:x+56,y-56:y+56,:] 
    flow_segment = np.array(flow_segment)
    roi_flow = flow_segment [:,x-56:x+56,y-56:y+56,:]
    rgb_segment_new = []
    flow_segment_new = []
    for i in range(len(roi)):
        rgb_segment_new.append(cv2.resize(roi[i], (224,224), interpolation=cv2.INTER_CUBIC))
        flow_segment_new.append(cv2.resize(roi_flow[i], (224,224), interpolation=cv2.INTER_CUBIC))

    #visualize(flow_segment_new[0][:,:,0])
    #visualize(rgb_segment_new[0])
    return rgb_segment_new, flow_segment_new