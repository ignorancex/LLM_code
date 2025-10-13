import cv2
import torch
import numpy as np

def read_video(path_to_video: str):
    """
    Read video by frames using its path
    """
    
    # load video 
    cap = cv2.VideoCapture(path_to_video)
    
    width_original, height_original = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #
    fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    full_frames = []
    i = 0 # current frame

    while(cap.isOpened()):
        if i == frames:
            break

        ret, frame = cap.read()

        i += 1
        if ret==True:
            full_frames.append(frame)
            p = i * 100 / frames
        else:
            break
    
    cap.release()
    
    return full_frames, fps
    

def torch2image(torch_image: torch.tensor) -> np.ndarray:
    batch = False
    
    if torch_image.dim() == 4:
        torch_image = torch_image[:8]
        batch = True
    
    device = torch_image.device
    mean = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2).to(device)
    std = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2).to(device)
    
    denorm_image = (std * torch_image) + mean
    
    if batch:
        denorm_image = denorm_image.permute(0, 2, 3, 1)
    else:
        denorm_image = denorm_image.permute(1, 2, 0)
    
    np_image = denorm_image.detach().cpu().numpy()
    np_image = np.clip(np_image * 255., 0, 255).astype(np.uint8)
    
    if batch:
        return np.concatenate(np_image, axis=1)
    else:
        return np_image
        

def normalize_and_torch(image: np.ndarray, use_cuda = True) -> torch.tensor:
    """
    Normalize image and transform to torch
    """
    if use_cuda:
        image = torch.tensor(image.copy(), dtype=torch.float32).cuda()
    else:
        image = torch.tensor(image.copy(), dtype=torch.float32)
    if image.max() > 1.:
        image = image/255.
    
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = (image - 0.5) / 0.5

    return image

def copy_head_back(s, t, M):
    mask = np.ones_like(s)
    mask_tr = cv2.warpAffine(mask, cv2.invertAffineTransform(M), (t.shape[1], t.shape[0]), borderValue=0.0)
    mask_tr = cv2.erode(mask_tr, np.ones((10, 10)))
    mask_tr = cv2.GaussianBlur(mask_tr, (5, 5), 0)

    image_tr = cv2.warpAffine(s, cv2.invertAffineTransform(M), (t.shape[1], t.shape[0]), borderValue=0.0)
    res = (t * (1 - mask_tr) + image_tr * mask_tr).astype(np.uint8)
    return res