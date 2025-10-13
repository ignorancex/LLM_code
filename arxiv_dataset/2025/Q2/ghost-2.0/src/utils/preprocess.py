import torch
import numpy as np
from skimage import transform as trans

def calc_arcface_borders(image_size=512):
    src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)
    arcface_src = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
         [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32)

    middle = src3[2, :][None, :]
    wide_src = (src3 - middle) * 0.5 + middle
    t = trans.SimilarityTransform()
    t.estimate(arcface_src, wide_src)
    corners = np.array([[0, 0, 1], [112, 112, 1]])
    corners_before = (t.params[:2, :] @ corners.T).T
    corners_before = (corners_before * (image_size / 112))
    corners_center = corners_before.mean(axis=0)
    corners_center[0] = image_size // 2
    corners_width = (corners_before[1, :] - corners_before[0, :]).mean() / 2
    corners_before2 = np.stack([corners_center - corners_width, corners_center + corners_width]).astype(int)
    return corners_before2


def blend_alpha(img, mask, background=None):
    
    if background is None:
        background = torch.zeros_like(img)
        
    return img * mask + background * (1 - mask)
    
def make_X_dict(X_arc, X_wide, X_mask=None, X_emotion=None, X_keypoints=None, segmentation=None):
    X_dict = {
        'source': {
            'face_arc': X_arc[:, :-1],
            'face_wide': X_wide[:, :-1]
        },
        'target': {
            'face_arc': X_arc[:, -1],
            'face_wide': X_wide[:, -1]
        }
    }
    
    if X_mask is not None:
        X_dict['source']['face_wide_mask'] = X_mask[: , :-1]
        X_dict['target']['face_wide_mask'] = X_mask[:, -1]

        X_dict['source']['face_wide'] = blend_alpha(X_dict['source']['face_wide'],
            X_dict['source']['face_wide_mask']
        )
        
        X_dict['target']['face_wide'] = blend_alpha(
            X_dict['target']['face_wide'],
            X_dict['target']['face_wide_mask']
        )

    if segmentation is not None:
        X_dict['source']['segmentation'] = segmentation[: , :-1]
        X_dict['target']['segmentation'] = segmentation[:, -1]
    
    if X_emotion is not None:
        X_dict['source']['face_emoca'] = X_emotion[:, :-1] 
        X_dict['target']['face_emoca'] = X_emotion[:, -1]    

    if X_keypoints is not None:
        X_dict['source']['keypoints'] = X_keypoints[:, :-1]
        X_dict['target']['keypoints'] = X_keypoints[:, -1]
        
    return X_dict