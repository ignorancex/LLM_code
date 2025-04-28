import numpy as np
import torch
from scipy.ndimage import zoom
import torch.nn as nn
from nilearn.image import load_img, new_img_like
from tqdm import tqdm

def test_single_nii(nii_fname, net, patch_size=[256, 256], output_fname='prediction.nii.gz',device='cpu'):

    image = load_img(nii_fname).get_fdata()

    if len(image.shape) == 3:
        prediction = np.zeros_like(image)
        for ind in tqdm(range(image.shape[2])):
            slice = image[:, :, ind]
            slice = 255.0*slice/np.max(slice)
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                # previous using 0
                slice = zoom(
                    slice, (patch_size[0] / x, patch_size[1] / y), order=3)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().to(device) #.cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(
                        out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[:,:,ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []

    v = new_img_like(nii_fname,np.uint16(prediction))
    v.to_filename(output_fname)
 

