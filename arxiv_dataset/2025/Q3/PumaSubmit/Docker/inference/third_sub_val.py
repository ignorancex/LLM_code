import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from inference.utils_infer.LoadPumaData_test import PumaTissueDataset_test
from torch.utils.data import DataLoader
from inference.utils_infer.LoadPumaData_test import load_data_tissue
from inference.utils_submit import validate_with_augmentations_and_ensembling
import os
import torch
import numpy as np
import tifffile
from pathlib import Path
from utils_submit import compute_puma_dice_micro_dice_prediction
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig
import json
import torch.nn.functional as F


def inference_tissue(data_path,res):
    tissue_labels = ['tissue_white_background','tissue_stroma','tissue_blood_vessel','tissue_tumor','tissue_epidermis','tissue_necrosis']
    val_images = load_data_tissue(target_size= (1024,1024),data_path=data_path, tissue_labels= tissue_labels, im_size=(1024,1024))

    val_set = PumaTissueDataset_test(val_images,
                                n_class1=6,
                                size1=(1024,1024),
                                transform=None,
                                     paths=res)
    device2 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    dataloader = DataLoader(val_set, shuffle=False, drop_last=False, **val_loader_args)
    # model1 = smp.Unet(classes=6,encoder_weights=None,)
    # PATH = "/opt/app/inference/Model_weights/checkpointUnet.pth"
    # # PATH = "/home/ntorbati/PycharmProjects/pythonProject/inference/Model_weights/checkpointUnet.pth"
    #
    # model1.load_state_dict(torch.load(PATH, weights_only=True))
    # model1.to(device2)
    # model1.eval()
    n_folds = 5

    config_path = "custom_segformer_config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config = SegformerConfig.from_dict(config_dict)

    # Initialize the model with custom config
    model_segformer = SegformerForSemanticSegmentation(config)

    model_blood = smp.Unet(classes=6,encoder_weights=None,)

    for image, pth in dataloader:

        # else:
        #     model_weight_path = 'foldRawMetastatic'

        image = image.to(device=device2, dtype=torch.float32, memory_format=torch.channels_last)
        weights_list = []
        model_weight_path = 'foldAllSegformer0'
        for folds in range(n_folds):
            dir_checkpoint = Path(
                'Model_weights/'+ model_weight_path + str(
                    folds) + '/')

            PATH = str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(1))
            weights_list.append(PATH)

            # model1.to(device2)
        with torch.no_grad():
            pred1 = validate_with_augmentations_and_ensembling(model=model_segformer,
                                                               image_tensor=image,
                                                               device=device2,
                                                               weights_list=weights_list,
                                                               )
            pred1[2] = 0 * pred1[2]
            pred1[7] = 0 * pred1[7]
            pred1 = torch.argmax(pred1, dim=0).cpu().numpy()
            pred1[pred1 > 5] = pred1[pred1 > 5] - 5

        weights_list = []
        model_weight_path = 'foldRawPrimary'

        for folds in range(n_folds):
            dir_checkpoint = Path(
                'Model_weights/'+ model_weight_path + str(
                    folds) + '/')

            PATH = str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(1))
            weights_list.append(PATH)

            # model1.to(device2)
        with torch.no_grad():
            pred2 = validate_with_augmentations_and_ensembling(model=model_blood,
                                                               image_tensor=image,
                                                                device = device2,
                                                                weights_list = weights_list,
                                                                )
            pred2 = torch.argmax(pred2, dim=0).cpu().numpy()
            pred2[pred2 > 5] = pred2[pred2 > 5] - 5
            pred1[pred2 == 2] = 2

        mask_pred = np.copy(pred1)
        mask_pred = mask_pred.astype(np.uint8)
        # mask_pred = model1(image)
        # mask_pred = mask_pred.argmax(dim=1).cpu().numpy().astype(np.uint8)

        min_val, max_val = mask_pred.min(), mask_pred.max()

        # Modify TIFF resolution metadata directly using tifffile
        new_file_path =  pth[0]

        # Write the image with the correct resolution
        with tifffile.TiffWriter(new_file_path) as tif:
            tif.write(
                mask_pred,
                resolution=(300, 300),  # Set resolution to 300 DPI for both X and Y
                extratags=[
                    ('MinSampleValue', 'I', 1, int(1)),
                    ('MaxSampleValue', 'I', 1, int(max_val)),
                ]
            )
        # Verify the new resolution
        with tifffile.TiffFile(new_file_path) as tif:
            for i, page in enumerate(tif.pages):
                print(f"Page {i} shape: {page.shape}, dtype: {page.dtype}")
                print(f"XResolution: {page.tags['XResolution'].value}")
                print(f"YResolution: {page.tags['YResolution'].value}")
                print(f"ResolutionUnit: {page.tags['ResolutionUnit'].value}")
                for tag in page.tags.values():
                    name, value = tag.name, tag.value
                    print(f"{name}: {value}")
        # Debug: Verify file was saved
        if os.path.exists(new_file_path):
            print(f"Successfully saved: {new_file_path}")
        else:
            print(f"Failed to save: {new_file_path}")

        print(f'Wrote tissue file at: {new_file_path}')
        # print(mask_pred.shape)
        # print(np.max(mask_pred))
        for i in range(1):
            print('new image' + str(i))

            im = np.transpose(image[i].cpu().numpy(), (1, 2, 0))
            plt.subplot(1, 2, 1)
            plt.imshow(im)

            pm = mask_pred
            plt.subplot(1, 2, 2)
            plt.imshow(pm)

            plt.show()


if __name__ == '__main__':
    # save_segformer_config(6)
    path = '/input/images/melanoma-wsi/'
    out_path = '/output/images/melanoma-tissue-mask-segmentation/'
    # out_path = '/home/ntorbati/PycharmProjects/PUMA-challenge-baseline-track12/output/images/melanoma-tissue-mask-segmentation/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # path = '/home/ntorbati/PycharmProjects/PUMA-challenge-baseline-track2/inference/input/images/melanoma-wsi/'
    all_tissue_data = np.sort([path + image for image in os.listdir(path)])
    images = [path + f for f in os.listdir(path) if f.endswith('.tif') and not f.endswith('_context.tif')]
    res = [out_path + f for f in os.listdir(path) if f.endswith('.tif') and not f.endswith('_context.tif')]
    inference_tissue(images,res)