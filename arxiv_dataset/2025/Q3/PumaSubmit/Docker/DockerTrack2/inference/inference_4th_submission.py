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
from utils_submit import calculate_micro_dice_score_with_masks
from utils_submit import save_segformer_config
def inference_tissue(data_path,res,fold = None, show_result = False):
    tissue_labels = ['tissue_white_background','tissue_stroma','tissue_blood_vessel','tissue_tumor','tissue_epidermis','tissue_necrosis']
    val_images = load_data_tissue(target_size= (1024,1024),data_path=data_path, tissue_labels= tissue_labels, im_size=(1024,1024))
    # val_images = np.load('/home/ntorbati/STORAGE/PumaDataset/1024_ims/ims_panopt.npy')
    device2 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    val_set = PumaTissueDataset_test(val_images,
                                n_class1=6,
                                size1=(1024,1024),
                                transform=None,
                                     paths=res,
                                     device1=device2)
    val_loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    dataloader = DataLoader(val_set, shuffle=False, drop_last=False, **val_loader_args)
    n_folds = 5


    # classifier configs
    config_path = "custom_segformer_config11.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config = SegformerConfig.from_dict(config_dict)

    # Initialize the model with custom config
    model_classifier = SegformerForSemanticSegmentation(config)

    # classifier configs
    config_path = "custom_segformer_config6.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config = SegformerConfig.from_dict(config_dict)

    # Initialize the model with custom config
    model_segmenter = SegformerForSemanticSegmentation(config)


    model_unet = smp.Unet(classes=6, encoder_weights=None, )

    with torch.no_grad():
        for image, pth in dataloader:
            image = image.to(device=device2, dtype=torch.float32, memory_format=torch.channels_last)
            weights_list = []
            model_weight_path = 'classifier'
            for folds in range(n_folds):
                dir_checkpoint = Path(
                    'Model_weights/'+ model_weight_path + str(
                        folds) + '/')

                PATH = str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(1))
                weights_list.append(PATH)

                # model1.to(device2)
            pred_classifier = validate_with_augmentations_and_ensembling(model=model_classifier,
                                                               image_tensor=image,
                                                               device=device2,
                                                               weights_list=weights_list,
                                                               )
            pred_classifier_classes = torch.argmax(pred_classifier, dim=0).cpu().numpy()

            if (pred_classifier_classes == 4).sum() > 0:
                model_weight_path = 'Primary'
            elif ((pred_classifier_classes< 6).sum() - (pred_classifier_classes ==  0).sum()) > (pred_classifier_classes> 5).sum():
                model_weight_path = 'Primary'
            else:
                model_weight_path = 'Metastatic'



            tissue_type = pth[0].replace('.tif','.txt')
            with open(tissue_type, "w") as f:
                f.write(model_weight_path)


            weights_list = []
            for folds in range(n_folds):
                dir_checkpoint = Path(
                    'Model_weights/'+ model_weight_path + str(
                        folds) + '/')

                PATH = str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(1))
                weights_list.append(PATH)

                # model1.to(device2)
            pred_segmenter = validate_with_augmentations_and_ensembling(model=model_segmenter,
                                                               image_tensor=image,
                                                               device=device2,
                                                               weights_list=weights_list,
                                                               )
            pred_segmenter[2] = 0*pred_segmenter[2]


            weights_list = []
            model_weight_path = 'unet'

            for folds in range(n_folds):
                dir_checkpoint = Path(
                    'Model_weights/'+ model_weight_path + str(
                        folds) + '/')

                PATH = str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(1))
                weights_list.append(PATH)

                # model1.to(device2)
            pred_unet = validate_with_augmentations_and_ensembling(model=model_unet,
                                                               image_tensor=image,
                                                                device = device2,
                                                                weights_list = weights_list,
                                                                )
            pred_unet_classes = torch.argmax(pred_unet, dim=0).cpu().numpy()
            pred_unet_classes[pred_unet_classes>5] = pred_unet_classes[pred_unet_classes > 5] - 5



            #ensemble epidermis
            pred_segmenter[4] = 0.5*pred_segmenter[4] + 0.5*pred_unet[4]
            pred_segmenter[5] = 0.5*pred_segmenter[5] + 0.5*pred_unet[5]
            # pred_segmenter[2] = 0.5*pred_segmenter[2] + 0.5*pred_unet[2]#+ 0.33*pred_classifier[10]

            #replace blood from unet
            pred_segmenter_classes = torch.argmax(pred_segmenter, dim=0).cpu().numpy()
            pred_segmenter_classes[pred_unet_classes == 2] = pred_unet_classes[pred_unet_classes == 2]

            mask_pred = np.copy(pred_segmenter_classes)
            mask_pred = mask_pred.astype(np.uint8)
            # mask_pred = model1(image)
            # mask_pred = mask_pred.argmax(dim=1).cpu().numpy().astype(np.uint8)
            # plt.imshow(mask_pred)
            # plt.show()
            #
            # if (np.sum(mask_pred==5)>0):
            #     print('new necrosis')

            min_val, max_val = mask_pred.min(), mask_pred.max()

            # Modify TIFF resolution metadata directly using tifffile
            new_file_path =  pth[0]

            if show_result:
                im = np.transpose(image[0].cpu().numpy(), (1, 2, 0))
                plt.subplot(1, 2, 1)
                plt.imshow(im)

                pm = mask_pred
                plt.subplot(1, 2, 2)
                plt.imshow(pm)

                plt.show()

            # tissue_pred_file = pth[0].replace('.tif','.txt')
            #
            # # tissue_pred_file = "/opt/app/inference/tissue_pred.txt"
            #
            # with open(tissue_pred_file, "w") as f:
            #     f.write(new_file_path)



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
            # with tifffile.TiffFile(new_file_path) as tif:
            #     for i, page in enumerate(tif.pages):
            #         print(f"Page {i} shape: {page.shape}, dtype: {page.dtype}")
            #         print(f"XResolution: {page.tags['XResolution'].value}")
            #         print(f"YResolution: {page.tags['YResolution'].value}")
            #         print(f"ResolutionUnit: {page.tags['ResolutionUnit'].value}")
            #         for tag in page.tags.values():
            #             name, value = tag.name, tag.value
            #             print(f"{name}: {value}")
            # Debug: Verify file was saved
            # if os.path.exists(new_file_path):
            #     print(f"Successfully saved: {new_file_path}")
            # else:
            #     print(f"Failed to save: {new_file_path}")
            #
            # print(f'Wrote tissue file at: {new_file_path}')
            # print(mask_pred.shape)
            # print(np.max(mask_pred))
            # for i in range(1):
            #     print('new image' + str(i))
            #
            #     im = np.transpose(image[i].cpu().numpy(), (1, 2, 0))
            #     plt.subplot(1, 2, 1)
            #     plt.imshow(im)
            #
            #     pm = mask_pred
            #     plt.subplot(1, 2, 2)
            #     plt.imshow(pm)
            #
            #     plt.show()
    #
    # print('classifier_acc = ' ,true_cases/all_cases)


if __name__ == '__main__':
    # save_segformer_config(6)
    path = '/input/images/melanoma-wsi/'
    out_path = '/output/images/melanoma-tissue-mask-segmentation/'

    # path = '/home/ntorbati/PycharmProjects/PUMA-challenge-baseline-track12/test/'
    # out_path = '/home/ntorbati/PycharmProjects/PUMA-challenge-baseline-track12/output/images/melanoma-tissue-mask-segmentation/'


    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # all_tissue_data = np.sort([path + image for image in os.listdir(path)])
    images = [path + f for f in os.listdir(path) if f.endswith('.tif') and not f.endswith('_context.tif')]
    res = [out_path + f for f in os.listdir(path) if f.endswith('.tif') and not f.endswith('_context.tif')]
    inference_tissue(images, res)

    # for fold in [0]:#range(5):
    #     path = '/home/ntorbati/PycharmProjects/pythonProject/validation_images/foldAllSegformer' + str(fold) + '/'#'/home/ntorbati/PycharmProjects/PUMA-challenge-baseline-track2/test/fold' + str(fold) + '/'
    #     out_path = '/home/ntorbati/PycharmProjects/PUMA-challenge-baseline-track12/output/images/melanoma-tissue-mask-segmentation/'
    #
    #     #
    #
    #     if not os.path.exists(out_path):
    #         os.makedirs(out_path)
    #
    #     # all_tissue_data = np.sort([path + image for image in os.listdir(path)])
    #     images = [path + f for f in os.listdir(path) if f.endswith('.tif') and not f.endswith('_context.tif')]
    #     res = [out_path + f for f in os.listdir(path) if f.endswith('.tif') and not f.endswith('_context.tif')]
    #     inference_tissue(images,res,fold)
    #     # micro_dices, mean_micro_dice = calculate_micro_dice_score_with_masks('/home/ntorbati/PycharmProjects/pythonProject/validation_ground_truth/fold00/', out_path,
    #     #                                                                      (1024,1024), eps=0.00001)
    #     # print(mean_micro_dice)
