import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from inference.utils_infer.LoadPumaData_test import PumaTissueDataset_test
from torch.utils.data import DataLoader
from inference.utils_infer.LoadPumaData_test import load_data_tissue
from inference.utils_submit import validate_with_augmentations_and_ensembling
import os
import torch
import numpy as np
from pathlib import Path
import tifffile
def inference_tissue(data_path,res,fold = None,show_result = False):
    tissue_labels = ['tissue_white_background','tissue_stroma','tissue_blood_vessel','tissue_tumor','tissue_epidermis','tissue_necrosis']
    val_images = load_data_tissue(target_size= (1024,1024),data_path=data_path, tissue_labels= tissue_labels, im_size=(1024,1024))
    device2 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    val_set = PumaTissueDataset_test(val_images,
                                n_class1=6,
                                size1=(1024,1024),
                                transform=None,
                                     paths=res,
                                     device1=device2,)
    val_loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    dataloader = DataLoader(val_set, shuffle=False, drop_last=False, **val_loader_args)
    n_folds = 5



    # model_unet = smp.Unet(classes=11, encoder_weights=None, in_channels=4)
    model_unet = smp.UnetPlusPlus(
        encoder_name="resnet50",
        in_channels=4, classes=11, encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16),
        encoder_weights=None
    )



    with torch.no_grad():
        for image, pth in dataloader:
            tis_im = torch.tensor(tifffile.imread(pth[0])*50, dtype=torch.float32)
            tis_im /= 255
            image = torch.concatenate([image,tis_im.unsqueeze(0).unsqueeze(0)],dim=1)
            image = image.to(device=device2, dtype=torch.float32, memory_format=torch.channels_last)
            weights_list = []
            model_weight_path = 'foldNucleiUnetPP10'
            for folds in range(n_folds):
                dir_checkpoint = Path(
                    'Model_weights/'+ model_weight_path + str(
                        folds) + '/')

                PATH = str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(1))
                weights_list.append(PATH)

                # model1.to(device2)
            pred_segmenter = validate_with_augmentations_and_ensembling(model=model_unet,
                                                               image_tensor=image,
                                                               device=device2,
                                                               weights_list=weights_list,
                                                               )
            pred_segmenter_classes = torch.argmax(pred_segmenter, dim=0).cpu().numpy()

            mask_pred = np.copy(pred_segmenter_classes)
            mask_pred = mask_pred.astype(np.uint8)

            new_file_path =  pth[0].replace('.tif','.npy')
            np.save(new_file_path, mask_pred)
            if show_result:
                im = np.transpose(image[0].cpu().numpy(), (1, 2, 0))
                plt.subplot(1, 2, 1)
                plt.imshow(im)

                pm = mask_pred
                plt.subplot(1, 2, 2)
                plt.imshow(pm)

                plt.show()
            # print(np.max(mask_pred))
            # plt.imshow(mask_pred)
            # plt.show()
            # mask_pred = model1(image)
            # mask_pred = mask_pred.argmax(dim=1).cpu().numpy().astype(np.uint8)

            # min_val, max_val = mask_pred.min(), mask_pred.max()
            #
            # # Modify TIFF resolution metadata directly using tifffile
            # new_file_path =  pth[0].replace('.tif','_nuclei.tif')
            #
            #
            #
            # # nuclei_pred_file = "/opt/app/inference/nuclei_pred.txt"
            # # with open(nuclei_pred_file, "w") as f:
            # #     f.write(new_file_path)
            #
            #
            #
            # # Write the image with the correct resolution
            # with tifffile.TiffWriter(new_file_path) as tif:
            #     tif.write(
            #         mask_pred,
            #         resolution=(300, 300),  # Set resolution to 300 DPI for both X and Y
            #         extratags=[
            #             ('MinSampleValue', 'I', 1, int(1)),
            #             ('MaxSampleValue', 'I', 1, int(max_val)),
            #         ]
            #     )
            # # Verify the new resolution
            # with tifffile.TiffFile(new_file_path) as tif:
            #     for i, page in enumerate(tif.pages):
            #         print(f"Page {i} shape: {page.shape}, dtype: {page.dtype}")
            #         print(f"XResolution: {page.tags['XResolution'].value}")
            #         print(f"YResolution: {page.tags['YResolution'].value}")
            #         print(f"ResolutionUnit: {page.tags['ResolutionUnit'].value}")
            #         for tag in page.tags.values():
            #             name, value = tag.name, tag.value
            #             print(f"{name}: {value}")
            # # Debug: Verify file was saved
            # if os.path.exists(new_file_path):
            #     print(f"Successfully saved: {new_file_path}")
            # else:
            #     print(f"Failed to save: {new_file_path}")
            #
            # print(f'Wrote tissue file at: {new_file_path}')


if __name__ == '__main__':

    # tissue_pred_file = "/opt/app/inference/tissue_pred.txt"
    # with open(tissue_pred_file, "r") as f:
    #     tissue_path = f.read()
    # os.remove(tissue_pred_file)

    path = '/input/images/melanoma-wsi/'
    out_path = '/output/images/melanoma-tissue-mask-segmentation/'

    # path = '/home/ntorbati/PycharmProjects/PUMA-challenge-baseline-track12/test/'
    # out_path = '/home/ntorbati/PycharmProjects/PUMA-challenge-baseline-track12/output/images/melanoma-tissue-mask-segmentation/'


    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # all_tissue_data = np.sort([path + image for image in os.listdir(path)])
    images = [path + f for f in os.listdir(path) if f.endswith('.tif') and not f.endswith('_context.tif')]
    res = [out_path + f for f in os.listdir(path) if f.endswith('.tif') and not f.endswith('_context.tif')]
    # tissues = [tissue_pred_file + f for f in os.listdir(tissue_pred_file) if f.endswith('.tif') and not f.endswith('_context.tif')]
    inference_tissue(images, res)

    # for fold in range(5):
    #     path = '/home/ntorbati/PycharmProjects/PUMA-challenge-baseline-track2/test/fold' + str(fold) + '/'
    #     out_path = '/home/ntorbati/PycharmProjects/PUMA-challenge-baseline-track12/output/images/melanoma-tissue-mask-segmentation/'
    #
    # # tissue_pred_file = '/home/ntorbati/PycharmProjects/PUMA-challenge-baseline-track12/output/images/'
    #
    #     if not os.path.exists(out_path):
    #         os.makedirs(out_path)
    #
    #     # all_tissue_data = np.sort([path + image for image in os.listdir(path)])
    #     images = [path + f for f in os.listdir(path) if f.endswith('.tif') and not f.endswith('_context.tif')]
    #     res = [out_path + f for f in os.listdir(path) if f.endswith('.tif') and not f.endswith('_context.tif')]
    #     # tissues = [tissue_pred_file + f for f in os.listdir(tissue_pred_file) if f.endswith('.tif') and not f.endswith('_context.tif')]
    #     inference_tissue(images,res,fold)
    #     # micro_dices, mean_micro_dice = calculate_micro_dice_score_with_masks('/home/ntorbati/PycharmProjects/pythonProject/validation_ground_truth/fold00/', out_path,
    #     #                                                                      (1024,1024), eps=0.00001)
    #     # print(mean_micro_dice)
    #







