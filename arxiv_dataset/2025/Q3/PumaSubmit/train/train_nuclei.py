import numpy as np
np.bool=np.bool_
import segmentation_models_pytorch as smp
import torch
from train_puma_dice import train_model
from sklearn.model_selection import KFold
from pathlib import Path
from utils import copy_data,adapt_checkpoint
from torch import nn
import cv2

if __name__ == '__main__':
    tissue_labels = ['tissue_white_background','tissue_stroma','tissue_blood_vessel','tissue_tumor','tissue_epidermis','tissue_necrosis']
    tissue_images_path ='/home/ntorbati/STORAGE/PumaDataset/01_training_dataset_tif_ROIs/'
    tissue_labels_path = '/home/ntorbati/STORAGE/PumaDataset/01_training_dataset_geojson_tissue/'
    final_target_size = (1024,1024)

    n_class = 11 # number of dataset classes

    # fine tune or transfer learning specs
    fine_tune = False # if fineTune from another dataset with different number of classes
    use_necros = True # we have used MUTILS panoptic dataset in our training to increase method performance for Necrosis tissue
    progressive = False # if True model trains in a progressive way. First. last layer, second, Decoder layer, third, Encoder layer
    parallel = False # parallel GPU

    device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 4

    val_percent = 0.2

    if use_necros:
        image_data_necros = np.load('/home/ntorbati/STORAGE/PumaDataset/1024_ims/ims_panopt_nuclei.npy')
        image_data_necros = image_data_necros[0:100]


        train_images_necros = np.zeros((image_data_necros.shape[0], final_target_size[0], final_target_size[1]),dtype=np.uint8)
        for ind in range(image_data_necros.shape[0]):
            tis_path = '/home/ntorbati/STORAGE/PanopticDataset/tissues_panopt/' + str(ind) + '.tif'
            im = cv2.cvtColor(cv2.imread(tis_path), cv2.COLOR_BGR2GRAY)
            train_images_necros[ind] = im*50
        image_data_necros = np.concatenate((image_data_necros, train_images_necros[:,:,:,np.newaxis]), axis=3)



        mask_data_necros = np.load('/home/ntorbati/STORAGE/PumaDataset/1024_ims/masks_panopt_nuclei.npy')
        mask_data_necros = mask_data_necros[0:100]

        res = 880
        image_data_necros = image_data_necros[:, 0:res, 0:res, :]
        mask_data_necros = mask_data_necros[:, 0:res, 0:res]

        image_data_necros = np.concatenate((np.array([cv2.resize(img[:,:,0:3], final_target_size, interpolation=cv2.INTER_LINEAR) for img in image_data_necros]),
                                                       np.array([cv2.resize(img[:,:,3], final_target_size, interpolation=cv2.INTER_NEAREST) for img in image_data_necros])[:,:,:,np.newaxis]),
                                           axis=3)

        mask_data_necros = np.array([cv2.resize(mask, final_target_size, interpolation=cv2.INTER_NEAREST) for mask in mask_data_necros])



    image_data = np.load('/home/ntorbati/STORAGE/PumaDataset/1024_ims/ims.npy')
    mask_data = np.load('/home/ntorbati/STORAGE/PumaDataset/1024_ims/masks_nuclei_10classes.npy')



    image_data_metas = image_data[0:102]
    mask_data_metas = mask_data[0:102,:,:,1]


    image_data_primary = image_data[103:]
    mask_data_primary = mask_data[103:,:,:,1]


    image_data = np.empty(0)
    mask_data = np.empty(0)







    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    indices_metas = np.arange(image_data_metas.shape[0])
    indices_primary = np.arange(image_data_primary.shape[0])


    splits_metas = list(kf.split(indices_metas))
    splits_primary = list(kf.split(indices_primary))


    for folds in [1,2,3,4]:#range(n_folds):
        print('training fold ', str(folds))
        train_index_primary = splits_primary[folds][0]
        val_index_primary = splits_primary[folds][1]
        print(val_index_primary)
        train_index_metas = splits_metas[folds][0]
        val_index_metas = splits_metas[folds][1]


        # here we stick the 4 channel from tissue predictions
        train_images_tissues = np.zeros((len(train_index_primary), final_target_size[0], final_target_size[1]),dtype=np.uint8)
        for ind in range(len(train_index_primary)):
            tis_path = '/home/ntorbati/PycharmProjects/pythonProject/validation_prediction/PrimarytissueFinal/all/training_set_primary_roi_' + f"{train_index_primary[ind]+1:03d}" + '.tif'
            im = cv2.cvtColor(cv2.imread(tis_path), cv2.COLOR_BGR2GRAY)
            train_images_tissues[ind] = im*50
        train_data_primary = np.concatenate((image_data_primary[train_index_primary], train_images_tissues[:,:,:,np.newaxis]), axis=3)

        val_images_tissues = np.zeros((len(val_index_primary), final_target_size[0], final_target_size[1]),dtype=np.uint8)
        for ind in range(len(val_index_primary)):
            tis_path = '/home/ntorbati/PycharmProjects/pythonProject/validation_prediction/PrimarytissueFinal/all/training_set_primary_roi_' + f"{val_index_primary[ind]+1:03d}" + '.tif'
            im = cv2.cvtColor(cv2.imread(tis_path), cv2.COLOR_BGR2GRAY)
            val_images_tissues[ind] = im*50
        val_data_primary = np.concatenate((image_data_primary[val_index_primary], val_images_tissues[:,:,:,np.newaxis]), axis=3)

        train_images_tissues = np.zeros((len(train_index_metas), final_target_size[0], final_target_size[1]),dtype=np.uint8)
        for ind in range(len(train_index_metas)):
            tis_path = '/home/ntorbati/PycharmProjects/pythonProject/validation_prediction/PrimarytissueFinal/all/training_set_metastatic_roi_' + f"{train_index_metas[ind]+1:03d}" + '.tif'
            im = cv2.cvtColor(cv2.imread(tis_path), cv2.COLOR_BGR2GRAY)
            train_images_tissues[ind] = im*50
        train_data_metas = np.concatenate((image_data_metas[train_index_metas], train_images_tissues[:,:,:,np.newaxis]), axis=3)

        val_images_tissues = np.zeros((len(val_index_metas), final_target_size[0], final_target_size[1]),dtype=np.uint8)
        for ind in range(len(val_index_metas)):
            tis_path = '/home/ntorbati/PycharmProjects/pythonProject/validation_prediction/PrimarytissueFinal/all/training_set_metastatic_roi_' + f"{val_index_metas[ind]+1:03d}" + '.tif'
            im = cv2.cvtColor(cv2.imread(tis_path), cv2.COLOR_BGR2GRAY)
            val_images_tissues[ind] = im*50
        val_data_metas = np.concatenate((image_data_metas[val_index_metas], val_images_tissues[:,:,:,np.newaxis]), axis=3)


        val_images = np.concatenate((val_data_metas,val_data_primary),axis=0)
        val_masks = np.concatenate((mask_data_metas[val_index_metas], mask_data_primary[val_index_primary]), axis=0)
        print('necros area = ', np.sum(np.where(val_masks == 5)))
        train_images = np.concatenate((train_data_metas,train_data_primary),axis=0)##
        train_masks = np.concatenate((mask_data_metas[train_index_metas], mask_data_primary[train_index_primary]), axis=0)##



        if use_necros:
            train_images = np.concatenate((train_images,image_data_necros),axis=0)
            train_masks = np.concatenate((train_masks,mask_data_necros),axis=0)


        dir_checkpoint = Path('E:/b/Model_weights/foldNucleiUnetPP10from4' + str(folds) + '/')
        val_save_path = '/home/ntorbati/PycharmProjects/pythonProject/validation_ground_truth/foldUNPP10from4' + str(folds)
        val_save_path1 = '/home/ntorbati/PycharmProjects/pythonProject/validation_images/foldUNPP10from4' + str(folds)
        # output_folder = '/home/ntorbati/PycharmProjects/pythonProject/validation_prediction/foldUNP10' + str(folds)
        output_folder = '/home/ntorbati/PycharmProjects/pythonProject/validation_prediction/NucleiP10from4'+ str(folds)


        copy_data(validation_indices = val_index_primary, data_path = tissue_labels_path,data_path1= tissue_images_path, save_path = val_save_path,save_path1 = val_save_path1, data_type = 'primary')
        copy_data(validation_indices = val_index_metas, data_path = tissue_labels_path,data_path1=tissue_images_path, save_path = val_save_path,save_path1 = val_save_path1, data_type = 'metastatic')

        train_indexes = np.linspace(0, len(train_images)-1, len(train_images))

        class_weights = [
            0.0012,
            0.0585,  # nuclei_endothelium
            0.2746,  # nuclei_plasma_cell
            0.1312,  # nuclei_stroma
            0.0012,  # nuclei_tumor
            0.0585,  # nuclei_histiocyte
            0.1569,  # nuclei_apoptosis
            0.0247,  # nuclei_epithelium
            0.3466,  # nuclei_melanophage
            0.1843,  # nuclei_neutrophil
            0.0236  # nuclei_lymphocyte
        ]

        class_weights = torch.tensor(class_weights, device=device2,dtype=torch.float16)
        iters = [80]
        lr = 1e-6

        scals = 0


        # model1 = smp.Unet(classes=n_class,in_channels=4)
        model1 = smp.UnetPlusPlus(
            encoder_name="resnet50", encoder_weights="imagenet",
            in_channels=4, classes=n_class, encoder_depth=5,
            decoder_channels=(256, 128, 64, 32, 16),
        )

        fineTune_PATH = "/home/ntorbati/PycharmProjects/pythonProject/E:/b/Model_weights/foldNucleiUnetPP4" + str(folds) + "/checkpoint_epoch1.pth"

        cp = torch.load(fineTune_PATH, weights_only=True,map_location="cuda:0")
        if "module." in list(cp.keys())[0]:
            cp = {k.replace("module.", ""): v for k, v in cp.items()}

        cp = adapt_checkpoint(cp, model1)
        model1.load_state_dict(cp, strict=False)  # strict=True to ensure all trained layers match

        if parallel:
            model1 = nn.DataParallel(model1)
        model1.to(device2)
        model1.n_classes = n_class

        target_size = final_target_size
        size = target_size[0]

        model1 = train_model(
        model = model1,
        device = device2,
        epochs = iters[scals],
        batch_size = 3,
        learning_rate = lr,
        val_percent = 0.2,
        save_checkpoint = True,
        img_scale = 0.5,
        amp = False,
        weight_decay=0.7,  # learning rate decay rate
        momentum = 0.999,
        gradient_clipping = 1.0,
        target_siz=target_size,
        n_class=n_class,
        image_data1=train_images,
        mask_data1=train_masks,
        val_images = val_images,
        val_masks = val_masks,
        class_weights = class_weights,
        augmentation=True,# default None
        val_batch=1,
        early_stopping=100,
        ful_size=final_target_size,
            val_augmentation=True,
            train_indexes = train_indexes,
            input_folder=val_save_path1,
            output_folder=output_folder,
            ground_truth_folder=val_save_path,
            folds = n_folds,
            dir_checkpoint=dir_checkpoint,
            logg=False,
            progressive=progressive,
            model_name='unet',
            val_sleep_time=0,
            nuclei = True,
            # stick_tissue = False,
            # nuclei=10,
        )
