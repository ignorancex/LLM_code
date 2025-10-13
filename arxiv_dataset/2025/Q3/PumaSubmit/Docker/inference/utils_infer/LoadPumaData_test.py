import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
from skimage.draw import polygon2mask
import torch
import cv2
from torch.utils.data import Dataset

def load_data_tissue(target_size = (128,128), data_path = '',tissue_labels = None,im_size = (1024,1024)):

    X_all = []
    for image in data_path:
        print(image)
        X_all.append(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB))
    X_all = np.array(X_all)
    X_all = X_all.astype('float32')
    return X_all


class PumaTissueDataset_test(Dataset):
    def __init__(self,
                 imgs1,
                 n_class1 = 6,
                 size1 = (1024,1024),
                 device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 transform = None,
                 mode="valid",
                 target_size = (1024,1024),
                 paths= ''
                 ):
        self.imgs = imgs1
        self.paths = paths
        self.n_class = n_class1
        self.image = torch.zeros((3,size1[0],size1[1]), device=device1)
        # self.mask = torch.zeros((n_class1,size1[0],size1[1]), device=device1)
        self.size = size1
        self.target_size = target_size
        self.device = device1
        self.transform = transform
        self.mode = mode
        if transform is not None:
            self.setup_augmentor(0)
        return

    def setup_augmentor(self, seed):
        self.augmentor = self.__get_augmentation(self.mode, seed)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])
        return



    def __len__(self):
        lent = len(self.imgs)
        return lent

    def __getitem__(self, idx):
        image = self.imgs[idx].astype("float32")
        # if self.target_size[0]!=self.size[0]:
        # image = cv2.resize(image, self.size)
        # mask = 255*np.ones((self.size[0],self.size[1],3),dtype=np.uint8)
        # if self.transform is not None:
        #     if self.shape_augs is not None:
        #         shape_augs = self.shape_augs.to_deterministic()
        #         image = shape_augs.augment_image(image)
        #         mask = shape_augs.augment_image(mask)
        #         image = (255-mask) + image
        #
        #     if self.input_augs is not None:
        #         input_augs = self.input_augs.to_deterministic()
        #         image = input_augs.augment_image(image)
        image = image/255
        image = np.transpose(image, (2, 0, 1))
        image = np.copy(image)
        pth = self.paths[idx]






        return image,pth

    def __get_augmentation(self, mode, rng):
        if mode == "train":
            shape_augs = [
                # * order = ``0`` -> ``cv2.INTER_NEAREST``
                # * order = ``1`` -> ``cv2.INTER_LINEAR``
                # * order = ``2`` -> ``cv2.INTER_CUBIC``
                # * order = ``3`` -> ``cv2.INTER_CUBIC``
                # * order = ``4`` -> ``cv2.INTER_CUBIC``
                # ! for pannuke v0, no rotation or translation, just flip to avoid mirror padding

                iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # translate by -A to +A percent (per axis)
                    translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                    shear=(-5, 5),  # shear by -5 to +5 degrees
                    rotate=(-179, 179),  # rotate by -179 to +179 degrees
                    order=0,  # use nearest neighbour
                    backend="cv2",  # opencv for fast processing
                    seed=rng
                ),
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.size[0], self.size[1], position="center"
                ),
                iaa.Fliplr(0.5, seed=rng),
                iaa.Flipud(0.5, seed=rng),
            ]

            input_augs = [
                iaa.OneOf(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: gaussian_blur(*args, max_ksize=3),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: median_blur(*args, max_ksize=3),
                        ),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                        iaa.ElasticTransformation(alpha=(0.0, 2.0), sigma=0.25, seed=rng),
                        iaa.GaussianBlur(sigma=(0.0, 1.0),seed=rng),
                        # iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30),seed=rng),
                        # iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
                        # iaa.MultiplySaturation((0.5, 1.5)),


                    ]
                ),
                iaa.Sequential(
                    [

                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_hue(*args, range=(-8, 8)),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_saturation(
                                *args, range=(-0.2, 0.2)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_brightness(
                                *args, range=(-26, 26)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_contrast(
                                *args, range=(0.75, 1.25)
                            ),
                        ),
                    ],
                    random_order=True,
                ),
            ]
        elif mode == "valid":
            shape_augs = [
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                )
            ]
            input_augs = []

        return shape_augs, input_augs





if __name__ == "__main__":
    tissue_images_path ='E:/PumaDataset/01_training_dataset_tif_ROIs/'
    tissue_labels_path = 'E:/PumaDataset/01_training_dataset_geojson_tissue/'

    all_tissue_data = [tissue_images_path + image for image in os.listdir(tissue_images_path)]
    all_tissue_labels = [tissue_labels_path + labels for labels in os.listdir(tissue_labels_path)]

    tissue_labels = ['tissue_white_background','tissue_stroma','tissue_blood_vessel','tissue_tumor','tissue_epidermis','tissue_necrosis']
    target_size = (512,512)
    X_data, y_data = load_data_tissue(target_size = target_size,
                                      data_path = all_tissue_data[0:3],
                                      annot_path = all_tissue_labels[0:3],
                                      tissue_labels = tissue_labels,
                                      im_size = [1024,1024])
    n_class  = 6
    train_dataset = PumaTissueDataset(X_data,
                                      y_data,
                                      n_class1=n_class,
                                      size1=target_size,
                                    device1=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                      transform = True)
    image,mask = train_dataset.__getitem__(0)
    a = 2
    plt.imshow(X_data[0,:,:]/255)
    plt.show()
    plt.imshow(y_data[0,:,:,2])
    plt.show()
    plt.imshow(y_data[0,:,:,3])
    plt.show()
    plt.imshow(image[0,:,:].cpu())
    plt.show()
    a=1