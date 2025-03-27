# Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import shutil
import sys
import os
import time

import matplotlib.pyplot as plt
import torch as th
import glob
import numpy as np
import math

from skimage.transform import resize

sys.path.append('/Data/MolecularEL/airlab')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import airlab as al

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    gray[gray > 1.] = 1.
    return gray

def rgb2gray_blue(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = b

    return gray


def airlab(hema_root, lbl_root, pas_root, pas_output_root, warp_output_root, displacement_root, overlap_folder, image_badcase_dir):
    start = time.time()

    # set the used data type
    dtype = th.float32
    # set the device for the computaion to CPU
    #device = th.device("cpu")
    device = th.device("cuda")

    # In order to use a GPU uncomment the following line. The number is the device index of the used GPU
    # Here, the GPU with the index 0 is used.
    # device = th.device("cuda:0")

    # load the image data and normalize to [0, 1]
    # fixed_image = al.read_image_as_tensor("./data/affine_test_image_2d_fixed.png", dtype=dtype, device=device)
    # moving_image = al.read_image_as_tensor("./data/affine_test_image_2d_moving.png", dtype=dtype, device=device)

    #moving_image = al.image_from_numpy(plt.imread("/Data2/HumanKidney/Auto_label/PAS_png/PT/final_image_20X_patches_WSI/1024_24064_lbl.png")[:,:,0], (), (), dtype=dtype, device=device)


    # only blue channel

    # moving_image = al.image_from_numpy(rgb2gray_blue(plt.imread(lbl_root)[:,:,:3]), (), (), dtype=dtype, device=device)
    moving_image = al.read_image_as_tensor(lbl_root, dtype=dtype, device=device)
    # moving_image = al.image_from_numpy(plt.imread(lbl_root)[:,:,3], (), (), dtype=dtype, device=device)
    #fixed_image = al.read_image_as_tensor("/Data2/HumanKidney/Auto_label/PAS_png/PT/final_image_20X_patches_WSI_registration/1024_24064_img.png_hema.png", dtype=dtype, device=device)
    fixed_image = al.read_image_as_tensor(hema_root, dtype=dtype, device=device)

    fixed_image.image = fixed_image.image / 255.
    moving_image.image = moving_image.image / 255.

    'intensity changes'
    moving_image.image = 255 - (255 - moving_image.image) ** 2
    # moving_image.image = moving_image.image ** 2
    # print(fixed_image.image.max())
    # print(moving_image.image.max())
    # moving_image.image = moving_image.image * 2 #  / 255.

    # moving_image.image[moving_image.image > 1.] = 1.
    # moving_image.image = 1 - moving_image.image

    fixed_image, moving_image = al.utils.normalize_images(fixed_image, moving_image)

    # convert intensities so that the object intensities are 1 and the background 0. This is important in order to
    # calculate the center of mass of the object

    fixed_image.image = 1 - fixed_image.image

    'can do it reversed'
    #moving_image.image = 1 - 2 * moving_image.image
    #moving_image.image[moving_image.image < 0] = 0

    # create pairwise registration object
    registration = al.PairwiseRegistration()

    # choose the affine transformation model
    # transformation = al.transformation.pairwise.RigidTransformation(moving_image, opt_cm=True)
    transformation = al.transformation.pairwise.AffineTransformation(moving_image, opt_cm=True)
    # initialize the translation with the center of mass of the fixed image
    transformation.init_translation(fixed_image)

    registration.set_transformation(transformation)

    # choose the Mean Squared Error as image loss
    image_loss1 = al.loss.pairwise.MSE(fixed_image, moving_image)
    image_loss2 = al.loss.pairwise.SSIM(fixed_image, moving_image)
    image_loss3 = al.loss.pairwise.NCC(fixed_image, moving_image)

    registration.set_image_loss([image_loss1, image_loss2, image_loss3])
    # registration.set_image_loss([ image_loss2, image_loss3])

    # choose the Adam optimizer to minimize the objective
    optimizer = th.optim.Adam(transformation.parameters(), lr=0.001, amsgrad=True)

    registration.set_optimizer(optimizer,0.01)
    registration.set_number_of_iterations(200)

    # start the registration
    registration.start()

    # set the intensities back to the original for the visualisation
    fixed_image.image = 1 - fixed_image.image

    'can do it reversed'
    # moving_image.image = 1 - 2 * moving_image.image
    # moving_image.image[moving_image.image < 0] = 0
    # moving_image.image = 1 - moving_image.image

    # warp the moving image with the final transformation result
    displacement = transformation.get_displacement()
    affine_matrix = transformation.transformation_matrix.detach().cpu().numpy()

    x_displacement = np.max([displacement[0, 0, 0].abs() ,displacement[-1, 0, 0].abs() ,displacement[0, -1, 0].abs() ,displacement[-1, -1, 0].abs()])
    y_displacement = np.max([displacement[0, 0, 1].abs() ,displacement[-1, 0, 1].abs() ,displacement[0, -1, 1].abs() ,displacement[-1, -1, 1].abs()])
    # warped_image = al.transformation.utils.warp_image(moving_image, displacement)

    #fixed_image_RGB = al.image_from_numpy(plt.imread("/Data2/HumanKidney/Auto_label/PAS_png/PT/final_image_20X_patches_WSI/1024_24064_img.png"), (), (), dtype=dtype, device=device)
    moving_image_R = al.image_from_numpy(plt.imread(lbl_root)[:,:,0], (), (), dtype=dtype, device=device)
    moving_image_G = al.image_from_numpy(plt.imread(lbl_root)[:,:,1], (), (), dtype=dtype, device=device)
    moving_image_B = al.image_from_numpy(plt.imread(lbl_root)[:,:,2], (), (), dtype=dtype, device=device)

    warped_image_R = al.transformation.utils.warp_image(moving_image_R, displacement)
    warped_image_G = al.transformation.utils.warp_image(moving_image_G, displacement)
    warped_image_B = al.transformation.utils.warp_image(moving_image_B, displacement)

    finlbl_image_RGB = np.zeros((moving_image_R.image.shape[2], moving_image_R.image.shape[3], 3))
    finlbl_image_RGB[:,:,0] = warped_image_R.image[0,0].cpu().numpy()
    finlbl_image_RGB[:,:,1] = warped_image_G.image[0,0].cpu().numpy()
    finlbl_image_RGB[:,:,2] = warped_image_B.image[0,0].cpu().numpy()

    'can do it reversed'
    # finlbl_image_RGB[:,:,0] = 1 - 2 * warped_image_R.image[0,0].cpu().numpy()
    # finlbl_image_RGB[:,:,1] = 1 - 2 * warped_image_G.image[0,0].cpu().numpy()
    # finlbl_image_RGB[:,:,2] = 1 - 2* warped_image_B.image[0,0].cpu().numpy()
    #
    # finlbl_image_RGB[finlbl_image_RGB < 0] = 0

    overlap = plt.imread(pas_root)[:,:,:3] / 2 + finlbl_image_RGB
    overlap[overlap > 1.] = 1.
    overlap_root = pas_output_root.replace(os.path.dirname(pas_output_root), overlap_folder)

    if x_displacement > 0.2 or x_displacement > 0.2:
        shutil.copy(pas_root, pas_output_root.replace(os.path.dirname(pas_output_root), image_badcase_dir))
        plt.imsave(warp_output_root.replace(os.path.dirname(warp_output_root), image_badcase_dir), finlbl_image_RGB)
        np.save(displacement_root.replace(os.path.dirname(displacement_root), image_badcase_dir), affine_matrix)
        plt.imsave(overlap_root.replace(os.path.dirname(overlap_root), image_badcase_dir).replace('_img.png','_overlap.png'), overlap)
    else:
        shutil.copy(pas_root, pas_output_root)
        plt.imsave(warp_output_root, finlbl_image_RGB)
        np.save(displacement_root, affine_matrix)
        plt.imsave(overlap_root, overlap)

    end = time.time()

    #
    # print("=================================================================")
    #
    # print("Registration done in:", end - start, "s")
    # print("Result parameters:")
    # transformation.print()
    #
    # # plot the results
    # plt.subplot(131)
    # plt.imshow(fixed_image.numpy(), cmap='gray')
    # plt.title('Fixed Image')
    #
    # plt.subplot(132)
    # plt.imshow(moving_image.numpy(), cmap='gray')
    # plt.title('Moving Image')
    #
    # plt.subplot(133)
    # plt.imshow(warped_image.numpy(), cmap='gray')
    # plt.title('Warped Moving Image')
    #
    # plt.show()


if __name__ == '__main__':
    # case_folder = '/Data2/Auto_label/PAS+Endo_png'
    case_folder =  '/Data/MolecularEL/PAS+Endo_png'

    cases = glob.glob(os.path.join(case_folder, '*'))
    for ci in range(len(cases)):
        case = cases[ci]
        now_case = os.path.basename(case)
        lbl_input_dir = os.path.join(case, 'patches_20X')
        image_input_dir = lbl_input_dir# os.path.join(case, 'final_image_20X_patches_WSI_colordeconv')
        image_output_dir = os.path.join(case, 'patches_20X_registration_2048')
        image_badcase_dir = os.path.join(case, 'patches_20X_registration_badcase_2048')

        overlap_folder = os.path.join(case, 'patches_20X_registration_overlap')
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)
        if not os.path.exists(overlap_folder):
            os.makedirs(overlap_folder)
        if not os.path.exists(image_badcase_dir):
            os.makedirs(image_badcase_dir)

        images = glob.glob(os.path.join(image_input_dir, '*img.png'))
        for ii in range(len(images)):
            pas_root = images[ii]
            #if os.path.basename(pas_root) != '16896_19968_img.png':
            #    continue
            pas_output_root = pas_root.replace(image_input_dir, image_output_dir)
            lbl_root = pas_root.replace(image_input_dir, lbl_input_dir).replace('img','lbl')
            hema_root = pas_root#pas_root.replace('_img.png', '_img.png_hema.png')
            warp_output_root = lbl_root.replace(lbl_input_dir, image_output_dir)
            displacement_root = pas_output_root.replace('.png', '.npy')

            if not (os.path.exists(displacement_root.replace(os.path.dirname(displacement_root), image_badcase_dir)) or os.path.exists(displacement_root)):
                print(hema_root)
                try:
                    airlab(hema_root, lbl_root, pas_root, pas_output_root, warp_output_root, displacement_root, overlap_folder, image_badcase_dir)
                except:
                    print('error')