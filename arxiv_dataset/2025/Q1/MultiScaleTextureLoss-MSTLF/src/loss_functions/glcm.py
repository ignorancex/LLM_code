import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
import time
from itertools import product


def soft_binning(x, c, sigma):
    exponent = -((x - c) ** 2) / (2 * sigma ** 2)
    result = torch.exp(exponent)
    return result


def shift_image(x, d):
    result = torch.roll(x, shifts=d, dims=3)
    result[:, :, :, 0] = 0
    return result


def soft_binned_glcm(x, num_levels, min_r=-1, max_r=1, d=1, theta=0):
    centers = torch.linspace(min_r, max_r, num_levels)

    start = time.time()
    I_bins = [soft_binning(x, center, 0.005) for center in centers]
    stop = time.time()
    # print(f"Computation time 1-2: {stop - start}")

    start = time.time()
    I_s_bins = [shift_image(I_bin, d) for I_bin in I_bins]
    stop = time.time()
    # print(f"Computation time 2-3: {stop - start}")

    start = time.time()
    combinations = list(product(I_bins, I_s_bins))
    stop = time.time()
    # print(f"Computation time 3-4: {stop - start}")

    # Calculate occurrences with element-wise multiplication and summation
    start = time.time()
    occurrences = [torch.sum(I_bin * I_s_bin, dim=(2, 3)) for (I_bin, I_s_bin) in combinations]
    stop = time.time()
    # print(f"Computation time 5-6: {stop - start}")

    start = time.time()
    glcm = torch.cat(occurrences, dim=1).view(x.size(0), 1, num_levels, num_levels)
    stop = time.time()
    # print(f"Computation time 7-8: {stop - start}")

    start = time.time()
    glcm = glcm.permute(0, 1, 3, 2) + glcm
    stop = time.time()
    # print(f"Computation time 9-10: {stop - start}")

    # Normalize each GLCM
    glcm_sum = torch.sum(glcm, dim=(2, 3), keepdim=True)
    # Replace zeros in glcm_sum with a small value to avoid division by zero
    glcm_sum = torch.where(glcm_sum == 0, torch.ones_like(glcm_sum), glcm_sum)

    # Perform element-wise division with automatic differentiation
    glcm /= glcm_sum

    plt.imshow(glcm[0, 0, :, :].detach())
    plt.title("GLCM differentiable")
    plt.show()
    return glcm


def compute_haralick_features(glcm):
    num_gray_levels = glcm.shape[2]

    # Create two 1D tensors representing the indices along each dimension
    I = torch.arange(0, num_gray_levels).unsqueeze(1)  # Column vector
    J = torch.arange(0, num_gray_levels).unsqueeze(0)  # Row vector

    weights = (I - J) ** 2
    weights = weights.reshape((1, 1, num_gray_levels, num_gray_levels))
    contrast = torch.sum(glcm * weights, dim=(2, 3))
    return contrast


def texture_grid(img):
    list_haralick = []
    for i in [1, 3, 5, 7]:
        list_haralick.append(compute_haralick_features(soft_binned_glcm(img, 256, -1, 1, i, 0)).unsqueeze(0))
    stacked_list = torch.cat(list_haralick, dim=0).view(2, 1, 2)
    print(stacked_list)


def texture_loss(fake_im, real_im, opt):
    glcm_real = soft_binned_glcm(real_im)
    glcm_fake = soft_binned_glcm(fake_im)

    if opt.texture_criterion == 'max':
        delta_grids = (glcm_fake - glcm_real).view(opt.batch_size, -1)  # change shape from BxCxHxW to Bx(HXW)
        criterion_texture, _ = torch.max(delta_grids, dim=1)  # return a tensor 1xB

        # compute the loss function by averaging over the batch
        loss_cycle_texture = torch.mean(criterion_texture)
        return loss_cycle_texture, delta_grids, criterion_texture

        return loss_cycle_texture


def frobenius_dist(t1):
    dot_prod = t1 * t1
    return torch.sqrt(torch.sum(dot_prod, dim=1))


if __name__ == "__main__":
    import torch
    from itertools import product
    from skimage import data
    from skimage.feature import graycomatrix, graycoprops
    import time

    image = torch.tensor(data.camera()).unsqueeze(0).unsqueeze(1)
    print(image.shape)
    start = time.time()
    glcm_scikit = graycomatrix(image[0, 0, :, :], distances=[3], angles=[0], levels=256, symmetric=True, normed=True)
    print(graycoprops(glcm_scikit, 'contrast'))
    stop = time.time()
    # print(glcm_scikit)
    print(f"Computation time scikit: {stop - start}")
    plt.imshow(glcm_scikit[:, :, 0, 0])
    plt.title("GLCM scikit")
    plt.show()
    start = time.time()
    glcm_soft_binned = soft_binned_glcm(image, 256, 0, 255, 3, 0)
    print(glcm_soft_binned)
    print(compute_haralick_features(glcm_soft_binned).shape)
    # print(glcm_soft_binned)
    stop = time.time()
    print(f"Computation time soft binning: {stop - start}")
    plt.imshow(glcm_soft_binned[0, 0, :, :])
    plt.title("GLCM differentiable")
    plt.show()

    image2 = torch.tensor([[[[1, 1, 2],
                             [1, 1, 2],
                             [1, 2, 1]]],
                           [[[2, 3, 2],
                             [1, 1, 2],
                             [1, 2, 1]]]], dtype=torch.uint8, requires_grad=False)

    image_ = torch.tensor([[[[0, 0, 1],
                             [0, 0, 1],
                             [0, 1, 0]]],
                           [[[1, 2, 1],
                             [0, 0, 1],
                             [0, 1, 0]]]], dtype=torch.uint8, requires_grad=False)

    print(image2.shape)
    glcm_s = graycomatrix(image_[0, 0, :, :].detach().numpy(), distances=[1], angles=[0], levels=3, symmetric=True, normed=True)
    print(glcm_s)
    contrast_s = graycoprops(glcm_s, 'contrast')
    print(contrast_s)
    glcm_soft = soft_binned_glcm(image_, 3, 0, 2, 1, 0)
    print(glcm_soft)
    contrast_soft = compute_haralick_features(glcm_soft)
    print(contrast_soft)
    # print(image2.squeeze(1).shape)
    # print(torch.roll(image2, shifts=1, dims=3))

    # I_bin_1 = soft_binning(image2, 1, 0.1)
    # I_bin_2 = soft_binning(image2, 2, 0.1)
    # I_bin_3 = soft_binning(image2, 3, 0.1)

    # I_s_bin_1 = shift_image(I_bin_1)
    # I_s_bin_2 = shift_image(I_bin_2)
    # I_s_bin_3 = shift_image(I_bin_3)
    # print((I_bin_1 * I_s_bin_1).shape)
    # occurrences = [(I_bin_1 * I_s_bin_1).sum(dim=(2, 3)), (I_bin_2 * I_s_bin_1).sum(dim=(2, 3)), (I_bin_3 * I_s_bin_1).sum(dim=(2, 3)),
    #                (I_bin_1 * I_s_bin_2).sum(dim=(2, 3)), (I_bin_2 * I_s_bin_2).sum(dim=(2, 3)), (I_bin_3 * I_s_bin_2).sum(dim=(2, 3)),
    #                (I_bin_1 * I_s_bin_3).sum(dim=(2, 3)), (I_bin_2 * I_s_bin_3).sum(dim=(2, 3)), (I_bin_3 * I_s_bin_3).sum(dim=(2, 3))]

    # print((I_bin_1 * I_s_bin_1).sum(dim=(2, 3)).shape)
    # print(occurrences)

    # glcm = torch.cat(occurrences, dim=1).view(2, 1, 3, 3)
    # glcm_sym = glcm.permute(0, 1, 3, 2) + glcm
    # print(glcm_sym)

    # boundaries = torch.linspace(-1.0, 1.0, 16 + 1)
    # print(boundaries)
