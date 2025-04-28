from skimage.feature import graycomatrix, graycoprops
import torch
import cv2
import numpy as np


def texture_extractor(x, opt):
    x = cv2.normalize(x.detach().cpu().numpy(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                      dtype=cv2.CV_32F).astype(np.uint8)

    if opt.texture_offsets == 'all':
        shape = (x.shape[0], 1, 4, 4)  # 4,4
        spatial_offset = [1, 3, 5, 7]
        angular_offset = [0, 45, 90, 135]
    elif opt.texture_offsets == '5':
        shape = (x.shape[0], 1, 1, 4)  # 4,4
        spatial_offset = [5]
        angular_offset = [0, 45, 90, 135]

    texture_matrix = torch.empty(shape)

    for i in range(0, x.shape[0]):
        for idx_d, d in enumerate(spatial_offset):
            for idx_theta, theta in enumerate(angular_offset):
                texture_d_theta = graycoprops(
                    graycomatrix(x[i, 0, :, :], distances=[d], angles=[theta], levels=256, symmetric=True,
                                 normed=True), "contrast")[0][0]
                texture_matrix[i, 0, idx_d - 1, idx_theta - 1] = texture_d_theta

    return texture_matrix


def texture_loss(fake_im, real_im, operator, opt, model=None):
    textures_real = texture_extractor(real_im, opt)
    textures_fake = texture_extractor(fake_im, opt)

    if opt.texture_criterion == 'attention':
        criterion = operator(textures_fake, textures_real).view(opt.batch_size, 1, 4, 4)
        normalized_criterion = (criterion - criterion.min()) / (criterion.max() - criterion.min())
        print(normalized_criterion.shape)
        out_attention, map, weight = model(normalized_criterion)
        print(out_attention.shape)
        loss_cycle_texture = abs(torch.mean(torch.sum(out_attention, dim=(2, 3))))

        return loss_cycle_texture, map, weight

    elif opt.texture_criterion == 'max':
        delta_grids = operator(textures_fake, textures_real).view(opt.batch_size, -1)  # change shape from BxCxHxW to Bx(HXW)
        criterion_texture, _ = torch.max(delta_grids, dim=1)  # return a tensor 1xB

        # compute the loss function by averaging over the batch
        loss_cycle_texture = torch.mean(criterion_texture)
        return loss_cycle_texture, delta_grids, criterion_texture

    elif opt.texture_criterion == 'average':
        delta_grids = operator(textures_fake, textures_real).view(opt.batch_size, -1)  # change shape from BxCxHxW to Bx(HXW)
        criterion_texture = torch.mean(delta_grids, dim=1)  # return a tensor 1xB
        loss_cycle_texture = torch.mean(criterion_texture)

        return loss_cycle_texture

    elif opt.texture_criterion == 'Frobenius':
        delta_grids = operator(textures_fake, textures_real).view(opt.batch_size, -1)  # change shape from BxCxHxW to Bx(HXW)
        criterion_texture = frobenius_dist(delta_grids)  # return a tensor 1xB
        loss_cycle_texture = torch.mean(criterion_texture)

        return loss_cycle_texture


def frobenius_dist(t1):
    dot_prod = t1 * t1
    return torch.sqrt(torch.sum(dot_prod, dim=1))
