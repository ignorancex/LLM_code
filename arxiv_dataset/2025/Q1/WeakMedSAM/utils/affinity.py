import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def get_indices_of_pairs(radius, size):
    search_dist = []

    for x in range(1, radius):
        search_dist.append((0, x))

    for y in range(1, radius):
        for x in range(-radius + 1, radius):
            if x * x + y * y < radius * radius:
                search_dist.append((y, x))

    radius_floor = radius - 1

    full_indices = np.reshape(
        np.arange(0, size[0] * size[1], dtype=np.int64), (size[0], size[1])
    )

    cropped_height = size[0] - radius_floor
    cropped_width = size[1] - 2 * radius_floor

    indices_from = np.reshape(
        full_indices[:-radius_floor, radius_floor:-radius_floor], [-1]
    )

    indices_to_list = []

    for dy, dx in search_dist:
        indices_to = full_indices[
            dy : dy + cropped_height,
            radius_floor + dx : radius_floor + dx + cropped_width,
        ]
        indices_to = np.reshape(indices_to, [-1])

        indices_to_list.append(indices_to)

    concat_indices_to = np.concatenate(indices_to_list, axis=0)

    return indices_from, concat_indices_to


def get_aff(x):
    ind_from, ind_to = get_indices_of_pairs(5, (x.size(2), x.size(3)))
    ind_from = torch.from_numpy(ind_from)
    ind_to = torch.from_numpy(ind_to)

    x = x.view(x.size(0), x.size(1), -1)

    ff = torch.index_select(x, dim=2, index=ind_from.cuda(non_blocking=True))
    ft = torch.index_select(x, dim=2, index=ind_to.cuda(non_blocking=True))

    ff = torch.unsqueeze(ff, dim=2)
    ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))

    aff = torch.exp(-torch.mean(torch.abs(ft - ff), dim=1))
    B = aff.shape[0]

    aff = aff.view(B, -1).cpu()

    ind_from_exp = (
        torch.unsqueeze(ind_from, dim=0).expand(ft.size(2), -1).contiguous().view(-1)
    )
    indices = torch.stack([ind_from_exp, ind_to])
    indices_tp = torch.stack([ind_to, ind_from_exp])

    area = x.size(2)
    indices_id = torch.stack(
        [torch.arange(0, area).long(), torch.arange(0, area).long()]
    )

    sparse_idx = torch.cat([indices, indices_id, indices_tp], dim=1)
    N = sparse_idx.size(1)
    batch_idx = torch.zeros((1, B * N))
    for i in range(B):
        start = i * N
        end = (i + 1) * N
        batch_idx[:, start:end] = i
    sparse_idx = sparse_idx.repeat(1, B)
    sparse_idx = torch.cat((batch_idx, sparse_idx), dim=0).long()

    sparse_val = torch.cat([aff, torch.ones([B, area]), aff], dim=1).view(-1)

    aff_mat = torch.sparse_coo_tensor(sparse_idx, sparse_val).to_dense().cuda()

    return aff_mat


def get_tran(x, model, beta=8, grid_ratio=8):
    mask = torch.zeros((x.size(0), 1, x.size(2), x.size(3))).cuda()
    for i in range(grid_ratio):
        for j in range(grid_ratio):
            cur_mask = model.forward_raw_mask(
                x,
                torch.tensor(
                    [[[int((i + 0.5) * grid_ratio), int((j + 0.5) * grid_ratio)]]]
                ).cuda(),
                torch.tensor([[1]]).cuda(),
            )
            mask += cur_mask
            
    mask = F.interpolate(mask, size=(64, 64)).cuda()

    aff_mat = torch.pow(get_aff(mask), beta)
    trans_mat = aff_mat / torch.sum(aff_mat, dim=1, keepdim=True)

    return trans_mat, mask
