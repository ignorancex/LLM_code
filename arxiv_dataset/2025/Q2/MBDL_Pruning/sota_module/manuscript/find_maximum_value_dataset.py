from get_from_config import get_dataset_from_config
from tqdm import tqdm
import numpy as np
import torch


def run(config):

    config['setting']['dataset'] = 'pmri_fastmri'
    config['setting']['mode'] = 'tra'

    x_min, x_max, smps_min, smps_max = np.inf, -np.inf, np.inf, -np.inf

    for idx, dataset in enumerate(get_dataset_from_config(config)):

        iter_ = tqdm(range(len(dataset)))
        for slice_idx in iter_:

            x_hat, smps_hat, y, mask, x, smps = dataset[slice_idx]

            x_min = min(x_min, torch.view_as_real(x).min())
            x_max = max(x_max, torch.view_as_real(x).max())

            smps_min = min(smps_min, torch.view_as_real(smps).min())
            smps_max = max(smps_max, torch.view_as_real(smps).max())

            iter_.set_description("idx: [%d] | x_min[%.2f] x_max[%.2f] smps_min[%.2f] smps_max[%.2f]" % (
                idx, x_min, x_max, smps_min, smps_max))
