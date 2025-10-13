import h5py
import os
import torch


def get_data(
    mode,
    root_path: str = '/opt/dataset/',
):

    assert mode in ['tra', 'tst', 'val']

    with h5py.File(os.path.join(root_path, 'dataset.hdf5'), 'r') as f:

        if mode == 'tst':
            smps = f['tstCsm'][:]
            x = f['tstOrg'][:]

        else:

            num_data = f['trnOrg'].shape[0]

            if mode == 'tra':

                smps = f['trnCsm'][:int(num_data * 0.8)]
                x = f['trnOrg'][:int(num_data * 0.8)]

            else:

                smps = f['trnCsm'][int(num_data * 0.8):]
                x = f['trnOrg'][int(num_data * 0.8):]

    x, smps = torch.from_numpy(x), torch.from_numpy(smps)

    return {
        'x': x,
        'smps': smps
    }
