import torch
import sys
sys.path.append('..')
from dataset import PRISM_MRI_Dataset
sys.path.remove('..')

data_folder = 'C:/Users/../PRISM/pipeline/data folders/IXI-original-slices-stripped/'
save_path = 'C:/Users/../PRISM/data/torch-datasets'
sites = ['Guys', 'HH', 'IOP']

for site in sites:
    for mode in ['train', 'test']:
        dir = data_folder+site
        dataset = PRISM_MRI_Dataset(data_dir=dir, mode=mode, modalities=['T2', 'PD'], stripped=True)
        print(f"Size of dataset {site}-{mode}: {len(dataset)}")
        torch.save(dataset, f'{save_path}/IXI-{site}-{mode}.pt')