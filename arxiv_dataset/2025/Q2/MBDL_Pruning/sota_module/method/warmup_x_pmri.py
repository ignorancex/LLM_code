from .warmup import DenoiserBase
from get_from_config import get_trainer_from_config, get_dataset_from_config, get_save_path_from_config, \
    get_module_from_config
from torch.utils.data import DataLoader
import os


class ImageWarmup(DenoiserBase):
    def __init__(self, net, sigma, lr, is_g_denoise):
        super().__init__(net, sigma, lr, is_g_denoise)

    def get_groundtruth_from_batch(self, batch):
        return batch[-2]


def run(config):
    if config['setting']['mode'] == 'tst':
        raise NotImplementedError()

    trainer = get_trainer_from_config(config)

    tra_dataset, val_dataset, tst_dataset = get_dataset_from_config(config)

    model = ImageWarmup(
        net=get_module_from_config(config, type_='x', use_sigma_map=(config['method']['deq_cal']['warmup']['x_ckpt'] == 'g_denoise')),
        sigma=config['method']['deq_cal']['warmup']['x_sigma'],
        lr=1e-4,
        is_g_denoise=(config['method']['deq_cal']['warmup']['x_ckpt'] == 'g_denoise')
    )

    tra_dataloader = DataLoader(
        tra_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
        drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['train']['num_workers'],
        drop_last=True
    )

    trainer.fit(
        model=model,
        train_dataloaders=tra_dataloader,
        val_dataloaders=val_dataloader,
    )
