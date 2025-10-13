import os.path
import torch.fft
from get_from_config import get_trainer_from_config, get_dataset_from_config, get_save_path_from_config
from torch.utils.data import Dataset, Subset, DataLoader
from method.dep_cal import DEQCalibration
from utility import convert_pl_outputs, check_and_mkdir, write_test, get_last_folder
import datetime
import pytorch_lightning as pl
from baseline.deepinpy.MoDL import MoDL
from fwd.pmri_modl import ftran, fmult
from ray import tune
from ray.tune import session


class DatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x0, smps, y, fwd_para, x, smps_gt = self.dataset[item]

        mask = fwd_para['mask']
        mask = mask.to(torch.bool)

        return x0, smps, y, mask, x


class MoDLParallelMRI(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = MoDL(
            mu_init=0.05,
            cg_max_iter=10,
            iteration=10,
            channels=2
        )

        self.loss = torch.nn.MSELoss()

    def forward(self, x, AHA):
        return self.net(x, AHA)

    def validation_step(self, batch, batch_idx):
        inp, smps, y, mask, target = batch

        AHA = lambda x: ftran(fmult(x, smps, mask), smps, mask)

        output = self(inp, AHA)

        loss = self.loss(
            torch.view_as_real(output), torch.view_as_real(target)
        )

        val_psnr, val_ssim = DEQCalibration.psnr_ssim_helper(output, target, 1)

        self.log(name='val_psnr', value=val_psnr, prog_bar=True)
        self.log(name='val_ssim', value=val_ssim, prog_bar=True)

        self.log(name='val_loss', value=loss, prog_bar=True)

        return output

    def training_step(self, batch, batch_idx):
        inp, smps, y, mask, target = batch

        AHA = lambda x: ftran(fmult(x, smps, mask), smps, mask)

        output = self(inp, AHA)

        loss = self.loss(
            torch.view_as_real(output), torch.view_as_real(target)
        )

        tra_psnr, tra_ssim = DEQCalibration.psnr_ssim_helper(output, target, 1)

        self.log(name='tra_psnr', value=tra_psnr, prog_bar=True)
        self.log(name='tra_ssim', value=tra_ssim, prog_bar=True)

        self.log("tra_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        inp, smps, y, mask, target = batch

        AHA = lambda x: ftran(fmult(x, smps, mask), smps, mask)

        output = self(inp, AHA)

        tst_psnr, tst_ssim = DEQCalibration.psnr_ssim_helper(output, target, 1)

        self.log(name='tst_psnr', value=tst_psnr, prog_bar=True)
        self.log(name='tst_ssim', value=tst_ssim, prog_bar=True)

        return {
            'tst_psnr_x_hat': tst_psnr.item(),
            'tst_ssim_x_hat': tst_ssim.item(),

            'x_hat': output,
            'x_init': target,
        }

    def test_epoch_end(self, outputs) -> None:

        if tune.is_session_enabled():
            ret = {}
            for k in self.trainer.logged_metrics:
                ret.update({
                    k: self.trainer.logged_metrics[k].item()
                })
            session.report(ret)

        log_dict, img_dict = convert_pl_outputs(outputs)

        save_path = get_save_path_from_config(self.config)

        save_path = os.path.join(
            save_path,
            'TEST_' + datetime.datetime.now().strftime("%m%d%H%M") + "_" + get_last_folder(save_path)
        )

        if self.config['test']['dec'] is not None:
            save_path = save_path + "_" + self.config['test']['dec']

        check_and_mkdir(save_path)
        write_test(
            save_path=save_path,
            log_dict=log_dict,
            img_dict=img_dict
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config['train']['lr'])

        return optimizer


def run(config):

    config['train']['max_epochs'] = 400  # default MoDL training epochs

    if config['setting']['mode'] == 'tra':
        config['test']['dec'] = 'BEST_val_loss'

    trainer = get_trainer_from_config(config)

    tra_dataset, val_dataset, tst_dataset = get_dataset_from_config(config)

    if config['setting']['mode'] == 'tst':
        model = MoDLParallelMRI.load_from_checkpoint(
            checkpoint_path=os.path.join(get_save_path_from_config(config), config['test']['checkpoint_path']),
            config=config
        )
    else:
        model = MoDLParallelMRI(config=config)

    if config['setting']['mode'] == 'dug':
        tst_dataset = Subset(tst_dataset, [10, 20, 30, 40, 50])

    ckpt_path = None

    if config['setting']['mode'] == 'tra':

        tra_dataloader = DataLoader(
            DatasetWrapper(tra_dataset),
            batch_size=config['train']['batch_size'],
            shuffle=True,
            num_workers=config['train']['num_workers'],
            drop_last=True
        )

        val_dataloader = DataLoader(
            DatasetWrapper(val_dataset),
            batch_size=config['train']['batch_size'],
            shuffle=False,
            num_workers=config['train']['num_workers'],
            drop_last=True
        )

        trainer.fit(
            model=model,
            train_dataloaders=tra_dataloader,
            val_dataloaders=val_dataloader
        )

        ckpt_path = 'best'

    tst_dataloader = DataLoader(
        DatasetWrapper(tst_dataset),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )

    trainer.test(
        model=model,
        dataloaders=tst_dataloader,
        ckpt_path=ckpt_path
    )
