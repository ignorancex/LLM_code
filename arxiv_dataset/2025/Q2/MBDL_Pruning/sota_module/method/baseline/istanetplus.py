import os.path
import torch.fft
from sota_module.get_from_config import get_trainer_from_config, get_dataset_from_config, get_save_path_from_config
from sota_module.baseline.e2e_varnet.varnet_module import E2EVarNetModule
from torch.utils.data import Dataset, Subset, DataLoader
from sota_module.method.dep_cal import DEQCalibration
from sota_module.utility import convert_pl_outputs, check_and_mkdir, write_test, get_last_folder
import datetime
import pytorch_lightning as pl
from sota_module.baseline.istanetplus import ISTANetplus
from torch import nn
from sota_module.fwd.pmri import ftran as ftran_pmri
from sota_module.fwd.pmri import fmult as fmult_pmri
from ray.air import session
from ray import tune


class ISTANetPlusLightening(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.num_layers = 9  # default value

        self.net = ISTANetplus(
        channels=2 if 'pmri' or "MoDLDataset" or "RealMeasurement" or "Merge" in config['setting']['dataset'] else 1
        )

        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, XPSY):
        """
        :param XPSY: input data dictionary which hold x, P, S, y, ftran, fmult
        :param x: undersampled image, shape: batch, 2, width, height; dtype: float32
        :param y: undersampled measurement, shape: batch, coils, width, height, 2; dtype: float32
        :param P: undersampling mask, shape: batch, width, height; dtype: float32
        :param S: Sensitivity map, shape: batch, coils, width, height; dtype: complex64
        :param ftran: function
        :param fmult: function
        """
        x, P, S, y, ftran, fmult = XPSY.getData()
        #print(f"[istanetplusLightening] x.shape: {x.shape} / x.dtype: {x.dtype}")
        #print(f"[istanetplusLightening] P.shape: {P.shape} / P.dtype: {P.dtype}")
        #print(f"[istanetplusLightening] S.shape: {S.shape} / S.dtype: {S.dtype}")
        #print(f"[istanetplusLightening] y.shape: {y.shape} / y.dtype: {y.dtype}")
        #x_recover = self.net(x, y, ftran, fmult)
        x_recover = self.net(XPSY)
        return x_recover

    def step_helper(self, batch, batch_idx):

        x_input, theta_input, y, mask, x_gt, theta_gt = batch

        if 'pmri' in self.config['setting']['dataset']:
            ftran = lambda y_: ftran_pmri(y=y_, smps=theta_input, mask=mask)
            fmult = lambda x_: fmult_pmri(x=x_, smps=theta_input, mask=mask)

        else:
            raise NotImplementedError()

        [x_output, loss_layers_sym] = self(x_input, y, ftran, fmult)

        # Compute and print loss
        loss_discrepancy = torch.mean(torch.pow(torch.view_as_real(x_output) - torch.view_as_real(x_gt), 2))

        loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
        for k in range(self.num_layers - 1):
            loss_constraint += torch.mean(torch.pow(loss_layers_sym[k + 1], 2))

        gamma = torch.Tensor([0.01]).to(self.device)

        # loss_all = loss_discrepancy
        loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)

        return x_output, x_gt, loss_all, loss_discrepancy, loss_constraint

    def training_step(self, batch, batch_idx):

        x_output, x_gt, loss_all, loss_discrepancy, loss_constraint = self.step_helper(batch, batch_idx)

        self.log(name='tra_loss_discrepancy', value=loss_discrepancy, prog_bar=True)
        self.log(name='tra_loss_constraint', value=loss_constraint, prog_bar=True)

        if batch_idx == 0:
            self.logger.log_image(key='tra_x_hat', images=[DEQCalibration.to_two_dim_magnitude_image(x_output)])

        return loss_all

    def validation_step(self, batch, batch_idx):
        x_output, x_gt, loss_all, loss_discrepancy, loss_constraint = self.step_helper(batch, batch_idx)

        val_psnr, val_ssim = DEQCalibration.psnr_ssim_helper(x_output, x_gt, 1)

        self.log(name='val_psnr', value=val_psnr, prog_bar=True)
        self.log(name='val_ssim', value=val_ssim, prog_bar=True)
        self.log(name='val_loss', value=loss_all, prog_bar=True)

        if batch_idx == 0:
            self.logger.log_image(key='val_x_hat', images=[DEQCalibration.to_two_dim_magnitude_image(x_output)])

    def test_step(self, batch, batch_idx):
        x_output, x_gt, loss_all, loss_discrepancy, loss_constraint = self.step_helper(batch, batch_idx)

        tst_psnr, tst_ssim = DEQCalibration.psnr_ssim_helper(x_output, x_gt, 1)

        self.log(name='tst_psnr', value=tst_psnr, prog_bar=True)
        self.log(name='tst_ssim', value=tst_ssim, prog_bar=True)

        return {
            'tst_psnr_x_hat': tst_psnr.item(),
            'tst_ssim_x_hat': tst_ssim.item(),

            'x_hat': x_output,
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
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-5)

        return optimizer


def run(config):

    config['train']['max_epochs'] = 400

    if config['setting']['mode'] == 'tra':
        config['test']['dec'] = 'BEST_val_loss'

    trainer = get_trainer_from_config(config)

    tra_dataset, val_dataset, tst_dataset = get_dataset_from_config(config)

    if config['setting']['mode'] == 'tst':
        model = ISTANetPlusLightening.load_from_checkpoint(
            checkpoint_path=os.path.join(get_save_path_from_config(config), config['test']['checkpoint_path']),
            config=config
        )
    else:
        model = ISTANetPlusLightening(config)

    if config['setting']['mode'] == 'dug':
        tst_dataset = Subset(tst_dataset, [10, 20, 30, 40, 50])

    ckpt_path = None

    if config['setting']['mode'] == 'tra':

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

        torch.autograd.set_detect_anomaly(True)

        trainer.fit(
            model=model,
            train_dataloaders=tra_dataloader,
            val_dataloaders=val_dataloader
        )

        ckpt_path = 'best'

    tst_dataloader = DataLoader(
        tst_dataset,
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
