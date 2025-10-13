import os.path
import torch.fft
from sota_module.get_from_config import get_trainer_from_config, get_dataset_from_config, get_save_path_from_config
from sota_module.baseline.e2e_varnet.varnet_module import E2EVarNetModule, VarNetModule
from torch.utils.data import Dataset, Subset, DataLoader
from sota_module.method.dep_cal import DEQCalibration
from sota_module.utility import convert_pl_outputs, check_and_mkdir, write_test, get_last_folder
import datetime
from ray import tune
from ray.tune import session

'''
class DatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x0, smps, y, mask, x, smps_gt = self.dataset[item]

        masked_kspace = torch.view_as_real(y)

        # mask = fwd_para['mask']
        mask = mask.unsqueeze(0).unsqueeze(-1)
        mask = mask.to(torch.bool)

        target = x

        max_value = torch.Tensor([1.0]).squeeze()

        return masked_kspace, mask, target, max_value
'''
class DatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        """
        NOT YET TO SPECIFY

        :param x0: undersample image, shape: batch, width, height; dtype: complex
        :param y: undersample measurement, shape: batch, width, height; dtype: complex
        :param smps: sensitivity maps, shape: batch, coils, width, height; dtype: complex
        :param mask: sampling mask, shape: batch, width, height; dtype: float/bool
        :return: undersampled measurement
        """
        #_, x0, _, y, x, mask, smps =self.dataset[item]
        _, x0, _, y, x, mask, smps =self.dataset[item]
        #x0 = x0.unsqueeze(0)
        #x = x.unsqueeze(0)
        #y = y.unsqueeze(0)
        #smps = smps.unsqueeze(0)

        #print(f"[e2evarnet.py] x0.shape: {x0.shape} / x0.dtype: {x0.dtype}")
        #print(f"[e2evarnet.py] y.shape: {y.shape} / y.dtype: {y.dtype}")
        #print(f"[e2evarnet.py] x.shape: {x.shape} / x0.dtype: {x.dtype}")
        #print(f"[e2evarnet.py] smps.shape: {smps.shape} / smps.dtype: {smps.dtype}")
        #print(f"[e2evarnet.py] mask.shape: {mask.shape} / mask.dtype: {mask.dtype}")

        #masked_kspace = torch.view_as_real(y)

        # mask = fwd_para['mask']
        mask = mask.unsqueeze(0).unsqueeze(-1)
        mask = mask.to(torch.bool)


        # print(f"mask.shape:{mask.shape}")

        target = x

        max_value = torch.Tensor([1.0]).squeeze()
        # raise ValueError()
        #return masked_kspace, mask, target, max_value

        #print(f"[e2evarnet.py] x0.shape: {x0.shape} / x0.dtype: {x0.dtype}")
        #print(f"[e2evarnet.py] y.shape: {y.shape} / y.dtype: {y.dtype}")
        #print(f"[e2evarnet.py] x.shape: {x.shape} / x0.dtype: {x.dtype}")
        #print(f"[e2evarnet.py] smps.shape: {smps.shape} / smps.dtype: {smps.dtype}")
        #print(f"[e2evarnet.py] mask.shape: {mask.shape} / mask.dtype: {mask.dtype}")

        return _, x0, _, y, target, mask, smps#, max_value


class VarNetWrapper(VarNetModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def validation_step(self, batch, batch_idx):
        masked_kspace, mask, target, max_value = batch

        output = self(masked_kspace, mask, None)
        output = torch.view_as_complex(output)

        loss = self.loss(output.abs().unsqueeze(1), target.abs().unsqueeze(1), data_range=max_value)

        val_psnr, val_ssim = DEQCalibration.psnr_ssim_helper(output, target, 1)

        self.log(name='val_psnr', value=val_psnr, prog_bar=True)
        self.log(name='val_ssim', value=val_ssim, prog_bar=True)

        self.log(name='val_loss', value=loss, prog_bar=True)

        return output

    def training_step(self, batch, batch_idx):
        masked_kspace, mask, target, max_value = batch

        output = self(masked_kspace, mask, None)
        output = torch.view_as_complex(output)

        loss = self.loss(output.abs().unsqueeze(1), target.abs().unsqueeze(1), data_range=max_value)

        tra_psnr, tra_ssim = DEQCalibration.psnr_ssim_helper(output, target, 1)

        self.log(name='tra_psnr', value=tra_psnr, prog_bar=True)
        self.log(name='tra_ssim', value=tra_ssim, prog_bar=True)

        self.log("tra_loss", loss)

        return loss

    def validation_step_end(self, val_logs):
        pass

    def validation_epoch_end(self, val_logs):
        pass

    def test_step(self, batch, batch_idx):
        masked_kspace, mask, target, max_value = batch

        output = self(masked_kspace, mask, None)
        output = torch.view_as_complex(output)

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



class E2EVarNetWrapper(E2EVarNetModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def validation_step(self, batch, batch_idx):
        masked_kspace, mask, target, max_value = batch

        output = self(masked_kspace, mask, None)
        output = torch.view_as_complex(output)

        loss = self.loss(output.abs().unsqueeze(1), target.abs().unsqueeze(1), data_range=max_value)

        val_psnr, val_ssim = DEQCalibration.psnr_ssim_helper(output, target, 1)

        self.log(name='val_psnr', value=val_psnr, prog_bar=True)
        self.log(name='val_ssim', value=val_ssim, prog_bar=True)

        self.log(name='val_loss', value=loss, prog_bar=True)

        return output

    def training_step(self, batch, batch_idx):
        masked_kspace, mask, target, max_value = batch

        output = self(masked_kspace, mask, None)
        output = torch.view_as_complex(output)

        loss = self.loss(output.abs().unsqueeze(1), target.abs().unsqueeze(1), data_range=max_value)

        tra_psnr, tra_ssim = DEQCalibration.psnr_ssim_helper(output, target, 1)

        self.log(name='tra_psnr', value=tra_psnr, prog_bar=True)
        self.log(name='tra_ssim', value=tra_ssim, prog_bar=True)

        self.log("tra_loss", loss)

        return loss

    def validation_step_end(self, val_logs):
        pass

    def validation_epoch_end(self, val_logs):
        pass

    def test_step(self, batch, batch_idx):
        masked_kspace, mask, target, max_value = batch

        output = self(masked_kspace, mask, None)
        output = torch.view_as_complex(output)

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


def run(config):

    config['train']['max_epochs'] = 50  # default E2E-Varnet training epochs

    if config['setting']['mode'] == 'tra':
        config['test']['dec'] = 'BEST_val_loss'

    trainer = get_trainer_from_config(config)

    tra_dataset, val_dataset, tst_dataset = get_dataset_from_config(config)

    if config['setting']['mode'] == 'tst':
        model = E2EVarNetWrapper.load_from_checkpoint(
            checkpoint_path=os.path.join(get_save_path_from_config(config), config['test']['checkpoint_path']),
            config=config
        )
    else:
        model = E2EVarNetWrapper(config)

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
