from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torch import nn
import torch
import copy
from torch_util.metrics import Mean, compare_psnr, compare_ssim, Stack, compare_snr

from torch_util.callback import CallbackList, BaseLogger, ModelCheckpoint, Tensorboard
from torch.utils.data import Dataset
from torch_util import losses
from torch_util.module import SpatialTransformer
import shutil
import datetime
import time
from Torch_Pruning import torch_pruning as tp

from torch_util.common import dict2pformat, write_test, abs_helper, check_and_mkdir, plot_helper
from torch_util.module import single_ftran, single_fmult, mul_ftran, mul_fmult, crop_images
from sota_module.fwd.pmri import ftran as ftran_pmri
from sota_module.fwd.pmri import fmult as fmult_pmri
from sota_module.baseline.e2e_varnet import fastmri
from sota_module.method.baseline.e2evarnet import DatasetWrapper
from sota_module.baseline.e2e_varnet.fastmri.data import transforms
from dataset.pmri_fastmri_brain import addwgn
from dataset.modl import MoDLDataset, CombinedDataset

class DictDataset(Dataset):
    def __init__(self, mode, data_dict, config):

        self.__data_dict = data_dict
        self.config = config

        assert mode in ['train', 'valid', 'test']

        self.__index_map = []

        n_slice = self.__data_dict['fixed_x'].shape[0]

        for slice_ in range(n_slice):
            self.__index_map.append([slice_])

        total_len = self.__index_map.__len__()
        print("total_len: ", total_len)

        if mode == 'train':
            self.__index_map = self.__index_map[0: 410]

        elif mode == 'valid':
            self.__index_map = self.__index_map[410: 470]

        else:
            self.__index_map = self.__index_map[470:]

    def __len__(self):
        return len(self.__index_map)

    def __getitem__(self, item):
        slice_, = self.__index_map[item]

        fixed_x = self.__data_dict['fixed_x'][slice_]
        moved_x = self.__data_dict['moved_x'][slice_]
        sensitivity_map = self.__data_dict['sensitivity_map'][slice_]

        fixed_y = self.__data_dict['fixed_y'][slice_]
        fixed_mask = self.__data_dict['fixed_mask'][slice_]
        fixed_y_tran = self.__data_dict['fixed_y_tran'][slice_]

        moved_y = self.__data_dict['moved_y'][slice_]
        moved_mask = self.__data_dict['moved_mask'][slice_]
        moved_y_tran = self.__data_dict['moved_y_tran'][slice_]

        mul_fixed_y = self.__data_dict['mul_fixed_y'][slice_]
        mul_fixed_mask = self.__data_dict['mul_fixed_mask'][slice_]
        mul_fixed_y_tran = self.__data_dict['mul_fixed_y_tran'][slice_]

        mul_moved_y = self.__data_dict['mul_moved_y'][slice_]
        mul_moved_y_tran = self.__data_dict['mul_moved_y_tran'][slice_]
        mul_moved_mask = self.__data_dict['mul_moved_mask'][slice_]

        return fixed_x, moved_x, sensitivity_map,\
            fixed_y, fixed_mask, fixed_y_tran,\
            moved_y, moved_mask, moved_y_tran,\
            mul_fixed_y, mul_fixed_mask, mul_fixed_y_tran,\
            mul_moved_y, mul_moved_y_tran, mul_moved_mask

class inputDataDict(Dataset):
    def __init__(self, x_init, P, S, y, module_name=None):
        self.x_init = x_init
        self.P = P
        self.S = S
        self.y = y
        self.module_name = module_name


    def __len__(self):
        return 1

    def getData(self):
        return self.x_init, self.P, self.S, self.y

class inputDataDictVarnet(Dataset):
    def __init__(self, masked_kspace, mask, num_low_frequencies=None):
        self.masked_kspace = masked_kspace
        self.mask = mask
        self.num_low_frequencies = num_low_frequencies

    def __len__(self):
        return 1

    def getData(self):
        return self.masked_kspace, self.mask, self.num_low_frequencies

class inputDataDictSensitivityModel(Dataset):
    def __init__(self, masked_kspace, mask, num_low_frequencies):
        self.masked_kspace = masked_kspace
        self.mask = mask
        self.num_low_frequencies = num_low_frequencies

    def __len__(self):
        return 1

    def getData(self):
        return self.masked_kspace, self.mask, self.num_low_frequencies

class inputDataDictISTANetplus(Dataset):
    def __init__(self, x, y, ftran, fmult):
        self.x = x
        self.y = y
        #self.ftran = ftran
        #self.fmult = fmult

        self.ftran = lambda y_: ftran_pmri(y=y_, smps=S, mask=P)
        self.fmult = lambda x_: fmult_pmri(x=x_, smps=S, mask=P)

    def __len__(self):
        return 1

    def getData(self):
        return self.x, self.y, self.ftran, self.fmult

class NormUnet(Dataset):
    def __init__(self, masked_kspace, mask, num_low_frequencies):
        self.masked_kspace = masked_kspace
        self.mask = mask
        self.num_low_frequencies = num_low_frequencies

    def __len__(self):
        return 1

    def getData(self):
        return self.masked_kspace, self.mask, self.num_low_frequencies

class inputDataDict1(Dataset):
    def __init__(self, x_init, P, S, y, ftran=None, fmult=None):
        self.x_init = x_init
        self.P = P
        self.S = S
        self.y = y

        if ftran != None and fmult != None:
            self.ftran = ftran
            self.fmult = fmult
        else:
            self.ftran = None
            self.fmult = None

    def __len__(self):
        return 1

    def getData(self):
        if self.ftran != None and self.fmult != None:
            return self.x_init, self.P, self.S, self.y, self.ftran, self.fmult
        else:
            return self.x_init, self.P, self.S, self.y

def train(
        load_dataset_fn,
        recon_module,
        regis_module,
        config
):
    train_dataset = DictDataset(
        mode='train',
        data_dict=load_dataset_fn(baseline_method=None),
        config=config)
    print("[train_dataset] total_len: ", train_dataset.__len__())

    valid_dataset = DictDataset(
        mode='valid',
        data_dict=load_dataset_fn(baseline_method=None),
        config=config)
    print("[valid_dataset] total_len: ", valid_dataset.__len__())

    ########################
    # Load Configuration
    ########################
    regis_batch = config['method']['proposed']['regis_batch']
    recon_batch = config['method']['proposed']['recon_batch']

    is_optimize_regis = config['method']['proposed']['is_optimize_regis']
    mul_coil = config['dataset']['multi_coil']
    recon_module_type = config['module']['recon']['recon_module_type']
    recon_module_type = config['module']['recon']['recon_module_type']
    is_trainable_gamma = config['module']['recon']['is_trainable_gamma']

    lambda_ = config['method']['lambda_']
    loss_regis_mse_COEFF = config['method']['loss_regis_mse']
    loss_regis_dice_COEFF = config['method']['loss_regis_dice']

    loss_recon_consensus_COEFF = config['method']['loss_recon_consensus']

    recon_lr, regis_lr = config['train']['recon_lr'], config['train']['regis_lr']
    recon_loss, regis_loss = config['train']['recon_loss'], config['train']['regis_loss']

    batch_size = config['train']['batch_size']

    num_workers = config['train']['num_workers']
    train_epoch = config['train']['train_epoch']
    verbose_batch = config['train']['verbose_batch']
    tensorboard_batch = config['train']['tensorboard_batch']
    checkpoint_epoch = config['train']['checkpoint_epoch']

    check_and_mkdir(config['setting']['root_path'])
    file_path = config['setting']['root_path'] + config['setting']['save_folder'] + '/'
    ########################
    # Dataset
    ########################

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    train_iter_total = int(train_dataset.__len__() / batch_size)

    valid_dataloader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False)
    valid_iter_total = int(valid_dataset.__len__() / 1)

    sample_indices = [10, 20, 30]
    valid_sample_fixed_x, valid_sample_moved_x, valid_sensitivity_map, valid_sample_single_fixed_y, valid_sample_single_fixed_mask, valid_sample_single_fixed_y_tran, valid_sample_single_moved_y, valid_sample_single_moved_mask, valid_sample_single_moved_y_tran, \
    valid_sample_mul_fixed_y, valid_sample_mul_fixed_mask, valid_sample_mul_fixed_y_tran, valid_sample_mul_moved_y, valid_sample_mul_moved_y_tran, valid_sample_mul_moved_mask = \
            (i.cuda() for i in next(iter(
            DataLoader(Subset(valid_dataset, sample_indices), batch_size=len(sample_indices)))))

    if mul_coil == False:
        valid_sample_fixed_y = valid_sample_single_fixed_y
        valid_sample_fixed_mask = valid_sample_single_fixed_mask
        valid_sample_fixed_y_tran = valid_sample_single_fixed_y_tran
        valid_sample_moved_y = valid_sample_single_moved_y
        valid_sample_moved_mask = valid_sample_single_moved_mask
        valid_sample_moved_y_tran = valid_sample_single_moved_y_tran

    else:  # mul_coil == True
        valid_sample_fixed_y = valid_sample_mul_fixed_y
        valid_sample_fixed_mask = valid_sample_mul_fixed_mask
        valid_sample_fixed_y_tran = valid_sample_mul_fixed_y_tran
        valid_sample_moved_y = valid_sample_mul_moved_y
        valid_sample_moved_mask = valid_sample_mul_moved_mask
        valid_sample_moved_y_tran = valid_sample_mul_moved_y_tran

    print(valid_sample_moved_x.shape, valid_sample_moved_y_tran.shape, valid_sample_fixed_x.shape,
          valid_sample_fixed_y_tran.shape)

    image_init = {
        'groundtruth': abs_helper(valid_sample_fixed_x),
        'zero-filled': abs_helper(valid_sample_fixed_y_tran),
    }

    ########################
    # Metrics
    ########################
    metrics = Mean()

    ########################
    # Extra-Definition
    ########################
    regis_module.cuda()
    recon_module.cuda()

    sim_loss_fn = losses.ncc_loss
    grad_loss_fn = losses.gradient_loss
    mse_loss_fn = losses.mse_loss

    loss_fn_dict = {
        'l1': nn.L1Loss,
        'l2': nn.MSELoss,
        'smooth_l1': nn.SmoothL1Loss
    }

    recon_loss_fn = loss_fn_dict[recon_loss]()

    trf_232 = SpatialTransformer([256, 232])
    trf_240 = SpatialTransformer([256, 240])
    trf_232.cuda()
    trf_240.cuda()

    ########################
    # Begin Training
    ########################
    regis_optimizer = Adam(regis_module.parameters(), lr=regis_lr)
    recon_optimizer = Adam(recon_module.parameters(), lr=recon_lr)

    check_and_mkdir(file_path)
    regis_callbacks = CallbackList(callbacks=[
        BaseLogger(file_path=file_path),
        Tensorboard(file_path=file_path, per_batch=tensorboard_batch),
        ModelCheckpoint(file_path=file_path + 'regis_model/',
                        period=checkpoint_epoch,
                        monitors=['valid_psnr', 'valid_ssim'],
                        modes=['max', 'max'])
    ])

    regis_callbacks.set_module(regis_module)
    regis_callbacks.set_params({
        'config': config,
        "lr": regis_lr,
        'train_epoch': train_epoch
    })

    recon_callbacks = CallbackList(callbacks=[
        ModelCheckpoint(file_path=file_path + 'recon_model/',
                        period=checkpoint_epoch,
                        monitors=['valid_psnr', 'valid_ssim'],
                        modes=['max', 'max']),
        BaseLogger(file_path=None)
    ])

    recon_callbacks.set_module(recon_module)
    recon_callbacks.set_params({
        'config': config,
        "lr": recon_lr,
        'train_epoch': train_epoch
    })

    regis_callbacks.call_train_begin_hook(image_init)
    recon_callbacks.call_train_begin_hook(image_init)

    global_batch = 1
    for global_epoch in range(1, train_epoch):

        regis_module.train()
        recon_module.train()

        iter_ = tqdm(train_dataloader, desc='Train [%.3d/%.3d]' % (global_epoch, train_epoch), total=train_iter_total)
        for i, train_data in enumerate(iter_):
            fixed_x, moved_x, sensitivity_map, single_fixed_y, single_fixed_mask, single_fixed_y_tran, single_moved_y, single_moved_mask, single_moved_y_tran, \
            mul_fixed_y, mul_fixed_mask, mul_fixed_y_tran, mul_moved_y, mul_moved_y_tran, mul_moved_mask = \
                (i.cuda() for i in train_data)

            if mul_coil == False:
                fixed_y = single_fixed_y
                fixed_mask = single_fixed_mask
                fixed_y_tran = single_fixed_y_tran
                moved_y = single_moved_y
                moved_mask = single_moved_mask
                moved_y_tran = single_moved_y_tran

            else: # mul_coil == True
                fixed_y = mul_fixed_y
                fixed_mask = mul_fixed_mask
                fixed_y_tran = mul_fixed_y_tran
                moved_y = mul_moved_y
                moved_mask = mul_moved_mask
                moved_y_tran = mul_moved_y_tran

            log_batch = {}

            regis_module.train()
            recon_module.train()

            if is_optimize_regis:

                for j in range(regis_batch):
                    if recon_module_type == "deq":
                        fixed_y_tran_recon, fixed_forward_iter, fixed_forward_res = recon_module(fixed_y_tran, fixed_mask, sensitivity_map, fixed_y)
                        moved_y_tran_recon, moved_forward_iter, moved_forward_res = recon_module(moved_y_tran, moved_mask, sensitivity_map, moved_y)
                    else: # pnp, red, original decolearn
                        fixed_y_tran_recon = recon_module(fixed_y_tran, fixed_mask, sensitivity_map, fixed_y)
                        moved_y_tran_recon = recon_module(moved_y_tran, moved_mask, sensitivity_map, moved_y)

                    import fastmri
                    fixed_y_tran_recon = fixed_y_tran_recon.permute([0, 2, 3, 1]).contiguous()
                    fixed_y_tran_recon = fastmri.complex_abs(fixed_y_tran_recon)
                    fixed_y_tran_recon = fixed_y_tran_recon.unsqueeze(1)

                    moved_y_tran_recon = moved_y_tran_recon.permute([0, 2, 3, 1]).contiguous()
                    moved_y_tran_recon = fastmri.complex_abs(moved_y_tran_recon)
                    moved_y_tran_recon = moved_y_tran_recon.unsqueeze(1)

                    fixed_y_tran_recon = torch.nn.functional.pad(fixed_y_tran_recon, [4, 4])
                    moved_y_tran_recon = torch.nn.functional.pad(moved_y_tran_recon, [4, 4])

                    # Check Above code

                    wrap_m2f, flow_m2f = regis_module(moved_y_tran_recon, fixed_y_tran_recon)

                    regis_recon_loss_m2f = sim_loss_fn(wrap_m2f, fixed_y_tran_recon)
                    regis_grad_loss_m2f = grad_loss_fn(flow_m2f)
                    regis_mse_loss_m2f = mse_loss_fn(wrap_m2f, fixed_y_tran_recon)

                    wrap_f2m, flow_f2m = regis_module(fixed_y_tran_recon, moved_y_tran_recon)

                    regis_recon_loss_f2m = sim_loss_fn(wrap_f2m, moved_y_tran_recon)
                    regis_grad_loss_f2m = grad_loss_fn(flow_f2m)
                    regis_mse_loss_f2m = mse_loss_fn(wrap_f2m, moved_y_tran_recon)

                    regis_loss = regis_recon_loss_m2f + regis_recon_loss_f2m
                    if lambda_ > 0:
                        regis_loss += lambda_ * (regis_grad_loss_m2f + regis_grad_loss_f2m)

                    if loss_regis_mse_COEFF > 0:
                        regis_loss += loss_regis_mse_COEFF * (regis_mse_loss_m2f + regis_mse_loss_f2m)

                    regis_optimizer.zero_grad()
                    regis_loss.backward()

                    #torch.nn.utils.clip_grad_value_(regis_module.parameters(), clip_value=1)
                    torch.nn.utils.clip_grad_value_(regis_module.parameters(), clip_value=0.5)
                    #torch.nn.utils.clip_grad_norm_(regis_module.parameters(), max_norm = 0.3)


                    regis_optimizer.step()

                    recon_module.zero_grad()
                    regis_module.zero_grad()

                    if j == (regis_batch - 1):
                        log_batch.update({
                            'registration_loss': regis_loss.item(),
                        })

            regis_module.train()
            recon_module.train()

            for j in range(recon_batch):
                if recon_module_type == "deq":
                    fixed_y_tran_recon, fixed_forward_iter, fixed_forward_res = recon_module(fixed_y_tran, fixed_mask, sensitivity_map, fixed_y)
                    moved_y_tran_recon, moved_forward_iter, moved_forward_res = recon_module(moved_y_tran, moved_mask, sensitivity_map, moved_y)
                else:
                    fixed_y_tran_recon = recon_module(fixed_y_tran, fixed_mask, sensitivity_map, fixed_y)
                    moved_y_tran_recon = recon_module(moved_y_tran, moved_mask, sensitivity_map, moved_y)

                if is_optimize_regis:
                    fixed_y_tran_recon_abs = torch.nn.functional.pad(
                        torch.sqrt(torch.sum(fixed_y_tran_recon ** 2, dim=1, keepdim=True)), [4, 4])
                    moved_y_tran_recon_abs = torch.nn.functional.pad(
                        torch.sqrt(torch.sum(moved_y_tran_recon ** 2, dim=1, keepdim=True)), [4, 4])

                    _, flow_m2f = regis_module(moved_y_tran_recon_abs, fixed_y_tran_recon_abs)
                    flow_m2f = flow_m2f[..., 4:-4]
                    wrap_m2f = torch.cat([trf_232(tmp, flow_m2f) for tmp in [
                        torch.unsqueeze(moved_y_tran_recon[:, 0], 1), torch.unsqueeze(moved_y_tran_recon[:, 1], 1)
                    ]], 1)
                    from torch_util.module import ftran, fmult
                    wrap_y_m2f = fmult(wrap_m2f.permute([0, 2, 3, 1]).contiguous(), sensitivity_map, fixed_mask, mul_coil)

                    _, flow_f2m = regis_module(fixed_y_tran_recon_abs, moved_y_tran_recon_abs)
                    flow_f2m = flow_f2m[..., 4:-4]
                    wrap_f2m = torch.cat([trf_232(tmp, flow_f2m) for tmp in [
                        torch.unsqueeze(fixed_y_tran_recon[:, 0], 1), torch.unsqueeze(fixed_y_tran_recon[:, 1], 1)
                    ]], 1)

                    wrap_y_f2m = fmult(wrap_f2m.permute([0, 2, 3, 1]).contiguous(), sensitivity_map, moved_mask, mul_coil)

                else:

                    from torch_util.module import ftran, fmult
                    wrap_y_m2f = fmult(wrap_m2f.permute([0, 2, 3, 1]).contiguous(), sensitivity_map, fixed_mask, mul_coil)
                    wrap_y_f2m = fmult(wrap_f2m.permute([0, 2, 3, 1]).contiguous(), sensitivity_map, moved_mask, mul_coil)


                wrap_y_m2f = wrap_y_m2f*fixed_mask
                wrap_y_f2m = wrap_y_f2m*moved_mask
                recon_loss_m2f = recon_loss_fn(wrap_y_m2f, fixed_y)
                recon_loss_f2m = recon_loss_fn(wrap_y_f2m, moved_y)

                recon_loss = recon_loss_f2m + recon_loss_m2f

                recon_loss_consensus_fixed = recon_loss_fn(
                    fmult(fixed_y_tran_recon.permute([0, 2, 3, 1]).contiguous(), sensitivity_map, fixed_mask, mul_coil),
                    fixed_y)
                recon_loss_consensus_moved = recon_loss_fn(
                    fmult(moved_y_tran_recon.permute([0, 2, 3, 1]).contiguous(), sensitivity_map, moved_mask, mul_coil),
                    moved_y)

                if loss_recon_consensus_COEFF > 0:
                    recon_loss += loss_recon_consensus_COEFF * (recon_loss_consensus_fixed + recon_loss_consensus_moved)

                recon_optimizer.zero_grad()
                recon_loss.backward()

                #torch.nn.utils.clip_grad_value_(recon_module.parameters(), clip_value=1)
                torch.nn.utils.clip_grad_value_(recon_module.parameters(), clip_value=0.5)
                #torch.nn.utils.clip_grad_norm_(recon_module.parameters(), max_norm=0.5)

                recon_optimizer.step()

                recon_module.zero_grad()
                regis_module.zero_grad()

                if j == (recon_batch - 1):
                    # to remove the gray background
                    #fixed_y_tran_recon[fixed_x == 0] = 0
                    if recon_module_type == "deq":
                        log_batch.update({
                            'reconstruction_loss': recon_loss.item(),
                            'train_ssim': compare_ssim(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),
                            'train_psnr': compare_psnr(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),
                            'forward_iter': fixed_forward_iter,
                            'forward_res': fixed_forward_res
                        })
                    else:
                        log_batch.update({
                            'reconstruction_loss': recon_loss.item(),
                            'train_ssim': compare_ssim(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),
                            'train_psnr': compare_psnr(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),
                        })


            metrics.update_state(log_batch)

            if (verbose_batch > 0) and (global_batch % verbose_batch == 0):
                iter_.write(("Batch [%.7d]:" % global_batch) + dict2pformat(log_batch))
                iter_.update()

            regis_callbacks.call_batch_end_hook(log_batch, global_batch)
            recon_callbacks.call_batch_end_hook(log_batch, global_batch)
            global_batch += 1

        regis_module.eval()
        recon_module.eval()

        with torch.no_grad():

            iter_ = tqdm(valid_dataloader, desc='Valid [%.3d/%.3d]' % (global_epoch, train_epoch),
                         total=valid_iter_total)
            for i, valid_data in enumerate(iter_):
                fixed_x, moved_x, sensitivity_map, single_fixed_y, single_fixed_mask, single_fixed_y_tran, single_moved_y, single_moved_mask, single_moved_y_tran, \
                mul_fixed_y, mul_fixed_mask, mul_fixed_y_tran, mul_moved_y, mul_moved_y_tran, mul_moved_mask = \
                    (i.cuda() for i in valid_data)

                if mul_coil == False:
                    fixed_y = single_fixed_y
                    fixed_mask = single_fixed_mask
                    fixed_y_tran = single_fixed_y_tran
                    moved_y = single_moved_y
                    moved_mask = single_moved_mask
                    moved_y_tran = single_moved_y_tran

                else:  # mul_coil == True
                    fixed_y = mul_fixed_y
                    fixed_mask = mul_fixed_mask
                    fixed_y_tran = mul_fixed_y_tran
                    moved_y = mul_moved_y
                    moved_mask = mul_moved_mask
                    moved_y_tran = mul_moved_y_tran

                if recon_module_type == "deq":
                    fixed_y_tran_recon, fixed_forward_iter, fixed_forward_res = recon_module(fixed_y_tran, fixed_mask, sensitivity_map, fixed_y)
                    moved_y_tran_recon, moved_forward_iter, moved_forward_res = recon_module(moved_y_tran, moved_mask, sensitivity_map, moved_y)
                else:
                    fixed_y_tran_recon = recon_module(fixed_y_tran, fixed_mask, sensitivity_map, fixed_y)
                    moved_y_tran_recon = recon_module(moved_y_tran, moved_mask, sensitivity_map, moved_y)

                # added for registration visualization

                instant_fixed_x = torch.nn.functional.pad(
                    torch.sqrt(torch.sum(fixed_x ** 2, dim=1, keepdim=True)), [4, 4])
                instant_moved_x = torch.nn.functional.pad(
                    torch.sqrt(torch.sum(moved_x ** 2, dim=1, keepdim=True)), [4, 4])
                instant_fixed_y_tran_recon = torch.nn.functional.pad(
                    torch.sqrt(torch.sum(fixed_y_tran_recon ** 2, dim=1, keepdim=True)), [4, 4])
                instant_moved_y_tran_recon = torch.nn.functional.pad(
                    torch.sqrt(torch.sum(moved_y_tran_recon ** 2, dim=1, keepdim=True)), [4, 4])

                wrap_m2f, flow_m2f = regis_module(instant_moved_y_tran_recon, instant_fixed_y_tran_recon)
                wrap_f2m, flow_f2m = regis_module(instant_fixed_y_tran_recon, instant_moved_y_tran_recon)

                # to remove the gray background
                #fixed_y_tran_recon[fixed_x == 0] = 0
                #moved_y_tran_recon[moved_x == 0] = 0


                if recon_module_type == "pnp":
                    gammaList = recon_module.getGamma()
                    alphaList = recon_module.getAlpha()
                    log_batch = {
                        'gamma': gammaList[0],
                        'alpha': alphaList[0],
                        'valid_ssim': compare_ssim(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),
                        'valid_psnr': compare_psnr(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),

                        'valid_moved_ssim': compare_ssim(abs_helper(moved_y_tran_recon), abs_helper(moved_x)).item(),
                        'valid_moved_psnr': compare_psnr(abs_helper(moved_y_tran_recon), abs_helper(moved_x)).item(),

                        'valid_wrap_ssim_m2f': compare_ssim(abs_helper(wrap_m2f), abs_helper(instant_fixed_x)).item(),
                        'valid_wrap_psnr_m2f': compare_psnr(abs_helper(wrap_m2f), abs_helper(instant_fixed_x)).item(),

                        'valid_wrap_ssim_f2m': compare_ssim(abs_helper(wrap_f2m), abs_helper(instant_moved_x)).item(),
                        'valid_wrap_psnr_f2m': compare_psnr(abs_helper(wrap_f2m), abs_helper(instant_moved_x)).item(),
                        }

                elif recon_module_type== "red":
                    gammaList = recon_module.getGamma()
                    muList = recon_module.getMu()
                    log_batch = {
                        'gamma': gammaList[0],
                        'mu': muList[0],
                        'valid_ssim': compare_ssim(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),
                        'valid_psnr': compare_psnr(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),

                        'valid_moved_ssim': compare_ssim(abs_helper(moved_y_tran_recon), abs_helper(moved_x)).item(),
                        'valid_moved_psnr': compare_psnr(abs_helper(moved_y_tran_recon), abs_helper(moved_x)).item(),

                        'valid_wrap_ssim_m2f': compare_ssim(abs_helper(wrap_m2f), abs_helper(instant_fixed_x)).item(),
                        'valid_wrap_psnr_m2f': compare_psnr(abs_helper(wrap_m2f), abs_helper(instant_fixed_x)).item(),

                        'valid_wrap_ssim_f2m': compare_ssim(abs_helper(wrap_f2m), abs_helper(instant_moved_x)).item(),
                        'valid_wrap_psnr_f2m': compare_psnr(abs_helper(wrap_f2m), abs_helper(instant_moved_x)).item(),
                    }
                elif recon_module_type=="deq":
                    gammaList = recon_module.getGamma()
                    muList = recon_module.getMu()
                    log_batch = {
                        'gamma': gammaList[0],
                        'mu': muList[0],
                        'valid_ssim': compare_ssim(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),
                        'valid_psnr': compare_psnr(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),

                        'valid_moved_ssim': compare_ssim(abs_helper(moved_y_tran_recon), abs_helper(moved_x)).item(),
                        'valid_moved_psnr': compare_psnr(abs_helper(moved_y_tran_recon), abs_helper(moved_x)).item(),

                        'valid_wrap_ssim_m2f': compare_ssim(abs_helper(wrap_m2f), abs_helper(instant_fixed_x)).item(),
                        'valid_wrap_psnr_m2f': compare_psnr(abs_helper(wrap_m2f), abs_helper(instant_fixed_x)).item(),

                        'valid_wrap_ssim_f2m': compare_ssim(abs_helper(wrap_f2m), abs_helper(instant_moved_x)).item(),
                        'valid_wrap_psnr_f2m': compare_psnr(abs_helper(wrap_f2m), abs_helper(instant_moved_x)).item(),
                        'fixed_forward_iter': fixed_forward_iter,
                        'fixed_forward_res': fixed_forward_res,
                        'moved_forward_iter': moved_forward_iter,
                        'moved_forward_res': moved_forward_res
                    }


                else:
                    log_batch = {
                        'valid_ssim': compare_ssim(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),
                        'valid_psnr': compare_psnr(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),
                    }

                metrics.update_state(log_batch)

            if recon_module_type == "deq":
                valid_sample_fixed_y_tran_recon, _, _ = recon_module(valid_sample_fixed_y_tran, valid_sample_fixed_mask,
                                                               valid_sensitivity_map, valid_sample_fixed_y)
                # added
                valid_sample_moved_y_tran_recon, _, _ = recon_module(valid_sample_moved_y_tran, valid_sample_moved_mask,
                                                               valid_sensitivity_map, valid_sample_moved_y)
            else:
                valid_sample_fixed_y_tran_recon = recon_module(valid_sample_fixed_y_tran, valid_sample_fixed_mask,
                                                               valid_sensitivity_map, valid_sample_fixed_y)
                # added
                valid_sample_moved_y_tran_recon = recon_module(valid_sample_moved_y_tran, valid_sample_moved_mask,
                                                               valid_sensitivity_map, valid_sample_moved_y)

            # added
            valid_sample_moved_y_tran_recon = torch.nn.functional.pad(
                torch.sqrt(torch.sum(valid_sample_moved_y_tran_recon ** 2, dim=1, keepdim=True)), [4, 4])
            valid_sample_fixed_y_tran_recon = torch.nn.functional.pad(
                torch.sqrt(torch.sum(valid_sample_fixed_y_tran_recon ** 2, dim=1, keepdim=True)), [4, 4])


            valid_sample_wrap, valid_sample_flow = regis_module(valid_sample_moved_y_tran_recon,
                                                                valid_sample_fixed_y_tran_recon)

            valid_grids = create_standard_grid(valid_sample_flow)
            valid_grids = valid_grids.cuda()


            valid_grids = trf_240(valid_grids, valid_sample_flow)

            valid_wrap_norm = create_grid_norm(valid_sample_flow)

        log_epoch = metrics.result()
        metrics.reset_state()

        image_epoch = {
            'prediction': abs_helper(valid_sample_fixed_y_tran_recon),

            'valid_sample_moved_y_tran_recon': abs_helper(valid_sample_moved_y_tran_recon),
            'valid_sample_fixed_y_tran_recon': abs_helper(valid_sample_fixed_y_tran_recon),
            "valid_sample_wrap": valid_sample_wrap,

            "valid_grids": valid_grids,
            "valid_wrap_norm": valid_wrap_norm,
        }

        regis_callbacks.call_epoch_end_hook(log_epoch, image_epoch, global_epoch)
        recon_callbacks.call_epoch_end_hook(log_epoch, image_epoch, global_epoch)

def test(
        load_dataset_fn,
        recon_module: nn.Module,
        regis_module: nn.Module,
        config,
):
    test_dataset = DictDataset(
        mode='test',
        data_dict=load_dataset_fn(baseline_method=None),
        config=config)
    print("[test_dataset] total_len: ", test_dataset.__len__())

    file_path = config['setting']['root_path'] + config['setting']['save_folder'] + '/'
    mul_coil = config['dataset']['multi_coil']
    recon_module_type = config['module']['recon']['recon_module_type']

    recon_checkpoint = file_path + config['test']['recon_checkpoint']
    recon_module.load_state_dict(torch.load(recon_checkpoint))

    recon_module.cuda()
    recon_module.eval()

    metrics = Stack()
    images = Stack()

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        iter_ = tqdm(test_dataloader, desc='Test', total=len(test_dataset))
        for i, test_data in enumerate(iter_):
            fixed_x, moved_x, sensitivity_map, single_fixed_y, single_fixed_mask, single_fixed_y_tran, single_moved_y, single_moved_mask, single_moved_y_tran, \
            mul_fixed_y, mul_fixed_mask, mul_fixed_y_tran, mul_moved_y, mul_moved_y_tran, mul_moved_mask = \
                (i.cuda() for i in test_data)

            if mul_coil == False:
                fixed_y = single_fixed_y
                fixed_mask = single_fixed_mask
                fixed_y_tran = single_fixed_y_tran
                moved_y = single_moved_y
                moved_mask = single_moved_mask
                moved_y_tran = single_moved_y_tran

            else: # mul_coil == True
                fixed_y = mul_fixed_y
                fixed_mask = mul_fixed_mask
                fixed_y_tran = mul_fixed_y_tran
                moved_y = mul_moved_y
                moved_mask = mul_moved_mask
                moved_y_tran = mul_moved_y_tran

            if recon_module_type == "deq":
                fixed_y_tran_recon, fixed_forward_iter, fixed_forward_res = recon_module(fixed_y_tran, fixed_mask, sensitivity_map, fixed_y)
            else:
                fixed_y_tran_recon = recon_module(fixed_y_tran, fixed_mask, sensitivity_map, fixed_y)

            fixed_y_tran_recon = abs_helper(fixed_y_tran_recon)

            fixed_x = abs_helper(fixed_x)
            fixed_y_tran = abs_helper(fixed_y_tran)
            # to remove the gray background
            #fixed_y_tran[fixed_x == 0] = 0
            #fixed_y_tran_recon[fixed_x == 0] = 0
            # Multipling mask to remove the gray background
            '''
            fixed_y_tran_recon = torch.view_as_complex(fixed_y_tran_recon.permute([0, 2, 3, 1]).contiguous())
            fixed_mask = torch.view_as_complex(fixed_mask)
            fixed_y_tran_recon = fixed_y_tran_recon * fixed_mask
            fixed_y_tran_recon = torch.view_as_real(fixed_y_tran_recon)
            fixed_y_tran_recon = fixed_y_tran_recon.permute([0, 3, 1, 2]).contiguous()
            fixed_mask = torch.view_as_real(fixed_mask)
            '''

            log_batch = {
                'zero_filled_ssim': compare_ssim(fixed_y_tran, fixed_x).item(),
                'zero_filled_psnr': compare_psnr(fixed_y_tran, fixed_x).item(),

                'prediction_ssim': compare_ssim(fixed_y_tran_recon, fixed_x).item(),
                'prediction_psnr': compare_psnr(fixed_y_tran_recon, fixed_x).item(),

            }

            images_batch = {
                'groundtruth': fixed_x.detach().cpu().numpy(),
                'zero_filled': fixed_y_tran.detach().cpu().numpy(),
                'prediction': fixed_y_tran_recon.detach().cpu().numpy(),

            }

            metrics.update_state(log_batch)
            images.update_state(images_batch)

    '''
    save_path = file_path + 'test_' + \
        '_' + config['setting']['save_folder'] + \
        '_' + config['test']['recon_checkpoint'].replace('/', '_') + '/'
    '''
    save_path = file_path + 'test_' + datetime.datetime.now().strftime("%m%d%H%M") + \
        '_' + config['setting']['save_folder'] + \
        '_' + config['test']['recon_checkpoint'].replace('/', '_') + '/'
    check_and_mkdir(save_path)

    check_and_mkdir(save_path + 'recon_model/')
    shutil.copy(recon_checkpoint, save_path + 'recon_model/')

    print("Writing results....")
    write_test(log_dict=metrics.result(), img_dict=images.result(), save_path=save_path,
               is_save_mat=config['test']['is_save_mat'])

from dataset.pmri_fastmri_brain import RealMeasurement, uniformly_cartesian_mask, fmult, ftran

def pruning_test(
        dataset,
        recon_module: nn.Module,
        regis_module: nn.Module,
        config, recon_module_type=None
):
    # Dataset processing
    generator = torch.Generator()
    generator.manual_seed(0)
    data_split_ratio = config['setting']['data_split_ratio']
    dataset_lengths = [int(len(dataset) * data_split_ratio[0]), int(len(dataset) * data_split_ratio[1]), len(dataset) - int(len(dataset) * data_split_ratio[0]) - int(len(dataset) * data_split_ratio[1])]
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, dataset_lengths,
                                                                               generator=generator)

    if recon_module_type in ["E2EVARNET", "VARNET"]:
        train_dataset = DatasetWrapper(train_dataset);valid_dataset = DatasetWrapper(valid_dataset);test_dataset = DatasetWrapper(test_dataset)

    print("[train_dataset] total_len: ", train_dataset.__len__())
    print("[valid_dataset] total_len: ", valid_dataset.__len__())
    print("[test_dataset] total_len: ", test_dataset.__len__())

    file_path = config['setting']['root_path'] + config['setting']['save_folder'] + '/'
    mul_coil = config['dataset']['multi_coil']
    if (recon_module_type == None):
        recon_module_type = config['pruning']['module']

    tensorboard_batch = config['train']['tensorboard_batch']

    #recon_checkpoint = file_path + config['test']['recon_checkpoint']
    #recon_module.load_state_dict(torch.load(recon_checkpoint))

    recon_module.cuda()
    recon_module.eval()

    recon_callbacks = CallbackList(callbacks=[
        BaseLogger(file_path=None),
        Tensorboard(file_path=file_path, per_batch=tensorboard_batch),
        ModelCheckpoint(file_path=None,
                        period=1,
                        monitors=['valid_psnr', 'valid_ssim'],
                        modes=['max', 'max'])
    ])

    recon_callbacks.set_module(recon_module)
    recon_callbacks.set_params({
        'config': config,
        'train_epoch': test_dataset.__len__()
    })

    metrics = Stack()
    images = Stack()
    batch_size = config['train']['batch_size']

    num_workers = config['train']['num_workers']
    train_epoch = config['train']['train_epoch']
    verbose_batch = config['train']['verbose_batch']
    tensorboard_batch = config['train']['tensorboard_batch']
    checkpoint_epoch = config['train']['checkpoint_epoch']

    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=num_workers)
    train_iter_total = int(train_dataset.__len__() / 1)

    valid_dataloader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False)
    valid_iter_total = int(valid_dataset.__len__() / 1)
    test_dataloader = DataLoader(
            test_dataset, batch_size=1, shuffle=False)
    test_iter_total = int(test_dataset.__len__() / 1)

    #test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    zero_filled_loss_ssim_sum = 0
    zero_filled_loss_psnr_sum = 0
    loss_ssim_sum = 0
    loss_psnr_sum = 0
    loss_count = 0
    nparam_sum = 0

    test_time_sum = 0

    with torch.no_grad():
        #iter_ = tqdm(test_dataloader, desc='Test', total=test_iter_total)
        iter_ = tqdm(test_dataloader, desc='Test', total=test_iter_total)
        for i, test_data in enumerate(iter_):
            single_y_tran, mul_y_tran, single_y, mul_y, x, mask, sensitivity_map = (i.cuda() for i in test_data)

            if mul_coil == False:
                y = single_y
                mask = mask
                y_tran = single_y_tran
            else:  # mul_coil == True
                y = mul_y
                mask = mask
                y_tran = mul_y_tran

            #print(f'zero_filled_psnr: {compare_psnr(abs_helper(mul_y_tran), abs_helper(x)).item()}')

            # Input Data Tupling
            #print(recon_module_type)
            XPSY_test = inputDataDict(y_tran, mask, sensitivity_map, y, module_name=recon_module_type)

            # time initial
            #torch.cuda.synchronize()
            #starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            #num_repetition = len(iter_)
            #timings = np.zeros((num_repetition, 1))
            # print(f"haha mask.shape: {mask.shape}")

            #starter.record()
            #init_time = time.time()
            if recon_module_type == "DEQ" and config['setting']['purpose'] != "pruning":
                y_tran_recon, forward_iter, forward_res = recon_module(XPSY_test)
            elif recon_module_type == "E2EVARNET":
                [y_tran_recon, estimated_sen_map] = recon_module(XPSY_test)
            else:
                y_tran_recon = recon_module(XPSY_test)
            #print(len(y_tran_recon))
            #test_time = time.time() - init_time
            #nparam = sum(param.numel() for param in recon_module.parameters())

            #ender.record()
            #torch.cuda.synchronize()
            #timings[i] = starter.elapsed_time(ender)

            # to remove the gray background
            #fixed_y_tran[fixed_x == 0] = 0
            #fixed_y_tran_recon[fixed_x == 0] = 0
            # Multipling mask to remove the gray background
            '''
            fixed_y_tran_recon = torch.view_as_complex(fixed_y_tran_recon.permute([0, 2, 3, 1]).contiguous())
            fixed_mask = torch.view_as_complex(fixed_mask)
            fixed_y_tran_recon = fixed_y_tran_recon * fixed_mask
            fixed_y_tran_recon = torch.view_as_real(fixed_y_tran_recon)
            fixed_y_tran_recon = fixed_y_tran_recon.permute([0, 3, 1, 2]).contiguous()
            fixed_mask = torch.view_as_real(fixed_mask)
            '''

            #y_tran[x == 0] = 0
            y_tran_recon[x == 0] = 0

            #y_tran_recon = abs_helper(y_tran_recon)
            #x = abs_helper(x)
            #y_tran = abs_helper(y_tran)


            log_batch = {
                'zero_filled_ssim': compare_ssim(absolute_helper(crop_images(y_tran)), absolute_helper(crop_images(x))).item(),
                'zero_filled_psnr': compare_psnr(absolute_helper(crop_images(y_tran)), absolute_helper(crop_images(x))).item(),
                'prediction_ssim': compare_ssim(absolute_helper(crop_images(y_tran_recon)), absolute_helper(crop_images(x))).item(),
                'prediction_psnr': compare_psnr(absolute_helper(crop_images(y_tran_recon)), absolute_helper(crop_images(x))).item(),
            }

            images_batch = {
                'groundtruth': abs_helper(x).detach().cpu().numpy(),
                'zero_filled': abs_helper(y_tran).detach().cpu().numpy(),
                'prediction': abs_helper(y_tran_recon).detach().cpu().numpy(),

            }
            plot_helper(file_path="reconstruction.png",
                        # img1=(torch.view_as_complex(x_gt.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
                        img1=(torch.view_as_complex(x.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
                        img2=(torch.view_as_complex(y_tran.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
                        img3=(torch.view_as_complex(y_tran_recon.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
                        img4=(torch.view_as_complex(y_tran_recon.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
                        img1_name='Ground Truth', img2_name='Input', img3_name='Recon',
                        img4_name='Recon-' + str(i), title='Test')



            metrics.update_state(log_batch)
            images.update_state(images_batch)
            recon_callbacks.call_batch_end_hook(log_batch, 1)
            if i % 15 == 0:
                iter_.write(("[Test Pruning]:") + dict2pformat(log_batch))
                iter_.update()
            zero_filled_loss_ssim_sum += compare_ssim(absolute_helper(crop_images(y_tran)), absolute_helper(crop_images(x))).item()
            zero_filled_loss_psnr_sum += compare_psnr(absolute_helper(crop_images(y_tran)), absolute_helper(crop_images(x))).item()
            loss_ssim_sum += compare_ssim(absolute_helper(crop_images(y_tran_recon)), absolute_helper(crop_images(x))).item()
            loss_psnr_sum += compare_psnr(absolute_helper(crop_images(y_tran_recon)), absolute_helper(crop_images(x))).item()
            #test_time_sum += test_time
            #nparam_sum += nparam
            loss_count += 1

    print(f"%%%[Test SSIM: {loss_ssim_sum/loss_count} / Test PSNR: {loss_psnr_sum/loss_count}]")

    return zero_filled_loss_ssim_sum/loss_count, zero_filled_loss_psnr_sum/loss_count, loss_ssim_sum/loss_count, loss_psnr_sum/loss_count

    '''
    save_path = file_path + 'test_' + \
        '_' + config['setting']['save_folder'] + \
        '_' + config['test']['recon_checkpoint'].replace('/', '_') + '/'
    '''
    #save_path = file_path + 'test_' + datetime.datetime.now().strftime("%m%d%H%M") + \
    #            '_' + config['setting']['save_folder'] + \
    #            '_' + config['test']['recon_checkpoint'].replace('/', '_') + '/'
    #check_and_mkdir(save_path)

    #check_and_mkdir(save_path + 'recon_model/')
    #shutil.copy(recon_checkpoint, save_path + 'recon_model/')

    #print("Writing results....")
    #write_test(log_dict=metrics.result(), img_dict=images.result(), save_path=save_path, is_save_mat=config['test']['is_save_mat'])

def noise_generator(x, sigma):
    if len(x.shape) == 3:
        noise = torch.randn(x.shape[0], x.shape[1], x.shape[2]).uniform_(0, 1).cuda()
    elif len(x.shape) == 4:
        noise = torch.randn(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).uniform_(0, 1).cuda()
    noise = noise * (sigma/255)
    x_hat = x + noise
    return x_hat

def absolute_helper(img, is_normalized=True):
    '''
    print(x.shape)
    if len(x.shape) == 3:
        n_slice, n_x, n_y = x.shape

        if is_normalized:
            for i in range(n_slice):
                x[i] -= torch.min(x[i])
                x[i] /= torch.max(x[i])

                x[i] *= 255

            x = x.to(torch.float32)

    else:
        x = x.permute([0,2,3,1]).contiguous()
        n_slice, n_x, n_y, n_c = x.shape

        #x = post_processing(np.expand_dims(x, -1))
        #x = np.expand_dims(x, -1)
        #x = np.squeeze(x)
        #x = x.unsqueeze(-1)
        #x = x.squeeze()

        if is_normalized:
            for i in range(n_slice):
                for j in range(n_c):
                    x[i, :, :, j] -= torch.min(x[i, :, :, j])
                    x[i, :, :, j] /= torch.max(x[i, :, :, j])

                    x[i, :, :, j] *= 255
            x = x.permute([0,3,1,2]).contiguous()
            x = x.to(torch.float32)

    return x
    '''
    img = torch.sqrt(torch.sum(img ** 2, dim=1, keepdim=True))
    if is_normalized:
        for i in range(img.shape[0]):
            img[i] = (img[i] - torch.min(img[i])) / (torch.max(img[i]) - torch.min(img[i]) + 1e-16)

    img = img.to(torch.float32)

    # img = torch.abs(img)
    # img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))

    return img


import json
# from sota_module.method.baseline.istanetplus import ISTANetPlusLightening
from sota_module.method.baseline.e2evarnet import E2EVarNetWrapper, DatasetWrapper

def pruning_recon_module(model, importance, dataset, sparsity, config, module_name=None, fine_tuning=True):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    nf_enc = config['module']['regis']['nf_enc']
    nf_dec = config['module']['regis']['nf_dec']
    iteration_k = config['module']['recon']['iteration_k']
    mu_list = config['module']['recon']['mu_list']  # for RED
    gamma_list = config['module']['recon']['gamma_list']  # for pnp and RED
    alpha_list = config['module']['recon']['alpha_list']  # for pnp
    recon_module_type = config['module']['recon']['recon_module_type']
    batch_size = config['train']['batch_size']
    num_workers = config['train']['num_workers']
    # sparsity = config['pruning']['ch_sparsity']
    mul_coil = config['dataset']['multi_coil']
    # %%%%%%%% Dataset processing %%%%%%%%
    generator = torch.Generator()
    generator.manual_seed(0)
    # Dataset processing
    data_split_ratio = config['setting']['data_split_ratio']
    dataset_lengths = [int(len(dataset) * data_split_ratio[0]), int(len(dataset) * data_split_ratio[1]),
                       len(dataset) - int(len(dataset) * data_split_ratio[0]) - int(len(dataset) * data_split_ratio[1])]

    ########## Produce Dummy Input for the pruning ##########
    dummy_index = [0]
    dummy_dataset = MoDLDataset()
    # dummy_train_dataset, dummy_valid_dataset, dummy_test_dataset = torch.utils.data.random_split(dummy_dataset, dataset_lengths, generator=generator)
    if module_name in ["E2EVARNET", "VARNET"]:
        dummy_dataset = DatasetWrapper(dummy_dataset)
        # dummy_train_dataset = DatasetWrapper(dummy_train_dataset);dummy_valid_dataset = DatasetWrapper(dummy_valid_dataset);dummy_test_dataset = DatasetWrapper(dummy_test_dataset)
    _, dummy_mul_y_tran, _, dummy_mul_y, dummy_x, dummy_mask, dummy_sensitivity_map = \
        (i.cuda() for i in next(iter(DataLoader(Subset(dummy_dataset, dummy_index), batch_size=1))))
    dummy_inputs = inputDataDict(dummy_mul_y_tran, dummy_mask, dummy_sensitivity_map, dummy_mul_y,
                                 module_name=module_name)
    #########################################################

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, dataset_lengths,
                                                                               generator=generator)
    if module_name in ["E2EVARNET", "VARNET"]:
        train_dataset = DatasetWrapper(train_dataset);
        valid_dataset = DatasetWrapper(valid_dataset);
        test_dataset = DatasetWrapper(test_dataset)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    train_iter_total = int(train_dataset.__len__() / batch_size)

    valid_dataloader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False)
    valid_iter_total = int(valid_dataset.__len__() / 1)

    # %%%%%%% Example Input Data Processing %%%%%%%
    sample_indices = [2]
    
    valid_sample_single_y_tran, valid_sample_mul_y_tran, valid_sample_single_y, valid_sample_mul_y, valid_sample_x, valid_sample_mask, valid_sample_sensitivity_map = \
        (i.cuda() for i in next(iter(
            DataLoader(Subset(valid_dataset, sample_indices), batch_size=len(sample_indices)))))

    if mul_coil == False:
        valid_sample_y = valid_sample_single_y
        valid_sample_mask = valid_sample_mask
        valid_sample_y_tran = valid_sample_single_y_tran
    else:  # mul_coil == True
        valid_sample_y = valid_sample_mul_y
        valid_sample_mask = valid_sample_mask
        valid_sample_y_tran = valid_sample_mul_y_tran
    unpruned_model = copy.deepcopy(model)

    example_inputs = inputDataDict(valid_sample_y_tran, valid_sample_mask, valid_sample_sensitivity_map, valid_sample_y,
                                   module_name=module_name)

    ignored_layers = []

    for ii, m in enumerate(model.modules()):

        if module_name in ["E2EVARNET", "VARNET"]:

            try:
                ignored_layers.append(m.varnet.sens_net)
            except:
                pass

        if (ii > sum(1 for _ in model.modules()) - 2):
            ignored_layers.append(m)

    model = model.cuda()

    model_sum = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if module_name in ["E2EVARNET", "VARNET"]:
        indexOfWeight = [0, 1, 2, 3, 4, 5, 6, 7]
        detected_unwrapped = []
        for i in indexOfWeight:
            detected_unwrapped.append(model.varnet.cascades[i].dc_weight)
        # = [model.varnet.cascades[0].dc_weight, model.varnet.cascades[2].dc_weight, model.varnet.cascades[0].dc_weight]
        unwrapped_parameters = []
        for ii, v in enumerate(detected_unwrapped):
            unwrapped_parameters.append((v, 0))
    else:
        unwrapped_parameters = None

    iterative_steps = config['pruning']['iterative_steps']  # progressive pruning

    if importance == "random":
        imp = tp.importance.RandomImportance()
        pruner = tp.pruner.MagnitudePruner(
            model,
            dummy_inputs,
            importance=imp,
            iterative_steps=iterative_steps,
            ch_sparsity=sparsity,
            # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            ignored_layers=ignored_layers,
            # unwrapped_parameters = unwrapped_parameters
        )
    elif importance == "l1":
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner = tp.pruner.MagnitudePruner(
            model,
            dummy_inputs,
            importance=imp,
            iterative_steps=iterative_steps,
            ch_sparsity=sparsity,
            # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            ignored_layers=ignored_layers
        )
    else:
        raise ValueError("Check the importance: {importance}")


    if fine_tuning == False:
        base_macs, base_nparams = tp.utils.count_ops_and_params(model, dummy_inputs)
        initial_nparams = base_nparams
        final_nparams = 0
        for i in range(iterative_steps):
            pruner.step()
            macs, nparams = tp.utils.count_ops_and_params(model, dummy_inputs)
            if i == iterative_steps - 1:
                final_nparams = nparams
            print(
                "  Iter %d/%d, Params: %.2f M => %.2f M"
                % (i + 1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
            )
            print(
                "  Iter %d/%d, MACs: %.2f G => %.2f G"
                % (i + 1, iterative_steps, base_macs / 1e9, macs / 1e9)
            )
    else:

        base_macs, base_nparams = tp.utils.count_ops_and_params(model, dummy_inputs)
        initial_nparams = base_nparams
        final_nparams = 0
        for i in range(iterative_steps):
            pruner.step()
            macs, nparams = tp.utils.count_ops_and_params(model, dummy_inputs)

            if i == iterative_steps - 1:
                final_nparams = nparams

            print(
                "  Iter %d/%d, Params: %.2f M => %.2f M"
                % (i + 1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
            )
            print(
                "  Iter %d/%d, MACs: %.2f G => %.2f G"
                % (i + 1, iterative_steps, base_macs / 1e9, macs / 1e9)
            )
            if importance == "group_norm":
                pruner.getGroupUpdate()
                if config['pruning']["student_teacher"] == True:
                    model, pruner = recon_or_prior_train(
                        dataset=dataset,
                        recon_module=model,
                        regis_module=None,
                        config=config, root_path=config['reconstruction']['save_root_path'], module_name=module_name,
                        pruner=pruner, unpruned_recon_module=unpruned_model, pruning_importance=importance
                    )
                else:
                    model, pruner = recon_or_prior_train(
                        dataset=dataset,
                        recon_module=model,
                        regis_module=None,
                        config=config, root_path=config['reconstruction']['save_root_path'], module_name=module_name,
                        pruner=pruner, pruning_importance=importance
                    )
            else:
                if config['pruning']["student_teacher"] == True:
                    model = recon_or_prior_train(
                        dataset=dataset,
                        recon_module=model,
                        regis_module=None,
                        config=config, root_path=config['reconstruction']['save_root_path'],
                        module_name=module_name, unpruned_recon_module=unpruned_model, pruning_importance=importance
                    )
                else:
                    model = recon_or_prior_train(
                        dataset=dataset,
                        recon_module=model,
                        regis_module=None,
                        config=config, root_path=config['reconstruction']['save_root_path'],
                        module_name=module_name, pruning_importance=importance
                    )
    return model, unpruned_model, initial_nparams, final_nparams



def recon_or_prior_train(
        dataset,
        recon_module,
        config, root_path, unpruned_recon_module = None, regis_module=False, sigma=0.9, is_optimize_regis = False, module_name=None, pruner = None, pruning_importance=None
):
    mul_coil = config['dataset']['multi_coil']
    purpose = config['setting']['purpose']
    print(f"%%%%%%[{purpose}]Is it multi coil? {mul_coil}%%%%%%")
    # Dataset processing
    generator = torch.Generator()
    generator.manual_seed(0)
    data_split_ratio = config['setting']['data_split_ratio']
    dataset_lengths = [int(len(dataset) * data_split_ratio[0]), int(len(dataset) * data_split_ratio[1]), len(dataset) - int(len(dataset) * data_split_ratio[0]) - int(len(dataset) * data_split_ratio[1])]
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, dataset_lengths,
                                                                               generator=generator)

    if module_name in ["E2EVARNET", "VARNET"]:
        train_dataset = DatasetWrapper(train_dataset);valid_dataset = DatasetWrapper(valid_dataset);test_dataset = DatasetWrapper(test_dataset)

    print("[train_dataset] total_len: ", train_dataset.__len__())

    print("[valid_dataset] total_len: ", valid_dataset.__len__())

    ########################
    # Load Configuration
    ########################
    recon_batch = config['method']['proposed']['recon_batch']
    #is_optimize_regis = config['method']['proposed']['is_optimize_regis']
    #recon_module_type = config['module']['recon']['recon_module_type']
    recon_lr, regis_lr = config['train']['recon_lr'], config['train']['regis_lr']
    recon_loss, regis_loss = config['train']['recon_loss'], config['train']['regis_loss']

    batch_size = config['train']['batch_size']

    num_workers = config['train']['num_workers']
    train_epoch = config['train']['train_epoch']
    if module_name == "DEQ" and (config['setting']['purpose'] == 'pruning' and config['pruning']['usage_of_gt'] == True and config['pruning']['fine_tune_loss_type'] == "supervised"):
        train_epoch = 25
    elif module_name == "DEQ" and config['setting']['purpose'] == 'pruning':
        train_epoch = 100
    elif config['pruning']['usage_of_gt'] == True and config['pruning']['fine_tune_loss_type'] == "supervised":
        train_epoch = 100
    else:
        train_epoch = 150

    verbose_batch = config['train']['verbose_batch']
    tensorboard_batch = config['train']['tensorboard_batch']
    checkpoint_epoch = config['train']['checkpoint_epoch']

    check_and_mkdir(config['setting']['root_path'])
    file_path = config['setting']['root_path'] + config['setting']['save_folder'] + '/'
    loss_recon_consensus_COEFF = config['method']['loss_recon_consensus']
    ########################
    # Dataset
    ########################

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    train_iter_total = int(train_dataset.__len__() / batch_size)

    valid_dataloader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False)
    valid_iter_total = int(valid_dataset.__len__() / 1)

    test_dataloader = DataLoader(
            test_dataset, batch_size=1, shuffle=False)
    test_iter_total = int(test_dataset.__len__() / 1)



    if config['pruning']['usage_of_gt'] == False and config['pruning']['fine_tune_loss_type'] == "supervised":
        raise ValueError("Usage of gt and the fine tuning loss type do not match.")

    if (config['setting']['purpose'] == 'pruning' and config['pruning']['usage_of_gt'] == False):
        train_dataloader = test_dataloader
        train_iter_total = test_iter_total

    sample_indices = [2]

    #single_x_init,mul_x_init, single_y,mul_y, x, P, S,
    #valid_sample_fixed_x, valid_sample_moved_x, valid_sensitivity_map, valid_sample_single_fixed_y, valid_sample_single_fixed_mask, valid_sample_single_fixed_y_tran, valid_sample_single_moved_y, valid_sample_single_moved_mask, valid_sample_single_moved_y_tran, \
    #valid_sample_mul_fixed_y, valid_sample_mul_fixed_mask, valid_sample_mul_fixed_y_tran, valid_sample_mul_moved_y, valid_sample_mul_moved_y_tran, valid_sample_mul_moved_mask = \
    valid_sample_single_y_tran, valid_sample_mul_y_tran, valid_sample_single_y, valid_sample_mul_y, valid_sample_x, valid_sample_mask, valid_sample_sensitivity_map = \
        (i.cuda() for i in next(iter(
            DataLoader(Subset(valid_dataset, sample_indices), batch_size=len(sample_indices)))))

    print(f"valid_sample_single_y_tran.shape: {valid_sample_single_y_tran.shape}")
    print(f"valid_sample_mask.shape: {valid_sample_mask.shape}")

    if purpose == "prior_training":
        module_name = config['prior_training']['module']
        if mul_coil == False:
            # Process Single Coil Data
            valid_sample_single_y_tran = addwgn(valid_sample_x, config['prior_training']['prior_value'])
            #valid_sample_single_moved_y_tran = noise_generator(valid_sample_single_y_tran, 0.9)
        else:
            # Process Multi Coil Data
            valid_sample_mul_y_tran = addwgn(valid_sample_x, config['prior_training']['prior_value'])
            #valid_sample_mul_moved_y_tran = noise_generator(valid_sample_mul_moved_y_tran, 0.9)
    elif purpose == "reconstruction":
        module_name = config['reconstruction']['module']

    if mul_coil == False:
        valid_sample_y = valid_sample_single_y
        valid_sample_mask = valid_sample_mask
        valid_sample_y_tran = valid_sample_single_y_tran
    else:  # mul_coil == True
        valid_sample_y = valid_sample_mul_y
        valid_sample_mask = valid_sample_mask
        valid_sample_y_tran = valid_sample_mul_y_tran

    XPSY_valid_sample = inputDataDict(valid_sample_y_tran, valid_sample_mask, valid_sample_sensitivity_map, valid_sample_y, module_name=module_name)

    valid_sample_y_tran[valid_sample_x == 0] = 0

    image_init = {
        'groundtruth': abs_helper(valid_sample_x),
        'zero-filled': abs_helper(valid_sample_y_tran),
    }

    ########################
    # Metrics
    ########################
    metrics = Mean()

    ########################
    # Extra-Definition
    ########################
    recon_module.cuda()
    recon_module.train()

    sim_loss_fn = losses.ncc_loss
    grad_loss_fn = losses.gradient_loss
    mse_loss_fn = losses.mse_loss

    if config['pruning']['fine_tune_loss_type'] == "testdc" and config['setting']['purpose'] == "pruning":
        loss_fn_dict = {
            'l1': nn.L1Loss,
            'l2': nn.MSELoss,
            'smooth_l1': nn.SmoothL1Loss,
            'varnet_loss': fastmri.SSIMLoss(),
            'spice_varnet_loss': fastmri.SPICELoss(),
            'rei_loss': fastmri.REILoss(recon_module = recon_module, config=config, module_name = module_name),
            'rei_loss_forISTA': fastmri.REILossForISTA(recon_module = recon_module, config=config, module_name=module_name)
        }
    else:
        loss_fn_dict = {
            'l1': nn.L1Loss,
            'l2': nn.MSELoss,
            'smooth_l1': nn.SmoothL1Loss,
            'varnet_loss': fastmri.SSIMLoss(),
            'spice_varnet_loss': fastmri.SPICELoss()
        }
    if module_name in ["VARNET","E2EVARNET"]:
        if config['setting']['purpose'] == "pruning" and config['pruning']['fine_tune_loss_type'] in ["testdc"]:
            recon_loss_fn = loss_fn_dict['rei_loss']
            recon_optimizer = Adam(recon_module.parameters(), lr=1e-3, weight_decay=1e-8)
            from torch.optim.lr_scheduler import MultiStepLR
            # Reduce the learning rate at epoch 30. We start with 0.001 and then switch to 0.0001 after 30 epochs.
            scheduler = MultiStepLR(recon_optimizer, milestones=[30], gamma=0.1)

        else:
            recon_loss_fn = loss_fn_dict['varnet_loss']
            recon_optimizer = Adam(recon_module.parameters(), lr=1e-3, weight_decay=1e-8)

    elif module_name == "E2EVARNET":
        if config['setting']['purpose'] == "pruning" and config['pruning']['fine_tune_loss_type'] in ["testdc"]:
            recon_loss_fn = loss_fn_dict['spice_varnet_loss']
            recon_optimizer = Adam(recon_module.parameters(), lr=1e-3)
        else:
            recon_loss_fn = loss_fn_dict['varnet_loss']
            recon_optimizer = Adam(recon_module.parameters(), lr=1e-3)

    else:
        if config['setting']['purpose'] == "pruning" and config['pruning']['fine_tune_loss_type'] in ["testdc"]:
            recon_loss_fn = loss_fn_dict['rei_loss']
            # recon_optimizer = Adam(recon_module.parameters(), lr=1e-5)
            recon_optimizer = Adam(recon_module.parameters(), lr=1e-5, weight_decay=1e-8)
        else:
            recon_loss_fn = loss_fn_dict[recon_loss]()
            recon_optimizer = Adam(recon_module.parameters(), lr=recon_lr)
        if module_name not in ["E2EVARNET", "VARNET","EDSR", "UNET", "DEQ", "DU", "PNP", "RED"]:
            raise ValueError("Not valid module name has assigned")

    recon_loss_fn = recon_loss_fn.to(device)
    ########################
    # Begin Training
    ########################

    check_and_mkdir(file_path)
    #file_path = root_path + module_name +'/'
    recon_callbacks = CallbackList(callbacks=[
        BaseLogger(file_path=file_path),
        Tensorboard(file_path=file_path, per_batch=tensorboard_batch),
        ModelCheckpoint(file_path=file_path + 'recon_model/',
                        period=checkpoint_epoch,
                        monitors=['valid_psnr', 'valid_ssim'],
                        modes=['max', 'max']),
    ])

    recon_callbacks.set_module(recon_module)
    recon_callbacks.set_params({
        'config': config,
        "lr": recon_lr,
        'train_epoch': train_epoch
    })

    recon_callbacks.call_train_begin_hook(image_init)

    #train_dataloader = valid_dataloader

    global_batch = 1
    for global_epoch in range(1, train_epoch):
        iter_ = tqdm(train_dataloader, desc='Train [%.3d/%.3d]' % (global_epoch, train_epoch), total=train_iter_total)
        recon_module.train()
        if unpruned_recon_module != None:
            unpruned_recon_module.train()
        for i, train_data in enumerate(iter_):

            single_y_tran, mul_y_tran, single_y, mul_y, x, mask, sensitivity_map = (i.cuda() for i in train_data)

            if purpose == "prior_training":
                single_y_tran = addwgn(x, config['prior_training']['prior_value'])
                mul_y_tran = addwgn(x, config['prior_training']['prior_value'])

            if mul_coil == False:
                y = single_y
                mask = mask
                y_tran = single_y_tran
            else:  # mul_coil == True
                y = mul_y
                mask = mask
                y_tran = mul_y_tran

            XPSY_train = inputDataDict(y_tran, mask, sensitivity_map, y, module_name=module_name)

            log_batch = {}

            for j in range(recon_batch):
                ########################
                # Differentiate different modules
                ########################
                # if module_name == "DEQ" and config['setting']['purpose'] != "pruning":
                if module_name == "DEQ" and config['setting']['purpose'] != "pruning":
                    y_tran_recon, forward_iter, forward_res = recon_module(XPSY_train)

                elif module_name == "E2EVARNET":
                    if config['setting']['purpose'] == "pruning" and config['pruning']['fine_tune_loss_type'] in ["school", "testdcPLUSschool", "supervisedPLUSschool"]:
                        [y_tran_recon, estimated_sen_map] = recon_module(XPSY_train)
                        [up_y_tran_recon, up_estimated_sen_map] = unpruned_recon_module(XPSY_train)
                    else:
                        [y_tran_recon, estimated_sen_map] = recon_module(XPSY_train)

                else:
                    if config['setting']['purpose'] == "pruning" and config['pruning']['fine_tune_loss_type'] in ["school", "testdcPLUSschool", "supervisedPLUSschool"]:
                        #print(f"[Decolearn] y_tran.shape: {y_tran.shape}")
                        up_y_tran_recon = unpruned_recon_module(XPSY_train)
                        y_tran_recon = recon_module(XPSY_train)
                    else:
                        y_tran_recon = recon_module(XPSY_train)

                ########################
                # Cleaning background
                ########################
                if config['setting']['purpose'] == "pruning" and config['pruning']['fine_tune_loss_type'] in ["school", "testdcPLUSschool", "supervisedPLUSschool"]:
                    up_y_tran_recon[x == 0] = 0
                    y_tran_recon[x == 0] = 0
                else:
                    y_tran_recon[x == 0] = 0

                recon_optimizer.zero_grad()

                if config['setting']['purpose'] == "pruning":
                    if config['pruning']['usage_of_gt'] == True:
                        if config['pruning']['fine_tune_loss_type'] == "supervised":
                            recon_loss = recon_loss_fn(y_tran_recon, x)
                        elif config['pruning']['fine_tune_loss_type'] == "supervisedPLUSschool":
                            recon_loss1 = recon_loss_fn(y_tran_recon, x)
                            # Use output of big network as soft target for small network
                            T = 1  # Temperature parameter for softmax
                            soft_targets = up_y_tran_recon.clone().detach()
                            # Calculate distillation loss between small network output and soft targets
                            distillation_loss = nn.functional.mse_loss(y_tran_recon, soft_targets)
                            recon_loss = recon_loss1 + distillation_loss

                        else:
                            raise ValueError("There is no matching supervised tuning method.")

                    elif config['pruning']['usage_of_gt'] == False:
                        if config['pruning']['fine_tune_loss_type'] == "testdcPLUSschool":
                            if mul_coil == True:
                                if module_name == "E2EVARNET":
                                    recon_loss_consensus1 = nn.MSELoss()(mul_fmult(torch.view_as_complex(y_tran_recon.permute([0, 2, 3, 1]).contiguous()),sensitivity_map, mask), mul_y)
                                else:
                                    recon_loss_consensus1 = recon_loss_fn(mul_fmult(
                                        torch.view_as_complex(y_tran_recon.permute([0, 2, 3, 1]).contiguous()),
                                        sensitivity_map, mask), mul_y)

                            else:  # mul_coil == False
                                if module_name == "E2EVARNET":
                                    recon_loss_consensus1 = nn.MSELoss()(single_fmult(torch.view_as_complex(y_tran_recon.permute([0, 2, 3, 1]).contiguous()), sensitivity_map, mask), singley)
                                else:
                                    recon_loss_consensus1 = recon_loss_fn(single_fmult(torch.view_as_complex(y_tran_recon.permute([0, 2, 3, 1]).contiguous()), sensitivity_map, mask), single_y)

                            # Use output of big network as soft target for small network
                            T = 1  # Temperature parameter for softmax
                            soft_targets = up_y_tran_recon.clone().detach()
                            # Calculate distillation loss between small network output and soft targets
                            distillation_loss = recon_loss_fn(y_tran_recon, soft_targets)
                            recon_loss = recon_loss_consensus1 + distillation_loss



                        elif config['pruning']['fine_tune_loss_type'] == "testdc":
                            if mul_coil == True:
                                if module_name == "E2EVARNET":
                                    recon_loss_consensus = recon_loss_fn(y0=mul_y, x0=y_tran, x1=y_tran_recon, y1=mul_fmult(torch.view_as_complex(y_tran_recon.permute([0, 2, 3, 1]).contiguous()), sensitivity_map, mask), S=sensitivity_map, P=mask)
                                    # recon_loss_consensus = recon_loss_fn(X=y_tran_recon, Y=y, P=mask, S=sensitivity_map, S_estimated=estimated_sen_map)
                                else:
                                    recon_loss_consensus = recon_loss_fn(y0=mul_y, x0=y_tran, x1=y_tran_recon, y1=mul_fmult(torch.view_as_complex(y_tran_recon.permute([0, 2, 3, 1]).contiguous()), sensitivity_map, mask), S=sensitivity_map, P=mask)
                                    # recon_loss_consensus = nn.L1Loss()(mul_fmult(torch.view_as_complex(y_tran_recon.permute([0, 2, 3, 1]).contiguous()), sensitivity_map, mask), mul_y)
                            else:  # mul_coil == False
                                if module_name == "E2EVARNET":
                                    raise ValueError("Single coil is not implemented yet.")
                                    raise ValueError("VARNET DOES NOT HAVE THE PROCESS FOR THE SINGLE COIL DATA")
                                else:
                                    raise ValueError("Single coil is not implemented yet.")
                                    recon_loss_consensus = recon_loss_fn(single_fmult(
                                        torch.view_as_complex(y_tran_recon.permute([0, 2, 3, 1]).contiguous()),
                                        sensitivity_map, mask), single_y)
                            recon_loss = loss_recon_consensus_COEFF * (recon_loss_consensus)

                        elif config['pruning']['fine_tune_loss_type'] == "school":
                            T = 1  # Temperature parameter for softmax
                            soft_targets = up_y_tran_recon.clone().detach()
                            recon_loss = recon_loss_fn(y_tran_recon, soft_targets)
                        else:
                            raise ValueError("There is no matching self-supervised tuning method.")
                    else:
                        raise ValueError("Define usage_of_gt with true or false")

                else:# config['setting']['purpose'] == "reconstruction":
                    recon_loss = recon_loss_fn(y_tran_recon, x)

                recon_loss.backward()

                if pruner != None and pruning_importance == "group_norm":
                    pruner.regularize(recon_module)

                recon_optimizer.step()

                plot_helper(file_path="reconstruction.png",
                            img1=(torch.view_as_complex(x.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
                            img2=(torch.view_as_complex(y_tran.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
                            img3=(torch.view_as_complex(y_tran_recon.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
                            img4=(torch.view_as_complex(y_tran_recon.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
                            img1_name='Ground Truth', img2_name='Input', img3_name='Recon',
                            img4_name='Recon-' + str(global_epoch), title=str(purpose) + " " + str(module_name))


                y_tran_recon[x == 0] = 0

                if j == (recon_batch - 1):
                    if module_name == "DEQ" and config['setting']['purpose'] != "pruning":
                        log_batch.update({
                            'reconstruction_loss': recon_loss.item(),
                            'gamma': recon_module.getGamma().item(),
                            'mu': recon_module.getMu().item(),
                            'alpha': recon_module.getAlpha().item(),
                            'undersample_ssim': compare_ssim(absolute_helper(crop_images(y_tran)), absolute_helper(crop_images(x))).item(),
                            'undersample_psnr': compare_psnr(absolute_helper(crop_images(y_tran)), absolute_helper(crop_images(x))).item(),
                            'train_ssim': compare_ssim(absolute_helper(crop_images(y_tran_recon)), absolute_helper(crop_images(x))).item(),
                            'train_psnr': compare_psnr(absolute_helper(crop_images(y_tran_recon)), absolute_helper(crop_images(x))).item(),
                            'forward_iter': forward_iter,
                            'forward_res': forward_res
                        })
                    elif module_name == "DEQ" and config['setting']['purpose'] == "pruning":
                        log_batch.update({
                            'reconstruction_loss': recon_loss.item(),
                            'gamma': recon_module.getGamma().item(),
                            'mu': recon_module.getMu().item(),
                            'alpha': recon_module.getAlpha().item(),
                            'undersample_ssim': compare_ssim(absolute_helper(crop_images(y_tran)), absolute_helper(crop_images(x))).item(),
                            'undersample_psnr': compare_psnr(absolute_helper(crop_images(y_tran)), absolute_helper(crop_images(x))).item(),
                            'train_ssim': compare_ssim(absolute_helper(crop_images(y_tran_recon)), absolute_helper(crop_images(x))).item(),
                            'train_psnr': compare_psnr(absolute_helper(crop_images(y_tran_recon)), absolute_helper(crop_images(x))).item(),
                        })

                    else:
                        if module_name in ["DU", "DU3D", "PNP", "PNP3D"]:
                            log_batch.update({
                                'reconstruction_loss': recon_loss.item(),
                                'gamma': recon_module.getGamma().item(),
                                'mu': recon_module.getMu().item(),
                                'alpha': recon_module.getAlpha().item(),
                                'undersample_ssim': compare_ssim(absolute_helper(crop_images(y_tran)), absolute_helper(crop_images(x))).item(),
                                'undersample_ssim': compare_ssim(absolute_helper(crop_images(y_tran)), absolute_helper(crop_images(x))).item(),
                                'undersample_psnr': compare_psnr(absolute_helper(crop_images(y_tran)), absolute_helper(crop_images(x))).item(),
                                'train_ssim': compare_ssim(absolute_helper(crop_images(y_tran_recon)), absolute_helper(crop_images(x))).item(),
                                'train_psnr': compare_psnr(absolute_helper(crop_images(y_tran_recon)), absolute_helper(crop_images(x))).item(),
                            })
                        else:
                            log_batch.update({
                                'reconstruction_loss': recon_loss.item(),
                                'undersample_ssim': compare_ssim(absolute_helper(crop_images(y_tran)), absolute_helper(crop_images(x))).item(),
                                'undersample_ssim': compare_ssim(absolute_helper(crop_images(y_tran)), absolute_helper(crop_images(x))).item(),
                                'undersample_psnr': compare_psnr(absolute_helper(crop_images(y_tran)), absolute_helper(crop_images(x))).item(),
                                'train_ssim': compare_ssim(absolute_helper(crop_images(y_tran_recon)), absolute_helper(crop_images(x))).item(),
                                'train_psnr': compare_psnr(absolute_helper(crop_images(y_tran_recon)), absolute_helper(crop_images(x))).item(),
                            })


            metrics.update_state(log_batch)

            if (verbose_batch > 0) and (global_batch % verbose_batch == 0):
                iter_.write(("Batch [%.7d]:" % global_batch) + dict2pformat(log_batch))
                iter_.update()

            recon_callbacks.call_batch_end_hook(log_batch, global_batch)
            global_batch += 1

        if module_name in ["VARNET", "E2EVARNET"] and config['setting']['purpose'] == "pruning" and config['pruning']['fine_tune_loss_type'] in ["testdc"]:
            scheduler.step()

        recon_module.eval()

        with torch.no_grad():

            iter_ = tqdm(valid_dataloader, desc='Valid [%.3d/%.3d]' % (global_epoch, train_epoch),
                         total=valid_iter_total)
            for i, valid_data in enumerate(iter_):
                single_y_tran, mul_y_tran, single_y, mul_y, x, mask, sensitivity_map = (i.cuda() for i in valid_data)

                if purpose == "prior_training":
                    module_name = config['prior_training']['module']
                    single_y_tran = addwgn(x, config['prior_training']['prior_value'])
                    mul_y_tran = addwgn(x, config['prior_training']['prior_value'])
                elif purpose == "reconstruction":
                    module_name = config['reconstruction']['module']
                #else:
                #    module_name = config['pruning']['module']
                if mul_coil == False:
                    y = single_y
                    mask = mask
                    y_tran = single_y_tran
                else:  # mul_coil == True
                    y = mul_y
                    mask = mask
                    y_tran = mul_y_tran


                XPSY_valid = inputDataDict(y_tran, mask, sensitivity_map, y, module_name=module_name)

                if module_name == "DEQ" and config['setting']['purpose'] != "pruning":
                    y_tran_recon, forward_iter, forward_res = recon_module(XPSY_valid)
                elif module_name == "E2EVARNET":
                    [y_tran_recon, estimated_sen_map] = recon_module(XPSY_valid)
                else:
                    #print(f"[Decolearn] y_tran.shape: {y_tran.shape}")
                    y_tran_recon = recon_module(XPSY_valid)

                y_tran_recon[x == 0] = 0

                if module_name == "PNP":
                    gammaList = recon_module.getGamma()
                    alphaList = recon_module.getAlpha()
                    log_batch = {
                        'gamma': gammaList.item(),
                        'alpha': alphaList.item(),
                        'undersample_ssim': compare_ssim(absolute_helper(crop_images(y_tran)), absolute_helper(crop_images(x))).item(),
                        'undersample_psnr': compare_psnr(absolute_helper(crop_images(y_tran)), absolute_helper(crop_images(x))).item(),
                        'valid_ssim': compare_ssim(absolute_helper(crop_images(y_tran_recon)), absolute_helper(crop_images(x))).item(),
                        'valid_psnr': compare_psnr(absolute_helper(crop_images(y_tran_recon)), absolute_helper(crop_images(x))).item(),
                    }

                elif module_name== "DU":
                    gammaList = recon_module.getGamma()
                    muList = recon_module.getMu()
                    log_batch = {
                        'gamma': gammaList.item(),
                        'mu': muList.item(),
                        'undersample_ssim': compare_ssim(absolute_helper(crop_images(y_tran)), absolute_helper(crop_images(x))).item(),
                        'undersample_psnr': compare_psnr(absolute_helper(crop_images(y_tran)), absolute_helper(crop_images(x))).item(),
                        'valid_ssim': compare_ssim(absolute_helper(crop_images(y_tran_recon)), absolute_helper(crop_images(x))).item(),
                        'valid_psnr': compare_psnr(absolute_helper(crop_images(y_tran_recon)), absolute_helper(crop_images(x))).item(),
                    }
                elif module_name=="DEQ" and config['setting']['purpose'] != "pruning":
                    gammaList = recon_module.getGamma()
                    alphaList = recon_module.getAlpha()
                    log_batch = {
                        'gamma': gammaList.item(),
                        'alpha': alphaList.item(),
                        'undersample_ssim': compare_ssim(absolute_helper(crop_images(y_tran)), absolute_helper(crop_images(x))).item(),
                        'undersample_psnr': compare_psnr(absolute_helper(crop_images(y_tran)), absolute_helper(crop_images(x))).item(),
                        'valid_ssim': compare_ssim(absolute_helper(crop_images(y_tran_recon)), absolute_helper(crop_images(x))).item(),
                        'valid_psnr': compare_psnr(absolute_helper(crop_images(y_tran_recon)), absolute_helper(crop_images(x))).item(),
                        'forward_iter': forward_iter,
                        'forward_res': forward_res,
                    }
                else:
                    log_batch = {
                        'undersample_ssim': compare_ssim(absolute_helper(crop_images(y_tran)), absolute_helper(crop_images(x))).item(),
                        'undersample_psnr': compare_psnr(absolute_helper(crop_images(y_tran)), absolute_helper(crop_images(x))).item(),
                        'valid_ssim': compare_ssim(absolute_helper(crop_images(y_tran_recon)), absolute_helper(crop_images(x))).item(),
                        'valid_psnr': compare_psnr(absolute_helper(crop_images(y_tran_recon)), absolute_helper(crop_images(x))).item(),
                    }

                metrics.update_state(log_batch)


            if module_name == "DEQ" and config['setting']['purpose'] != "pruning":
                valid_sample_y_tran_recon, _, _ = recon_module(XPSY_valid_sample)
            elif module_name == "E2EVARNET":
                [valid_sample_y_tran_recon, valid_sample_estimated_sen_map] = recon_module(XPSY_valid_sample)
            else:
                valid_sample_y_tran_recon = recon_module(XPSY_valid_sample)

            # added
            #valid_sample_y_tran_recon = torch.nn.functional.pad(
            #    torch.sqrt(torch.sum(valid_sample_y_tran_recon ** 2, dim=1, keepdim=True)), [4, 4])


        log_epoch = metrics.result()
        metrics.reset_state()


        valid_sample_y_tran_recon[valid_sample_x == 0] = 0
        if is_optimize_regis == True:
            #print(f"I AM HERE 1")
            image_epoch = {
                'prediction': absolute_helper(valid_sample_y_tran_recon),
            }
        else:
            #print(f"I AM HERE 2")
            image_epoch = {
                'prediction': absolute_helper(valid_sample_y_tran_recon),
                }

        recon_callbacks.call_epoch_end_hook(log_epoch, image_epoch, global_epoch)

    from datetime import datetime

    #checkOutput1 = recon_module(y_tran, mask, sensitivity_map, y)
    #print(f"\n\n\nLASTCHECK checkOutput1.shape: {checkOutput1.shape}")

    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%m_%d_%H_%M")
    save_path = file_path + dt_string + config['setting']['purpose']+ module_name + ".pt"
    torch.save(recon_module.state_dict(), save_path)
    #torch.save(recon_module, "/export/project/p.youngil/CSE400E/saved_model/" + module_name + ".pt")

    #checkOutput2 = recon_module(y_tran, mask, sensitivity_map, y)
    #print(f"\n\n\nLASTCHECK checkOutput2.shape: {checkOutput2.shape}")
    with open('config.json') as File:
        config = json.load(File)

    print(f"%%%% [Pruning] Module Loading: True %%%%")
    if module_name == "EDSR":
        recon_module.load_state_dict(
            torch.load(file_path + 'recon_model/'+'best_valid_psnr.pt', map_location=torch.device(device)))
    elif module_name == "DU":
        recon_module.load_state_dict(
            torch.load(file_path + 'recon_model/'+'best_valid_psnr.pt', map_location=torch.device(device)))
    elif module_name == "PNP":
        recon_module.load_state_dict(
            torch.load(file_path + 'recon_model/'+'best_valid_psnr.pt', map_location=torch.device(device)))
    elif module_name == "DEQ":
        recon_module.load_state_dict(
            torch.load(file_path + 'recon_model/'+'best_valid_psnr.pt', map_location=torch.device(device)))
    elif module_name == "E2EVARNET":
        recon_module.load_state_dict(
            torch.load(file_path + 'recon_model/'+'best_valid_psnr.pt', map_location=torch.device(device)))
    if pruner != None:
        return recon_module, pruner
    else:
        return recon_module

# Function for registration module visualization
# -------------------------------------

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
import tifffile as tiff
import torch
from PIL import Image, ImageDraw
from skimage.measure import find_contours
import scipy.io as sio
import ants

def create_standard_grid(grid):
    n_batch, _, n_width, n_height = grid.shape

    img_ants = ants.from_numpy(np.zeros([n_width, n_height], dtype=np.float32))

    img_array = ants.create_warped_grid(image=img_ants, foreground=0, background=1).numpy(True)
    img_array = np.transpose(img_array, [2, 0, 1])

    img_array -= np.amin(img_array)
    img_array /= np.amax(img_array)

    if n_batch > 1:
        img_grid = torch.stack(n_batch * [torch.from_numpy(img_array)], 0)
    else:
        img_grid = torch.from_numpy(img_array).unsqueeze(0)

    return img_grid

def create_grid_norm(grid):
    n_batch, n_dim, n_width, n_height = grid.shape

    if n_dim == 2:
        norm = torch.sqrt(grid[:, 0, :, :] ** 2 + grid[:, 1, :, :] ** 2)
        norm = norm.unsqueeze(1)

    else:
        raise NotImplementedError("Only n_dim = 2 Supported")

    return norm

