# System libs
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
import time
import cProfile

# Numerical libs
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.misc import imsave
from mir_eval.separation import bss_eval_sources

# Our libs
from arguments import ArgParser
from dataset import MUSICMixDataset
from models import ModelBuilder, activate
from utils import AverageMeter, \
    recover_rgb, magnitude2heatmap,\
    istft_reconstruction, warpgrid, \
    combine_video_audio, save_video, makedirs

from pesq import pesq
from pystoi.stoi import stoi

# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, nets, crit):
        super(NetWrapper, self).__init__()
        self.net_sound, self.net_frame, self.net_sign = nets
        self.crit = crit

    def forward(self, batch_data, args):
        mag_mix = batch_data['mag_mix']
        mags = batch_data['mags']
        frames = batch_data['frames']
        signs = batch_data['signs']
        mag_mix = mag_mix + 1e-10

        N = args.num_mix
        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # 0.0 warp the spectrogram
        if args.log_freq:
            grid_warp = torch.from_numpy(
                warpgrid(B, 256, T, warp=True)).to(args.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp) 
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp)

        # 0.1 calculate loss weighting coefficient: magnitude of input mixture
        if args.weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # 0.2 ground truth masks are computed after warpping!
        gt_masks = [None for n in range(N)]
        for n in range(N):
            if args.mask_type == 'binary':
                # for simplicity, mag_N > 0.5 * mag_mix
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            elif args.mask_type == 'ratio':
                gt_masks[n] = mags[n] / mag_mix
                # clamp to avoid large numbers in ratio masks
                gt_masks[n].clamp_(0., 5.)

        # 0.3 LOG magnitude
        log_mag_mix = torch.log(mag_mix).detach()
        # log_mag_mix = mag_mix

        # 1. forward net_frame -> Bx1xC
        feat_frames = [None for n in range(N)]
        for n in range(N):
            feat_frames[n] = self.net_frame.forward_multiframe(frames[n])
            feat_frames[n] = activate(feat_frames[n], args.img_activation)

        # 2. forward net_sign
        feat_signs = [None for n in range(N)]
        for n in range(N):
            feat_signs[n] = self.net_sign(signs[n])
            feat_signs[n] = activate(feat_signs[n], args.img_activation)

        # 3. sound synthesizer
        pred_masks = [None for n in range(N)]
        for n in range(N):
            pred_masks[n] = self.net_sound(log_mag_mix, feat_frames[n], feat_signs[n])
            pred_masks[n] = activate(pred_masks[n], args.output_activation)

        # 4. loss
        err = self.crit(pred_masks, gt_masks, weight).reshape(1)

        return err, \
            {'pred_masks': pred_masks, 'gt_masks': gt_masks, 'weight': weight,
             'mag_mix': mag_mix, 'mags': mags
            }

# Save checkpoints for networks
def save_checkpoint(nets, optimizer, history, epoch, args):
    print('Saving save_checkpoints at {} epochs.'.format(epoch))
    (net_sound, net_frame, net_sign) = nets
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    state = {'epoch': epoch, \
             'net_sound': net_sound.state_dict(), \
             'net_frame': net_frame.state_dict(), \
             'net_sign': net_sign.state_dict(), \
             'optimizer': optimizer.state_dict(),
             'history': history}

    torch.save(state, '{}/checkpoint_{}'.format(args.store_path, suffix_latest))

    cur_err = history['val']['err'][-1]
    if cur_err < args.best_err:
        args.best_err = cur_err
        torch.save(state, '{}/checkpoint_{}'.format(args.store_path, suffix_best))

# Calculate metrics
def calc_metrics(batch_data, outputs, args):
    # meters
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()
    pesq_meter = AverageMeter()
    stoi_meter = AverageMeter()
    gt_wavs = []
    pred_wavs = []

    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    audios = batch_data['audios']

    pred_masks_ = outputs['pred_masks']

    # unwarp log scale
    N = args.num_mix
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, pred_masks_[0].size(3), warp=False)).to(args.device)
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp)
        else:
            pred_masks_linear[n] = pred_masks_[n]

    # convert into numpy
    mag_mix = mag_mix.numpy()
    phase_mix = phase_mix.numpy()
    for n in range(N):
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()

        # threshold if binary mask
        if args.mask_type == 'binary':
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_binary_threshold).astype(np.float32)

    # loop over each sample
    for j in range(B):
        # save mixture
        # mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop, win_length=args.stft_win)

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # Predicted audio recovery
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

        # separation performance computes
        L = preds_wav[0].shape[0]
        gts_wav = [None for n in range(N)]
        valid = True
        for n in range(N):
            gts_wav[n] = audios[n][j, 0:L].numpy()
            valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
            valid *= np.sum(np.abs(preds_wav[n])) > 1e-5
        if valid:
            gt_wavs.append(np.asarray(gts_wav))
            pred_wavs.append(np.asarray(preds_wav))
            # sdr, sir, sar, _ = bss_eval_sources(
            #     np.asarray(gts_wav),
            #     np.asarray(preds_wav),
            #     False)
            #
            # # pesq & stoi
            # pesqs = []
            # stois = []
            # for gt_wav, pred_wav in zip(np.asarray(gts_wav), np.asarray(preds_wav)):
            #     pesqs.append(pesq(args.audRate, gt_wav, pred_wav, 'nb'))
            #     stois.append(stoi(gt_wav, pred_wav, args.audRate, extended=False))
            #
            # sdr_meter.update(sdr.mean())
            # sir_meter.update(sir.mean())
            # sar_meter.update(sar.mean())
            # pesq_meter.update(np.array(pesqs).mean())
            # stoi_meter.update(np.array(stois).mean())

    return [gt_wavs,
            pred_wavs,
            sdr_meter.average(),
            sir_meter.average(),
            sar_meter.average(),
            pesq_meter.average(),
            stoi_meter.average()]

# Evaluate model on validation set
def evaluate(netWrapper, loader, history, epoch, args):
    print('Evaluating at {} epochs...'.format(epoch))

    torch.set_grad_enabled(False)
    netWrapper.eval()

    loss_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()
    pesq_meter = AverageMeter()
    stoi_meter = AverageMeter()

    total_gt_audios = []
    total_pred_audios = []

    for i, batch_data in enumerate(loader):
        # forward pass
        err, outputs = netWrapper.forward(batch_data, args)
        err = err.mean()

        loss_meter.update(err.item())
        if not os.path.exists(args.store_path):
            os.makedirs(args.store_path)

        gt_audio, pred_audio, sdr, sir, sar, pesq, stoi = calc_metrics(batch_data, outputs, args)
        total_gt_audios += gt_audio
        total_pred_audios += pred_audio

        sdr_meter.update(sdr)
        sir_meter.update(sir)
        sar_meter.update(sar)
        pesq_meter.update(pesq)
        stoi_meter.update(stoi)

    print(args.store_path )
    np.save(args.store_path + '/gt' + str(epoch) + '.npy', total_gt_audios)
    np.save(args.store_path + '/pred' + str(epoch) + '.npy', total_pred_audios)

    log = '[Eval Summary] Epoch: {}, Loss: {:.4f}, \
          SDR: {:.4f}, SIR: {:.4f}, SAR: {:.4f}, PESQ: {:.4f}, STOI: {:.4f}' \
          .format(epoch,
                  loss_meter.average(),
                  sdr_meter.average(),
                  sir_meter.average(),
                  sar_meter.average(),
                  pesq_meter.average(),
                  stoi_meter.average())
    print(log)

    with open(args.store_path + '/evaluate-log.txt', 'a') as f:
        f.write(log + '\n')

    history['val']['epoch'].append(epoch)
    history['val']['err'].append(loss_meter.average())
    history['val']['sdr'].append(sdr_meter.average())
    history['val']['sir'].append(sir_meter.average())
    history['val']['sar'].append(sar_meter.average())
    history['val']['pesq'].append(pesq_meter.average())
    history['val']['stoi'].append(stoi_meter.average())

# Create optimizer
def create_optimizer(nets, args, checkpoint):
    (net_sound, net_frame, net_sign) = nets
    param_groups = [{'params': net_sound.parameters(), 'lr': args.lr_sound},
                    {'params': net_frame.parameters(), 'lr': args.lr_frame},
                    {'params': net_sign.parameters(), 'lr': args.lr_sign}
                   ]
    optimizer = torch.optim.SGD(param_groups, momentum=args.beta1, weight_decay=args.weight_decay)
    if args.mode == 'train' and checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    return optimizer

# Adjust learning rate
def adjust_learning_rate(optimizer, args):
    args.lr_sound *= 0.1
    args.lr_frame *= 0.1
    args.lr_sign *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1
        print(param_group['lr'])

# Train one epoch
def train(netWrapper, loader, optimizer, history, epoch, args):
    torch.set_grad_enabled(True)
    netWrapper.train()
    loss_meter = AverageMeter()

    for i, batch_data in enumerate(loader):
        netWrapper.zero_grad()
        err, _ = netWrapper.forward(batch_data, args)
        err = err.mean()
        loss_meter.update(err.item())
        err.backward()
        optimizer.step()

    log = 'Epoch: [{}], ' \
          'lr_sound: {}, lr_frame: {}, lr_sign: {},' \
          'loss: {:.4f}' \
          .format(epoch,
                  args.lr_sound,
                  args.lr_frame,
                  args.lr_sign,
                  loss_meter.average())
    print(log)
    with open(args.store_path + '/train-log.txt', 'a') as f:
        f.write(log + '\n')
    history['train']['epoch'].append(epoch)
    history['train']['err'].append(loss_meter.average())

# @profile
def main(args):
    # Dataset and Loader
    dataset_train = MUSICMixDataset(
        args.list_train, args, split='train')
    dataset_val = MUSICMixDataset(
        args.list_val, args, max_sample=args.num_val, split='val')
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        drop_last=False)
    args.epoch_iters = len(dataset_train) // args.batch_size
    print('1 Epoch = {} iters'.format(args.epoch_iters))

    checkpoint = None
    latest_checkpoint = '{}/checkpoint_{}'.format(args.store_path, 'latest.pth')
    best_checkpoint = '{}/checkpoint_{}'.format(args.store_path, 'best.pth')

    if args.mode == 'train':
        if os.path.isfile(latest_checkpoint):
            checkpoint = torch.load(latest_checkpoint)
            args.start_epoch =  checkpoint['epoch']
            print("*** Loading checkpoint from {} epoch during training. ***".format(args.start_epoch))
    elif args.mode == 'eval':
        if os.path.isfile(best_checkpoint):
            checkpoint = torch.load(best_checkpoint)
            print("*** Loading checkpoint from best checkpoint. ***".format(args.start_epoch))

    # Network Builders
    builder = ModelBuilder()
    net_sound = builder.build_sound(
        arch=args.arch_sound,
        )
    net_frame = builder.build_frame(
        arch=args.arch_frame,
        fc_dim=args.num_channels,
        pool_type=args.img_pool,
        pretrained=args.pretrained_frame
        )
    net_sign = builder.build_sign(
        arch=args.arch_sign,
        fc_dim=args.num_channels,
        pool_type=args.img_pool,
        pretrained=args.pretrained_sign
        )
    crit = builder.build_criterion(arch=args.loss)

    if checkpoint is not None:
            net_sound.load_state_dict(checkpoint['net_sound'])
            net_frame.load_state_dict(checkpoint['net_frame'])
            net_sign.load_state_dict(checkpoint['net_sign'])

    nets = (net_sound, net_frame, net_sign)
    netWrapper = NetWrapper(nets, crit)
    netWrapper = torch.nn.DataParallel(netWrapper, device_ids=range(args.num_gpus))
    netWrapper.to(args.device)

    # Set up optimizer
    optimizer = create_optimizer(nets, args, checkpoint)

    # History of peroformance
    history = {
        'train': {'epoch': [], 'err': []},
        'val': {'epoch': [], 'err': [], 'sdr': [], 'sir': [], 'sar': [], 'pesq': [], 'stoi': []}
    }
    if args.mode == 'train' and checkpoint is not None:
        history = checkpoint['history']

    # Eval mode
    # torch.cuda.synchronize()
    # tic = time.perf_counter()
    evaluate(netWrapper, loader_val, history, 0, args)
    # torch.cuda.synchronize()
    # epoch_time = time.perf_counter() - tic
    # print("The time of evaluate one epoch is {} seconds.".format(epoch_time))
    if args.mode == 'eval':
        print('Evaluating Done!')
        return

    # Training loop
    for epoch in range(args.start_epoch+1, args.num_epoch + 1):

        # torch.cuda.synchronize()
        # tic = time.perf_counter()
        train(netWrapper, loader_train, optimizer, history, epoch, args)
        # torch.cuda.synchronize()
        # epoch_time = time.perf_counter() - tic
        # print("The time of evaluate {} epoch is {} seconds.".format(epoch, epoch_time))

        # Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            evaluate(netWrapper, loader_val, history, epoch, args)
            # Save checkpoint
            save_checkpoint(nets, optimizer, history, epoch, args)

        # drop learning rate
        if epoch in args.lr_steps:
            adjust_learning_rate(optimizer, args)

    print('Training Done!')

if __name__ == '__main__':

    # arguments
    parser = ArgParser()
    args = parser.parse_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.device = torch.device("cuda")
    args.best_err = float("inf") # initialize best error with a big number

    args.store_path = args.store_path + '-' + args.arch_sound + '-' + args.arch_frame + '-' + args.arch_sign + '-' + args.fusion_type

    if args.new_train:
        makedirs(args.store_path, remove=True)
    else:
        if not os.path.exists(args.store_path):
            makedirs(args.store_path)

    with open(args.store_path + r'/arguments.txt', "w") as f:
        print("Input arguments:")
        f.write("Input arguments:" + "\n")
        for key, val in vars(args).items():
            print("{:16} {}".format(key, val))
            f.write("{:16} {}".format(key, val) + "\n")

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(args)
