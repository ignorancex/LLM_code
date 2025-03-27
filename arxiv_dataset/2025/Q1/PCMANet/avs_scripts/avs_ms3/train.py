import os
import time
import random
import shutil
import torch
import numpy as np
import argparse
import logging
from torch.utils.tensorboard import SummaryWriter

from config import cfg
from dataloader import MS3Dataset
from torchvggish import vggish
from loss import BCEIOU_loss

from utils import pyutils
from utils.utility import logger, mask_iou
from utils.system import setup_logging
import pdb


class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()
        self.audio_backbone = vggish.VGGish(cfg, device)

    def forward(self, audio):
        audio_fea = self.audio_backbone(audio)
        return audio_fea


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default="MS3", type=str, help="the MS3 setting")
    parser.add_argument("--visual_backbone", default="resnet", type=str,
                        help="use resnet50 or pvt-v2 as the visual backbone")

    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--max_epoches", default=30, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--num_workers", default=8, type=int)

    parser.add_argument("--resume_from", type=str, default='', help='the checkpoint file to resume from')
    parser.add_argument('--log_dir', default='./train_logs', type=str)

    args = parser.parse_args()

    if args.visual_backbone.lower() == "resnet":
        from model import ResNet_AVSModel as AVSModel

        print('==> Use ResNet50 as the visual backbone...')
    elif args.visual_backbone.lower() == "pvt":
        from model import PVT_AVSModel as AVSModel

        print('==> Use pvt-v2 as the visual backbone...')
    else:
        raise NotImplementedError("only support the resnet50 and pvt-v2")

    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    # Log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    # Logs
    prefix = args.session_name
    # log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
    log_dir = os.path.join(args.log_dir, f'{prefix}_{args.visual_backbone.lower()[:3]}_Model_Final')
    args.log_dir = log_dir

    # Save scripts
    script_path = os.path.join(log_dir, 'scripts')
    if not os.path.exists(script_path):
        os.makedirs(script_path, exist_ok=True)

    scripts_to_save = [
        'train.py', 'config.py', 'dataloader.py', 'loss.py',
        './model/ResNet_AVSModel.py' if args.visual_backbone.lower() == 'resnet' else './model/PVT_AVSModel.py',
        './model/resnet.py' if args.visual_backbone.lower() == 'resnet' else './model/pvt.py',
        './model/ca.py',
    ]
    for script in scripts_to_save:
        dst_path = os.path.join(script_path, script)
        try:
            shutil.copy(script, dst_path)
        except IOError:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(script, dst_path)

    # Checkpoints directory
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    # Set logger
    log_path = os.path.join(log_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, 'log.txt'))
    logger = logging.getLogger(__name__)
    logger.info('==> Config: {}'.format(cfg))
    logger.info('==> Arguments: {}'.format(args))
    logger.info('==> Experiment: {} | Backbone: {}'.format(args.session_name, args.visual_backbone))

    # Model
    model = AVSModel.PCMANet(channel=256, config=cfg)
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    logger.info("==> Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))

    # load pretrained S4 model
    if args.resume_from:
        if os.path.isfile(args.resume_from):
            try:
                model.load_state_dict(torch.load(args.resume_from))
                logger.info('==> Load pretrained model from %s' % args.resume_from)
            except:
                logger.info('==> Load pretrained model failed')
        else:
            logger.info('==> No checkpoint found at %s' % args.resume_from)
            raise FileNotFoundError('No checkpoint found at %s' % args.resume_from)

    # video backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device)
    audio_backbone.cuda()
    audio_backbone.eval()

    # Data
    train_dataset = MS3Dataset('train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.train_batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True)
    max_step = (len(train_dataset) // args.train_batch_size) * args.max_epoches

    val_dataset = MS3Dataset('val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.val_batch_size,
                                                 shuffle=False,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True)

    # Optimizer
    model_params = model.parameters()
    optimizer = torch.optim.Adam(model_params, lr=args.lr)
    avg_meter_total_loss = pyutils.AverageMeter('total_loss')
    avg_meter_miou = pyutils.AverageMeter('miou')
    writer = SummaryWriter(log_dir=os.path.join(log_dir, 'figure_log'))

    # Train
    best_epoch = 0
    global_step = 0
    miou_list = []
    max_miou = 0
    for epoch in range(args.max_epoches):
        for n_iter, batch_data in enumerate(train_dataloader):
            imgs, audio, mask, _ = batch_data  # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5 or 1, 1, 224, 224]

            imgs = imgs.cuda()
            audio = audio.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B * frame, C, H, W)
            mask_num = 5
            mask = mask.view(B * mask_num, 1, H, W)
            audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])  # [bs*5, 1, 96, 64]
            with torch.no_grad():
                audio_feature = audio_backbone(audio)  # [bs*5, 128]

            output1, output2, output3, output4, per = model(imgs, audio_feature)  # [bs*5, 1, 224, 224]
            loss = 1 * BCEIOU_loss(output1, mask) + 1 * BCEIOU_loss(output2, mask) + \
                1 * BCEIOU_loss(output3, mask) + 1 * BCEIOU_loss(output4, mask)

            avg_meter_total_loss.add({'total_loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            p4, p3, p2 = per
            writer.add_scalars('Confidence', {'p4': p4, 'p3': p3, 'p2': p2}, global_step)
            writer.add_scalar('Loss', loss.item(), global_step)

            global_step += 1
            if (global_step - 1) % 20 == 0:
                train_log = 'Iter:%5d/%5d, Total_Loss:%.4f, lr: %.4f, p4: %.4f, p3: %.4f, p2: %.4f' % (
                    global_step - 1, max_step, avg_meter_total_loss.pop('total_loss'),
                    optimizer.param_groups[0]['lr'], p4, p3, p2)
                logger.info(train_log)

        # Validation:
        model.eval()
        with torch.no_grad():
            for n_iter, batch_data in enumerate(val_dataloader):
                imgs, audio, mask, _ = batch_data  # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]

                imgs = imgs.cuda()
                audio = audio.cuda()
                mask = mask.cuda()
                B, frame, C, H, W = imgs.shape
                imgs = imgs.view(B * frame, C, H, W)
                mask = mask.view(B * frame, H, W)
                audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])
                with torch.no_grad():
                    audio_feature = audio_backbone(audio)

                output, _, _, _, _ = model(imgs, audio_feature)  # [bs*5, 1, 224, 224]

                miou = mask_iou(output.squeeze(1), mask)
                avg_meter_miou.add({'miou': miou})

            miou = (avg_meter_miou.pop('miou'))
            if miou > max_miou:
                model_save_path = os.path.join(checkpoint_dir, '%s_best.pth' % (args.session_name))
                torch.save(model.module.state_dict(), model_save_path)
                best_epoch = epoch
                logger.info('save best model to %s' % model_save_path)
            if epoch == args.max_epoches - 1:
                model_save_path = os.path.join(checkpoint_dir, '%s_last.pth' % (args.session_name))
                torch.save(model.module.state_dict(), model_save_path)
                logger.info('save last model to %s' % model_save_path)

            miou_list.append(miou)
            max_miou = max(miou_list)

            val_log = 'Epoch: {}, Miou: {}, maxMiou: {}'.format(epoch, miou, max_miou)
            # print(val_log)
            logger.info(val_log)

        model.train()
    logger.info('best val Miou {} at peoch: {}'.format(max_miou, best_epoch))
    writer.close()
