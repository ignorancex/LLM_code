import os
import time
import random
import shutil
import torch
import numpy as np
import argparse
import logging

from config import cfg
from dataloader import MS3Dataset
from torchvggish import vggish

from utils import pyutils
from utils.utility import logger, mask_iou, Eval_Fmeasure, save_mask
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

    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--weights", default='./train_logs/MS3_res_Model_FinalModel/checkpoints/MS3_best.pth', type=str)
    parser.add_argument("--save_pred_mask", action='store_true', default=True, help="save predited masks or not")
    parser.add_argument('--log_dir', default='./test_logs_debug', type=str)

    args = parser.parse_args()

    if args.visual_backbone.lower() == "resnet":
        from model import ResNet_AVSModel as AVSModel

        print('==> Use ResNet50 as the visual backbone...')
    elif args.visual_backbone.lower() == "pvt":
        from model import PVT_AVSModel as AVSModel

        print('==> Use pvt-v2 as the visual backbone...')
    else:
        raise NotImplementedError("only support the resnet50 and pvt-v2")

    # Log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
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
        'test.py', 'config.py', 'dataloader.py', 'loss.py',
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

    # Set logger
    log_path = os.path.join(log_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, 'log.txt'))
    logger = logging.getLogger(__name__)
    logger.info('==> Config: {}'.format(cfg))
    logger.info('==> Arguments: {}'.format(args))
    logger.info('==> Experiment: {}'.format(args.session_name))

    # Model
    model = AVSModel.PCMANet(channel=256, config=cfg)
    model.load_state_dict(torch.load(args.weights))
    model = torch.nn.DataParallel(model).cuda()
    logger.info('Load trained model %s' % args.weights)

    # audio backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device)
    audio_backbone.cuda()
    audio_backbone.eval()

    # Test data
    split = 'test'
    test_dataset = MS3Dataset(split)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.test_batch_size,
                                                  shuffle=False,
                                                  num_workers=args.num_workers,
                                                  pin_memory=True)

    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')

    # Test
    model.eval()
    with torch.no_grad():
        for n_iter, batch_data in enumerate(test_dataloader):
            imgs, audio, mask, video_name_list = batch_data  # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
            if video_name_list[0] != 'Kd42SbTQv2U_11':
                continue

            imgs = imgs.cuda()
            audio = audio.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B * frame, C, H, W)
            mask = mask.view(B * frame, H, W)
            audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])
            with torch.no_grad():
                audio_feature = audio_backbone(audio)

            output, _, _, _, _, V, O = model(imgs, audio_feature)  # [5, 1, 224, 224] = [bs=1 * T=5, 1, 224, 224]
            import cv2
            mask_save_path = os.path.join(log_dir, 'feat', video_name_list[0])
            if not os.path.exists(mask_save_path):
                os.makedirs(mask_save_path)
            for i in range(len(V)):
                v = V[i]
                v = torch.nn.functional.interpolate(v.unsqueeze(0).unsqueeze(0), (112, 112)).squeeze(0).squeeze(0)
                normalized_map = (v - v.min()) / (v.max() - v.min())  # 归一化到 [0, 1]
                normalized_map = (normalized_map * 255).byte().cpu().numpy()
                colored_map = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)  # 使用 'HOT' 映射
                path = os.path.join(mask_save_path, f'v{i}.png')
                cv2.imwrite(path, colored_map)

            for i in range(len(O)):
                o = O[i]
                o = torch.nn.functional.interpolate(o.unsqueeze(0).unsqueeze(0), (112, 112)).squeeze(0).squeeze(0)
                normalized_map = (o - o.min()) / (o.max() - o.min())  # 归一化到 [0, 1]
                normalized_map = (normalized_map * 255).byte().cpu().numpy()
                colored_map = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)  # 使用 'HOT' 映射
                path = os.path.join(mask_save_path, f'o{i}.png')
                cv2.imwrite(path, colored_map)

            # # save m3, m2, m1
            # # resize m3, m2, m1 to 224, 224
            # m = [torch.nn.functional.interpolate(m[i], size=(224, 224), mode='nearest') for i in range(3)]
            # from PIL import Image
            #
            # if args.save_pred_mask:
            #     mask_save_path = os.path.join(log_dir, 'm3')
            #     save_mask(m[0].squeeze(1), mask_save_path, video_name_list)
            #     mask_save_path = os.path.join(log_dir, 'm2')
            #     save_mask(m[1].squeeze(1), mask_save_path, video_name_list)
            #     mask_save_path = os.path.join(log_dir, 'm1')
            #     save_mask(m[2].squeeze(1), mask_save_path, video_name_list)
            # # save c3, c2, c1
            # for i in range(3):
            #     # resize c3, c2, c1 to 224, 224
            #     c[i] = torch.nn.functional.interpolate(c[i], size=(224, 224), mode='nearest')
            #     # 保留c3, c2, c1中0.01-0.99的部分，以外的部分置为0
            #     # mask = (c[i] > 0.01) * (c[i] < 0.99)
            #     c[i] = (1 - abs(c[i] - 0.5)) * 2 - 1
            #     # c[i] = c[i] * mask
            #
            # import cv2
            #
            # cmap = cv2.COLORMAP_PINK
            #
            # if args.save_pred_mask:
            #     mask_save_path = os.path.join(log_dir, 'c3')
            #     c3 = c[0].squeeze(1)
            #     c3 = c3.view(-1, 5, c3.shape[-2], c3.shape[-1])
            #     c3 *= 255
            #     bs = c3.shape[0]
            #
            #     for idx in range(bs):
            #         video_name = video_name_list[idx]
            #         mask_save_path = os.path.join(mask_save_path, video_name)
            #         if not os.path.exists(mask_save_path):
            #             os.makedirs(mask_save_path, exist_ok=True)
            #         one_video_masks = c3[idx]  # [5, 1, 224, 224]
            #         for video_id in range(len(one_video_masks)):
            #             one_mask = one_video_masks[video_id]
            #             one_mask = one_mask.cpu().data.numpy()
            #             one_mask = cv2.applyColorMap((one_mask).astype(np.uint8), cmap)
            #             output_name = "%s_%d.png" % (video_name, video_id)
            #
            #             # im = Image.fromarray(one_mask).convert('P')
            #             # im.save(os.path.join(mask_save_path, output_name), format='PNG')
            #             # cv2.imwrite(os.path.join(mask_save_path, output_name), one_mask)
            #             cv2.imwrite(os.path.join(mask_save_path, output_name), one_mask)
            #
            #     mask_save_path = os.path.join(log_dir, 'c2')
            #     c2 = c[1].squeeze(1)
            #     c2 = c2.view(-1, 5, c2.shape[-2], c2.shape[-1])
            #     c2 *= 255
            #     bs = c2.shape[0]
            #
            #     for idx in range(bs):
            #         video_name = video_name_list[idx]
            #         mask_save_path = os.path.join(mask_save_path, video_name)
            #         if not os.path.exists(mask_save_path):
            #             os.makedirs(mask_save_path, exist_ok=True)
            #         one_video_masks = c2[idx]
            #         for video_id in range(len(one_video_masks)):
            #             one_mask = one_video_masks[video_id]
            #             one_mask = one_mask.cpu().data.numpy()
            #             one_mask = cv2.applyColorMap((one_mask).astype(np.uint8), cmap)
            #             output_name = "%s_%d.png" % (video_name, video_id)
            #             # im = Image.fromarray(one_mask).convert('P')
            #             # im.save(os.path.join(mask_save_path, output_name), format='PNG')
            #             cv2.imwrite(os.path.join(mask_save_path, output_name), one_mask)
            #
            #     mask_save_path = os.path.join(log_dir, 'c1')
            #     c1 = c[2].squeeze(1)
            #     c1 = c1.view(-1, 5, c1.shape[-2], c1.shape[-1])
            #     c1 *= 255
            #     bs = c1.shape[0]
            #
            #     for idx in range(bs):
            #         video_name = video_name_list[idx]
            #         mask_save_path = os.path.join(mask_save_path, video_name)
            #         if not os.path.exists(mask_save_path):
            #             os.makedirs(mask_save_path, exist_ok=True)
            #         one_video_masks = c1[idx]
            #         for video_id in range(len(one_video_masks)):
            #             one_mask = one_video_masks[video_id]
            #             one_mask = one_mask.cpu().data.numpy()
            #             one_mask = cv2.applyColorMap((one_mask).astype(np.uint8), cmap)
            #             output_name = "%s_%d.png" % (video_name, video_id)
            #             # im = Image.fromarray(one_mask).convert('P')
            #             # im.save(os.path.join(mask_save_path, output_name), format='PNG')
            #             cv2.imwrite(os.path.join(mask_save_path, output_name), one_mask)
            if args.save_pred_mask:
                mask_save_path = os.path.join(log_dir, 'pred_masks')
                save_mask(output.squeeze(1), mask_save_path, video_name_list)

        #     miou = mask_iou(output.squeeze(1), mask)
        #     avg_meter_miou.add({'miou': miou})
        #     F_score = Eval_Fmeasure(output.squeeze(1), mask, log_dir)
        #     avg_meter_F.add({'F_score': F_score})
        #     print('n_iter: {}, iou: {}, F_score: {}'.format(n_iter, miou, F_score))
        #
        # miou = (avg_meter_miou.pop('miou'))
        # F_score = (avg_meter_F.pop('F_score'))
        # print('test miou:', miou.item())
        # print('test F_score:', F_score)
        # logger.info('test miou: {}, F_score: {}'.format(miou.item(), F_score))
