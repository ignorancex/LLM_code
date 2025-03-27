r""" Matcher testing code for one-shot segmentation """
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

import sys
sys.path.append('./')

from evaluation_util.common.logger import Logger, AverageMeter
from evaluation_util.common.vis import Visualizer
from evaluation_util.common.evaluation import Evaluator
from evaluation_util.common import utils
from evaluation_util.data.dataset import FSSDataset
# from evaluation_util.Matcher import build_matcher_oss

# XXX for diffusion
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.schedulers import DDPMScheduler
from transformers import CLIPVisionModelWithProjection, CLIPTokenizer

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from diffews.models.unet_2d_condition import MyUNet2DConditionModel as CustomUNet2DConditionModel

from marigold.util.scheduler_customized import DDIMSchedulerCustomized as DDIMScheduler
from diffews.marigold_pipeline_rgb_latent_noise import MarigoldPipelineRGBLatentNoise as MarigoldPipeline


import random
random.seed(0)

utils.fix_randseed(0)


# def test(matcher, dataloader, args=None):
#     r""" Test Matcher """

#     # Freeze randomness during testing for reproducibility
#     # Follow HSNet
#     utils.fix_randseed(0)
#     average_meter = AverageMeter(dataloader.dataset)

#     for idx, batch in enumerate(dataloader):

#         batch = utils.to_cuda(batch)
#         query_img, query_mask, support_imgs, support_masks = \
#             batch['query_img'], batch['query_mask'], \
#             batch['support_imgs'], batch['support_masks']

#         # 1. Matcher prepare references and target
#         matcher.set_reference(support_imgs, support_masks)
#         matcher.set_target(query_img)

#         # 2. Predict mask of target
#         pred_mask = matcher.predict()
#         matcher.clear()

#         assert pred_mask.size() == batch['query_mask'].size(), \
#             'pred {} ori {}'.format(pred_mask.size(), batch['query_mask'].size())

#         # 3. Evaluate prediction
#         area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
#         average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
#         average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

#         # Visualize predictions
#         if Visualizer.visualize:
#             Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
#                                                   batch['query_img'], batch['query_mask'],
#                                                   pred_mask, batch['class_id'], idx,
#                                                   area_inter[1].float() / area_union[1].float())

#     # Write evaluation results
#     average_meter.write_result('Test', 0)
#     miou, fb_iou, _ = average_meter.compute_iou()

#     return miou, fb_iou


def test_diffusion(pipe, dataloader, args=None):
    r""" Test Diffusion """

    # Freeze randomness during testing for reproducibility
    # Follow HSNet
    # utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        batch = utils.to_cuda(batch)
        query_img, query_mask, support_imgs, support_masks = \
            batch['query_img'], batch['query_mask'], \
            batch['support_imgs'], batch['support_masks']
        
        # XXX transform support_masks from [b, nshot, h, w] to [b, nshot, 3, h, w], and transform from [0, 1] to [-1, 1]
        support_masks = support_masks.unsqueeze(2).repeat(1, 1, 3, 1, 1) * 2 - 1
        
        # XXX for few shot, transform nshot to batch, which is from [b, nshot, c, h, w] to [b * nshot, c, h, w]
        support_imgs = support_imgs.reshape(-1, support_imgs.shape[-3], support_imgs.shape[-2], support_imgs.shape[-1])
        support_masks = support_masks.reshape(-1, support_masks.shape[-3], support_masks.shape[-2], support_masks.shape[-1])

        input_images_torch = [
            support_imgs,
            query_img,
            support_masks,
        ]

        # Predict mask of target
        pipe_out = pipe(
            input_images_torch,
            denoising_steps=args.denoise_steps,
            ensemble_size=args.ensemble_size,
            processing_res=args.img_size,
            batch_size=args.bsz,
            show_progress_bar=False,
            mode='seg',
            rgb_paths=batch['rgb_path'],
            seed=0,
        )

        pred_mask = pipe_out.seg_colored
        raw_mask = pred_mask
        # pred_mask from PIL to torch tensor
        pred_mask = transforms.functional.to_tensor(pred_mask)
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask[None]  # to [b, c, h, w]
        if args.r_threshold > 0:
            pred_mask_ori = pred_mask
            dynamic_thres = pred_mask.max()*args.r_threshold
            pred_mask = (pred_mask.mean(dim=1) > dynamic_thres).to(batch['query_mask'])
        # transform pred_mask to binary mask
        if args.threshold>0:
            pred_mask = (pred_mask.mean(dim=1) > args.threshold).to(batch['query_mask'])
        
        assert pred_mask.size() == batch['query_mask'].size(), \
            'pred {} ori {}'.format(pred_mask.size(), batch['query_mask'].size())

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'], # (batch['support_masks'][:, :, 0]+1)/2,
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask, batch['class_id'], idx,
                                                  area_inter[1].float() / area_union[1].float())
            import torchshow
            torchshow.save(raw_mask, f"vis/{idx}_raw_mask.png")

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou, _ = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Matcher Pytorch Implementation for One-shot Segmentation')

    # Dataset parameters
    parser.add_argument('--datapath', type=str, default='datasets')
    parser.add_argument('--benchmark', type=str, default='coco',
                        choices=['fss', 'coco', 'pascal', 'lvis', 'paco_part', 'pascal_part'])
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--img-size', type=int, default=518)
    parser.add_argument('--use_original_imgsize', action='store_true')
    parser.add_argument('--log-root', type=str, default='output/debug')
    parser.add_argument('--visualize', type=int, default=0)
    parser.add_argument('--vis_path', type=str, default='output/debug/vis')

    # DINOv2 and SAM parameters
    parser.add_argument('--dinov2-size', type=str, default="vit_large")
    parser.add_argument('--sam-size', type=str, default="vit_h")
    parser.add_argument('--dinov2-weights', type=str, default="models/dinov2_vitl14_pretrain.pth")
    parser.add_argument('--sam-weights', type=str, default="models/sam_vit_h_4b8939.pth")
    parser.add_argument('--use_semantic_sam', action='store_true', help='use semantic-sam')
    parser.add_argument('--semantic-sam-weights', type=str, default="models/swint_only_sam_many2many.pth")
    parser.add_argument('--points_per_side', type=int, default=64)
    parser.add_argument('--pred_iou_thresh', type=float, default=0.88)
    parser.add_argument('--sel_stability_score_thresh', type=float, default=0.0)
    parser.add_argument('--stability_score_thresh', type=float, default=0.95)
    parser.add_argument('--iou_filter', type=float, default=0.0)
    parser.add_argument('--box_nms_thresh', type=float, default=1.0)
    parser.add_argument('--output_layer', type=int, default=3)
    parser.add_argument('--dense_multimask_output', type=int, default=0)
    parser.add_argument('--use_dense_mask', type=int, default=0)
    parser.add_argument('--multimask_output', type=int, default=0)

    # Matcher parameters
    parser.add_argument('--num_centers', type=int, default=8, help='K centers for kmeans')
    parser.add_argument('--use_box', action='store_true', help='use box as an extra prompt for sam')
    parser.add_argument('--use_points_or_centers', action='store_true', help='points:T, center: F')
    parser.add_argument('--sample-range', type=str, default="(4,6)", help='sample points number range')
    parser.add_argument('--max_sample_iterations', type=int, default=30)
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--beta', type=float, default=0.)
    parser.add_argument('--exp', type=float, default=0.)
    parser.add_argument('--emd_filter', type=float, default=0.0, help='use emd_filter')
    parser.add_argument('--purity_filter', type=float, default=0.0, help='use purity_filter')
    parser.add_argument('--coverage_filter', type=float, default=0.0, help='use coverage_filter')
    parser.add_argument('--use_score_filter', action='store_true')
    parser.add_argument('--deep_score_norm_filter', type=float, default=0.1)
    parser.add_argument('--deep_score_filter', type=float, default=0.33)
    parser.add_argument('--topk_scores_threshold', type=float, default=0.7)
    parser.add_argument('--num_merging_mask', type=int, default=10, help='topk masks for merging')

    # XXX Diffusion parameters
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="Bingxin/Marigold",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--scheduler_load_path",
        type=str,
        default=None,
        help="Scheduler load path.",
    )
    parser.add_argument(
        "--unet_ckpt_path",
        type=str,
        default=None,
        # default="./logs/logs_sd21_seg_pretrain/checkpoint-10000",
        help="Checkpoint path for unet.",
    )
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=10,
        help="Diffusion denoising steps, more stepts results in higher accuracy but slower inference speed.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="the threshold to transform the output to mask",
    )
    parser.add_argument(
        "--r_threshold",
        type=float,
        default=0.0,
        help="relative_thres",
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=10,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    # add train_time_steps
    parser.add_argument(
        "--test_timestep",
        type=int,
        default=1,
        help="timesteps for testing",
    )
    



    args = parser.parse_args()
    args.sample_range = eval(args.sample_range)

    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    Logger.initialize(args, root=args.log_root)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    # Model initialization
    # if not args.use_semantic_sam:
    #     matcher = build_matcher_oss(args)
    # else:
    #     from matcher.Matcher_SemanticSAM import build_matcher_oss as build_matcher_semantic_sam_oss
    #     matcher = build_matcher_semantic_sam_oss(args)


    # XXX Diffusion model initialization

    # logging_dir = "logs"
    # logging_dir = os.path.join(args.log_root, logging_dir)
    # accelerator_project_config = ProjectConfiguration(project_dir=args.log_root, logging_dir=logging_dir)
    # accelerator = Accelerator(
    #     gradient_accumulation_steps=1,
    #     mixed_precision="fp16",
    #     log_with="tensorboard",
    #     project_config=accelerator_project_config,
    # )

    if args.half_precision:
        dtype = torch.float16
        logger.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32

    if args.unet_ckpt_path is not None:
        unet = CustomUNet2DConditionModel.from_pretrained(
            args.unet_ckpt_path, subfolder="unet", revision=args.non_ema_revision
        )
    else:
        unet = CustomUNet2DConditionModel.from_pretrained(
            args.checkpoint, subfolder="unet", revision=args.non_ema_revision
        )
    
    vae = AutoencoderKL.from_pretrained(
        args.checkpoint, subfolder="vae",
    )

    tokenizer = CLIPTokenizer.from_pretrained(
        args.checkpoint, subfolder="tokenizer",
    )

    marigold_params_ckpt = dict(
        torch_dtype=dtype,
        unet=unet,
        vae=vae,
        controlnet=None,
        text_embeds=None, # TODO: change it for sdxl model.
        image_projector=None, # TODO: change it for using image projector.
        customized_head=None,
        image_encoder=None
    )

    if args.scheduler_load_path:
        marigold_params_ckpt['scheduler'] = DDIMScheduler.from_pretrained(args.scheduler_load_path, subfolder="scheduler")

    pipe = MarigoldPipeline.from_pretrained(args.checkpoint, **marigold_params_ckpt)

    pipe = pipe.to(device)
    # pipe.set_progress_bar_config(disable=True)
    pipe.test_timestep = args.test_timestep
    try:
        import xformers
        pipe.enable_xformers_memory_efficient_attention()
    except:
        print("No xformers!")
        raise


    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize)

    # Dataset initialization
    FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)
    # dataloader_test = accelerator.prepare(dataloader_test)

    # # Test Matcher
    # with torch.no_grad():
    #     test_miou, test_fb_iou = test(matcher, dataloader_test, args=args)
    # Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    # Logger.info('==================== Finished Testing ====================')

    # XXX Test diffusion
    with torch.no_grad():
        test_miou, test_fb_iou = test_diffusion(pipe, dataloader_test, args=args)
    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')