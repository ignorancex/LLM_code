import torch as th
import numpy as np

from extract_feature.slowfast.extract_feature.video_loader import (
    VideoLoader, clip_iterator, pack_pathway_output)
from torch.utils.data import DataLoader
import argparse
from extract_feature.slowfast.extract_feature.model import build_model
from extract_feature.slowfast.extract_feature.preprocessing import Preprocessing, Normalize
from extract_feature.slowfast.extract_feature.random_sequence_shuffler import RandomSequenceSampler
from extract_feature.slowfast.slowfast.config.defaults import get_cfg
import extract_feature.slowfast.slowfast.utils.checkpoint as cu
from tqdm import tqdm
from extract_feature.slowfast.extract_feature.prefetch_loader import PrefetchLoader
import sys
import os
import time
from extract_feature.slowfast.extract_feature.yuv_reader import YuvRgbConverter
from lavis.models import load_model_and_preprocess

FEATURE_LENGTH = 2304
YUV2RGB = YuvRgbConverter()

device = th.device("cuda" if th.cuda.is_available() else "cpu")

from run_on_video.data_utils import ClipFeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Easy video feature extractor')

    parser.add_argument(
        '--csv',
        type=str, default='extract_feature/tmp.csv',
        help='input csv with video input path')
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="extract_feature/slowfast/configs/Kinetics/c2/extract_SLOWFAST_8x8_R50.yaml",
        type=str,
    )
    parser.add_argument(
        '--batch_size', type=int, default=4, help='batch size')
    parser.add_argument(
        '--half_precision', type=int, default=1,
        help='output half precision float')
    parser.add_argument(
        '--dataflow', action='store_true',
        help='enable dataflow')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='allow overwrite output files')
    parser.add_argument(
        '--num_decoding_thread', type=int, default=0,
        help='Num parallel thread for video decoding')
    parser.add_argument(
        '--target_framerate', type=int, default=30,
        help='decoding frame per second')
    parser.add_argument(
        '--clip_len', type=str, default='3/2',
        help='decoding length of clip (in seconds)')
    parser.add_argument(
        '--min_num_features', type=int, default=1,
        help='minimum number of features')
    parser.add_argument(
        '--pix_fmt', type=str, default="rgb24", choices=["rgb24", "yuv420p"],
        help='decode video into RGB format')
    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


@th.no_grad()
def extract_features(video_loader, video_processor, sf_model, preprocess,
                     blip_model, vis_processors, clip_feature_extractor, cfg, args, failed_log, n_dataset):
    """
    For classification:
    Perform mutli-view testing that uniformly samples
        N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        video_loader (loader): video testing loader.
        sf_model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    # Enable eval mode.
    sf_model.eval()
    norm = Normalize(
        mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    totatl_num_frames = 0

    total_time = 0
    pbar = tqdm(total=n_dataset)

    for _, data in enumerate(video_loader):
        video = data['video']
        video_shape_len = len(video.shape)
        input_file = data['input']
        sf_output_file = data['sf_output']
        clip_output_file = data['clip_output']
        blip_output_file = data['blip_output']
        if isinstance(input_file, (list,)):
            input_file = input_file[0]
            sf_output_file = sf_output_file[0]
            clip_output_file = clip_output_file[0]
            blip_output_file = blip_output_file[0]
        if video_shape_len == 6:
            video = video.squeeze(0)
        video_shape_len = len(video.shape)
        if video_shape_len == 5:
            n_chunk = len(video)
            print("\nProcessing", input_file, "n_chunk", n_chunk)

            sf_features = th.cuda.HalfTensor(
                n_chunk, FEATURE_LENGTH).fill_(0)
            clip_loader = PrefetchLoader(clip_iterator(video, args.batch_size))

            for _, (min_ind, max_ind, fast_clip) in enumerate(clip_loader):
                # B T H W C
                fast_clip = fast_clip.float()
                if args.pix_fmt == "yuv420p":
                    fast_clip = YUV2RGB(fast_clip)
                fast_clip = fast_clip.permute(0, 4, 1, 2, 3)
                # -> B C T H W
                fast_clip = fast_clip / 255.
                fast_clip = norm(fast_clip)
                inputs = pack_pathway_output(cfg, fast_clip)
                # Perform the forward pass.
                th.cuda.synchronize()
                start_time = time.time()
                batch_features = sf_model(inputs)
                th.cuda.synchronize()
                end_time = time.time()
                total_time += end_time - start_time
                sf_features[min_ind:max_ind] = batch_features.half()

            sf_features = sf_features.cpu().numpy().astype('float16')
            totatl_num_frames += sf_features.shape[0]

            print("SF Feature Len: ", len(sf_features))
            save_video_feature(sf_features, sf_output_file)

            clip_feature = clip_feature_extractor.encode_video(input_file)
            clip_feature = clip_feature.cpu().numpy().astype('float16')
            print("Clip Feature Len: ", len(clip_feature))
            save_video_feature(clip_feature, clip_output_file)

            blip_feature = encode_video_blip(video_processor, input_file, vis_processors, model=blip_model)
            blip_feature = blip_feature.cpu().numpy().astype('float16')
            print("Blip Feature Len: ", len(blip_feature))
            save_video_feature(blip_feature, blip_output_file)

        elif os.path.isfile(sf_output_file):
            print(f'\nVideo {input_file} already processed.')
        elif not os.path.isfile(input_file):
            failed_log.write(f'\n{input_file}, does not exist.\n')
        else:
            failed_log.write(f'\n{input_file}, failed at ffprobe.\n')
        pbar.update(1)

    print(f"Total number of frames: {totatl_num_frames}")
    print(f"Model inference time: {total_time}")


@th.no_grad()
def encode_video_blip(video_loader, video_path: str, vis_processors, model):
    video_frames, _ = video_loader.read_raw_image_from_video_file(video_path)  # (T, H, W, 3)
    n_frames = len(video_frames)
    video_features = []
    for i in range(n_frames):
        image = vis_processors["eval"](video_frames[i]).unsqueeze(0).to(device)
        sample = {"image": image}
        features_image = model.extract_features(sample, mode="image")
        video_features.append(features_image.image_embeds[:, 0, :])
    video_features = th.cat(video_features, dim=0)
    return video_features  # (T=#frames, d) torch tensor


def save_video_feature(result, result_save_path):
    dirname = os.path.dirname(result_save_path)
    if not os.path.exists(dirname):
        print(f"Output directory {dirname} does not exists" +
              ", creating...")
        os.makedirs(dirname)
    try:
        np.savez_compressed(result_save_path, features=result)
    except Exception as e:
        print(e)
        print(result_save_path)


def main():
    """
    Main function to extract features.
    """
    opts = parse_args()
    cfg = load_config(opts)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    th.manual_seed(cfg.RNG_SEED)
    failed_log = open(opts.csv.split(".csv")[0] + "_failed.txt", "w")
    assert opts.target_framerate % opts.min_num_features == 0

    preprocess = Preprocessing(
        "3d", cfg, target_fps=opts.target_framerate,
        size=224, clip_len=opts.clip_len, padding_mode='tile',
        min_num_clips=opts.min_num_features)
    dataset = VideoLoader(
        opts.csv,
        preprocess,
        framerate=opts.target_framerate,
        size=224,
        centercrop=True,
        pix_fmt=opts.pix_fmt,
        overwrite=opts.overwrite
    )
    n_dataset = len(dataset)
    sampler = RandomSequenceSampler(n_dataset, 10)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opts.num_decoding_thread,
        sampler=sampler if n_dataset > 10 else None,
    )

    sf_model = build_model(cfg)

    # model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor",
    #                                                                   model_type="pretrain", is_eval=True,
    #                                                                   device=device)  # Blip2 featrures

    blip_model, vis_processors, _ = load_model_and_preprocess(name="blip_feature_extractor",
                                                              model_type="base", is_eval=True,
                                                              device=device)  # Blip featrures

    clip_feature_extractor = ClipFeatureExtractor(
        framerate=(1.0/int(opts.clip_len)), size=224, centercrop=True,
        model_name_or_path="ViT-B/32", device="cuda"
    )

    extract_features(loader, clip_feature_extractor.video_loader, sf_model, preprocess, blip_model, vis_processors,
                     clip_feature_extractor, cfg, opts, failed_log, n_dataset)


if __name__ == "__main__":
    main()
