import os
import time
from pprint import pformat

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm

from third_party.opensora.acceleration.parallel_states import set_sequence_parallel_group
from third_party.opensora.registry import build_module
from third_party.opensora.utils.config_utils import parse_configs
from third_party.opensora.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype

from third_party.fairseq.data.data_utils import lengths_to_mask

from v2sflow.registry import DATASETS, MODELS, SCHEDULERS

from einops import rearrange
import numpy as np

def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # == init distributed env ==
    if is_distributed():
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()
        enable_sequence_parallelism = coordinator.world_size > 1
        if enable_sequence_parallelism:
            set_sequence_parallel_group(dist.group.WORLD)
    else:
        coordinator = None
        enable_sequence_parallelism = False
    set_random_seed(seed=cfg.get("seed", 1024))

    # == init logger ==
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 1)
    progress_wrap = tqdm if verbose == 1 else (lambda x: x)

    # ======================================================
    # build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")
    # == build dataset ==
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info("Dataset contains %s samples.", len(dataset))
    # == build dataloader ==
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 1),
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=dataset.collater,
        shuffle=False,
    )

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == build diffusion model ==
    model = (
        build_module(
            cfg.model,
            MODELS,
        )
        .to(device, dtype)
        .eval()
    )
    if cfg.get("load", None) is not None:
        state_dict = torch.load(cfg.load)
        model.load_state_dict(state_dict)
        print(f"checkpoint loaded from {cfg.load}")
        # from colossalai.checkpoint_io import GeneralCheckpointIO
        # checkpoint_io = GeneralCheckpointIO()
        # checkpoint_io.load_model(model, os.path.join(cfg.load, "model"))
        # print(f"checkpoint loaded from {cfg.load}")
    else:
        print(f"use pretrained checkpoint")

    if cfg.get("encoder_cfg", None) is not None:
        encoder = (
            build_module(
                cfg.encoder_cfg,
                MODELS,
            )
            .to(device, dtype)
            .eval()
        )
        if cfg.get("encoder", None) is not None:
            state_dict = torch.load(cfg.encoder)
            encoder.load_state_dict(state_dict)
            print(f"loaded encoder from {cfg.encoder}")
    else:
        encoder = None

    if cfg.get("vocoder", None) is not None and cfg.get("vocoder_cfg", None) is not None:
        import json
        from third_party.fairseq.models.text_to_speech.vocoder import HiFiGANVocoder
        import soundfile as sf
        with open(cfg.vocoder_cfg) as f:
            vocoder_cfg = json.load(f)
        vocoder = HiFiGANVocoder(cfg.vocoder, vocoder_cfg)
        vocoder.to(device)
        sampling_rate = vocoder_cfg["sampling_rate"]
        print(f"loaded vocoder from {cfg.vocoder}")
    else:
        vocoder = None

    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # ======================================================
    # inference
    # ======================================================
    # == Iter over all samples ==
    start_idx = 0
    save_dir = cfg.save_dir
    with torch.no_grad():
        with tqdm(enumerate(dataloader), total=len(dataloader)) as pbar:
            for step, batch in tqdm(enumerate(dataloader)):
                if encoder is not None:
                    assert batch.get("audios", None) is None
                    assert batch.get("contents", None) is None
                    assert batch.get("pitchs", None) is None
                    assert batch.get("speaker", None) is None

                    video = batch["videos"].to(device, dtype)  # [B, T, C]
                    video_length = batch["video_lengths"].to(device)  # [B]
                    video_padding_mask = ~lengths_to_mask(video_length)
                    encoder_out = encoder(video_feat=video, video_padding_mask=video_padding_mask)

                    # dummy input
                    batch["audios"] = torch.zeros(video.size(0), round(video.size(1) * (dataset.audio_fps / dataset.video_fps)) // dataset.audio_stack, 80 * dataset.audio_stack)
                    batch["audio_lengths"] = (video_length.float() * (dataset.audio_fps / dataset.video_fps)).round().long() // dataset.audio_stack

                    if encoder.content_encoder is not None:
                        batch["contents"] = encoder_out["content"].unsqueeze(1)
                        batch["content_lengths"] = (video_length.float() * (dataset.content_fps / dataset.video_fps)).round().long()
                    if encoder.pitch_encoder is not None:
                        batch["pitchs"] = encoder_out["pitch"].unsqueeze(1).repeat_interleave(2, dim=-1) ## 12.5 -> 25
                        batch["pitch_lengths"] = (video_length.float() * (dataset.pitch_fps / dataset.video_fps)).round().long()
                        batch["pitchs"] = batch["pitchs"][:, :, :batch["pitch_lengths"].max()]
                    if encoder.speaker_encoder is not None:
                        batch["speaker"] = encoder_out["speaker"]

                audio = batch["audios"].to(device, dtype)  # [B, T, C]
                audio_length = batch["audio_lengths"].to(device)  # [B]

                if batch.get("contents", None) is not None:
                    content = batch["contents"].to(device, torch.int32)  # [B, 1, T]
                    content_length = batch["content_lengths"].to(device)  # [B]
                else:
                    content = None
                if batch.get("pitchs", None) is not None:
                    pitch = batch["pitchs"].to(device, torch.int32)  # [B, 1, T]
                    pitch_length = batch["pitch_lengths"].to(device)  # [B]
                else:
                    pitch = None
                if batch.get("speaker", None) is not None:
                    speaker = batch["speaker"].to(device, dtype)
                else:
                    speaker = None

                model_args = {}

                with torch.no_grad():
                    x = audio
                    model_args["x_mask_for_padding"] = lengths_to_mask(audio_length)
                    if content is not None:
                        model_args["content"] = content.squeeze(1)
                        model_args["content_mask_for_padding"] = lengths_to_mask(content_length)
                    if pitch is not None:
                        model_args["pitch"] = pitch.squeeze(1)
                        model_args["pitch_mask_for_padding"] = lengths_to_mask(pitch_length)
                    if speaker is not None:
                        model_args["speaker"] = speaker

                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        model_args[k] = v.to(device, dtype)

                # == sampling ==
                torch.manual_seed(1024)
                z = torch.randn_like(x)
                samples = scheduler.sample(
                    model,
                    z=z,
                    device=device,
                    additional_args=model_args,
                    progress=verbose >= 2,
                )
                B, T, C = samples.size()
                samples = rearrange(samples, "B T (S C) -> B (T S) C", T=T, S=dataset.audio_stack)
                samples = samples.cpu().numpy()

                # == save samples ==
                if is_main_process():
                    for path, mel, audio_len in zip(batch["path"], samples, audio_length):
                        mel = mel[:audio_len * dataset.audio_stack]
                        if dataset.audio_max is not None and dataset.audio_min is not None:
                            mel = np.clip(mel, -1, 1)
                            mel = (mel + 1) / 2 * (dataset.audio_max - dataset.audio_min) + dataset.audio_min

                        if vocoder is not None:
                            x = torch.from_numpy(mel).to(device)
                            audio = vocoder(x).cpu().numpy().squeeze(0)
                            assert "/video/" in path
                            assert dataset.root_path in path
                            audio_save_path = os.path.splitext(path.replace("/video/", "/generated_audio/").replace(dataset.root_path, save_dir))[0]+".wav"
                            assert audio_save_path != path
                            os.makedirs(os.path.dirname(audio_save_path), exist_ok=True)
                            sf.write(audio_save_path, audio, sampling_rate)

                start_idx += 1
    logger.info("Inference finished.")
    logger.info("Saved %s samples to %s", start_idx, save_dir)


if __name__ == "__main__":
    main()
