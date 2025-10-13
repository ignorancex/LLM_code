import scipy
import argparse
import numpy as np
import os
import cv2

import torch
from pytorchvideo.transforms import UniformTemporalSubsample
import torchvision
from torchvision import transforms

from va_core.model.svg.pipeline_svg import VideoAudioGenWithSVGPipeline
from va_core.dataset import get_dataset_info


def main():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--model_path", type=str, default=None, help="path to model.")
    parser.add_argument("--dataset_name", type=str, default=None, help="dataset name")
    parser.add_argument("--max_samples", type=int, default=None, help="num of data for generation.")
    parser.add_argument("--n_steps", type=int, default=50, help="num of steps for generation.")
    parser.add_argument("--n_try", type=int, default=1, help="num of trials for each prompt.")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size.")
    parser.add_argument("--length", type=int, default=4, help="length of audio to be generated.")
    parser.add_argument("--fps", type=int, default=4, help="fps of videos to be generated.")
    parser.add_argument("--cfg_a", type=float, default=2.5, help="strength of classfier-free guidance for audio.")
    parser.add_argument("--cfg_v", type=float, default=7.5, help="strength of classfier-free guidance for video.")
    parser.add_argument("--seed", type=int, default=141, help="random seed.")
    parser.add_argument("--out_dir", type=str, default="out", help="name of output directory.")
    parser.add_argument("--gamma", type=float, default=1.5, help="gamma for timestep adjustment.")
    parser.add_argument("--save_original", action="store_true", help="save original audio-video pairs if true.")
    parser.add_argument("--save_jpeg", action="store_true", help="save jpegs of generated videos if true.")
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(args.seed)
    duration_per_sample = args.length
    fps = args.fps
    n_frames_per_sample = duration_per_sample * fps
    resolution = 256
    sampling_rate = 16000
    neg_prompt = "bad quality, worse quality"

    ds_common_args = {
        "dataset_names": args.dataset_name,
        "duration_per_sample": duration_per_sample,
        "sampling_rate": sampling_rate
    }
    test_dataset_info = get_dataset_info(dataset_types="test",
                                         max_samples=args.max_samples,
                                         tokenizer=lambda x: x,
                                         **ds_common_args)
    test_dataset = test_dataset_info.dataset
    test_collate_fn = test_dataset_info.collate_fn
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=test_collate_fn,
        batch_size=args.batch_size,
        drop_last=False,
    )
    tf_video = torch.nn.Sequential(
        UniformTemporalSubsample(n_frames_per_sample, temporal_dim=1)
    )
    resize_and_crop = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
        ]
    )
    tf_audio = torch.nn.Sequential(
        UniformTemporalSubsample(sampling_rate * duration_per_sample, temporal_dim=-1)
    )

    pipe = VideoAudioGenWithSVGPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    if args.gamma > 0.0:
        pipe.scheduler_v.config.gamma = np.sqrt(args.gamma)
        pipe.scheduler_a.config.gamma = 1. / np.sqrt(args.gamma)

    pipe = pipe.to("cuda", torch.float16)

    # Prepare initial latents
    # audio
    vocoder_upsample_factor = np.prod(pipe.vocoder.config.upsample_rates) / pipe.vocoder.config.sampling_rate
    height_a = int(args.length / vocoder_upsample_factor)
    if height_a % (pipe.vae_a_scale_factor * (2 ** pipe.unet_a.num_upsamplers)) != 0:
        r = pipe.vae_a_scale_factor * (2 ** pipe.unet_a.num_upsamplers)
        height_a = int(np.ceil(height_a / r)) * r
    shape_a = (args.batch_size, pipe.unet_a.config.in_channels, height_a // pipe.vae_a_scale_factor, pipe.vocoder.config.model_in_dim // pipe.vae_a_scale_factor)
    # video
    shape_v = (args.batch_size, pipe.unet_v.config.in_channels, args.length * args.fps, 256 // pipe.vae_v_scale_factor, 256 // pipe.vae_v_scale_factor)

    os.makedirs(args.out_dir, exist_ok=True)
    out_dir_gen_wav = os.path.join(args.out_dir, "gen_wav")
    out_dir_gen_mp4 = os.path.join(args.out_dir, "gen_mp4")
    os.makedirs(out_dir_gen_wav, exist_ok=True)
    os.makedirs(out_dir_gen_mp4, exist_ok=True)
    if args.save_jpeg:
        out_dir_gen_jpg = os.path.join(args.out_dir, "gen_jpg")
        os.makedirs(out_dir_gen_jpg, exist_ok=True)
    if args.save_original:
        out_dir_orig_wav = os.path.join(args.out_dir, "orig_wav")
        out_dir_orig_mp4 = os.path.join(args.out_dir, "orig_mp4")
        os.makedirs(out_dir_orig_wav, exist_ok=True)
        os.makedirs(out_dir_orig_mp4, exist_ok=True)
        if args.save_jpeg:
            out_dir_orig_jpg = os.path.join(args.out_dir, "orig_jpg")
            os.makedirs(out_dir_orig_jpg, exist_ok=True)
    flist_name = os.path.join(args.out_dir, "gen_flist.txt")

    with open(flist_name, 'w') as flist:
        for n in range(args.n_try):
            for i, batch in enumerate(test_dataloader):
                prompt = batch["text"]
                bs = batch["audio"].shape[0]
                audio_orig = tf_audio(batch["audio"].to(device="cpu"))
                video_orig = tf_video(batch["video"].to(device="cpu"))
                video_orig = resize_and_crop(video_orig.flatten(end_dim=1) / 255.0).reshape([bs, n_frames_per_sample, 3, resolution, resolution])
                latents_a_init = torch.randn(shape_a, generator=generator, dtype=torch.float16)
                latents_v_init = torch.randn(shape_v, generator=generator, dtype=torch.float16)
                output = pipe(prompt, prompt, negative_prompt_v=[neg_prompt]*bs, negative_prompt_a=[neg_prompt]*bs, length_in_s=duration_per_sample, frame_rate=fps, num_inference_steps=args.n_steps, guidance_scale_a=args.cfg_a, guidance_scale_v=args.cfg_v, height_v=resolution, width_v=resolution, latents_a=latents_a_init, latents_v=latents_v_init)

                # save gen results
                for b in range(bs):
                    video_to_save = (output.videos[b] * 255.0).astype(np.uint8)
                    print(video_to_save.shape)
                    audio_to_save = output.audios[b]
                    prompt_str = prompt[b].replace(" ", "-")
                    if len(prompt_str) > 80:
                        prompt_str = prompt_str[:80]
                    mp4_file_name = os.path.join(out_dir_gen_mp4, f"{str(i)}_{str(b)}_{str(n)}_{prompt_str}.mp4")
                    wav_file_name = os.path.join(out_dir_gen_wav, f"{str(i)}_{str(b)}_{str(n)}_{prompt_str}.wav")
                    scipy.io.wavfile.write(wav_file_name, rate=sampling_rate, data=audio_to_save)
                    torchvision.io.write_video(mp4_file_name, video_to_save, fps=fps, video_codec="libx264", audio_array=audio_to_save.reshape((1, -1)), audio_fps=sampling_rate, audio_codec="aac")
                    if args.save_jpeg:
                        dir_jpg = os.path.join(out_dir_gen_jpg, f"{str(i)}_{str(b)}_{str(n)}_{prompt_str}")
                        os.makedirs(dir_jpg, exist_ok=True)
                        for t in range(n_frames_per_sample):
                            img = video_to_save[t][:,:,::-1]
                            if t == 0:
                                res = img.shape[:2]
                            cv2.imwrite(os.path.join(dir_jpg, f"{t:03d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    prompt_for_csv = prompt[b].replace("\"", "\"\"")
                    flist.write(f"\"{mp4_file_name}\",\"{wav_file_name}\",\"{prompt_for_csv}\"\n")
                
                # save orig
                if args.save_original:
                    video_to_save = (video_orig.numpy().transpose((0, 1, 3, 4, 2)) * 255.0).astype(np.uint8)
                    print(video_to_save.shape)
                    audio_to_save = audio_orig.numpy()
                    for b in range(bs):
                        prompt_str = prompt[b].replace(" ", "-")
                        if len(prompt_str) > 80:
                            prompt_str = prompt_str[:80]
                        scipy.io.wavfile.write(os.path.join(out_dir_orig_wav, f"{str(i)}_{str(b)}_{str(n)}_{prompt_str}.wav"), rate=sampling_rate, data=audio_to_save[b])
                        torchvision.io.write_video(os.path.join(out_dir_orig_mp4, f"{str(i)}_{str(b)}_{str(n)}_{prompt_str}.mp4"), video_to_save[b], fps=fps, video_codec="libx264", audio_array=audio_to_save[b].reshape((1, -1)), audio_fps=sampling_rate, audio_codec="aac")
                        if args.save_jpeg:
                            dir_jpg = os.path.join(out_dir_orig_jpg, f"{str(i)}_{str(b)}_{str(n)}_{prompt_str}")
                            os.makedirs(dir_jpg, exist_ok=True)
                            for t in range(n_frames_per_sample):
                                img = video_to_save[b][t][:,:,::-1]
                                img = cv2.resize(img, res)
                                cv2.imwrite(os.path.join(dir_jpg, f"{t:03d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])


if __name__ == "__main__":
    main()
