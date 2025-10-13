import scipy
import argparse
import numpy as np
import os
import cv2
import pandas as pd

import torch
import torchvision

from va_core.model.svg.pipeline_svg import VideoAudioGenWithSVGPipeline


def main():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--model_path", type=str, default=None, help="path to model.")
    parser.add_argument("--caption_file", type=str, default=None, help="path to caption list file")
    parser.add_argument("--max_samples", type=int, default=None, help="num of data for generation.")
    parser.add_argument("--n_steps", type=int, default=50, help="num of steps for generation.")
    parser.add_argument("--n_try", type=int, default=1, help="num of trials for each prompt.")
    parser.add_argument("--length", type=int, default=4, help="length of audio to be generated.")
    parser.add_argument("--fps", type=int, default=4, help="fps of videos to be generated.")
    parser.add_argument("--cfg_a", type=float, default=2.5, help="strength of classfier-free guidance for audio.")
    parser.add_argument("--cfg_v", type=float, default=7.5, help="strength of classfier-free guidance for video.")
    parser.add_argument("--seed", type=int, default=141, help="random seed.")
    parser.add_argument("--out_dir", type=str, default="out", help="name of output directory.")
    parser.add_argument("--gamma", type=float, default=1.5, help="gamma for timestep adjustment.")
    parser.add_argument("--save_jpeg", action="store_true", help="save jpegs of generated videos if true.")
    args = parser.parse_args()

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    duration_per_sample = args.length
    fps = args.fps
    n_frames_per_sample = duration_per_sample * fps
    resolution = 256
    sampling_rate = 16000
    neg_prompt = "bad quality, worse quality"

    # Load models
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
    flist_name = os.path.join(args.out_dir, "gen_flist.txt")

    with open(flist_name, 'w') as flist:
        for n in range(args.n_try):
            with open(args.caption_file, 'r') as f:
                for idx, prompt in enumerate(f):
                    prompt = "".join(prompt.splitlines()).strip("\"")
                    latents_a_init = torch.randn(shape_a, generator=generator, dtype=torch.float16)
                    latents_v_init = torch.randn(shape_v, generator=generator, dtype=torch.float16)
                    output = pipe(prompt, prompt, negative_prompt_v=neg_prompt, negative_prompt_a=neg_prompt, length_in_s=duration_per_sample, frame_rate=fps, num_inference_steps=args.n_steps, guidance_scale_a=args.cfg_a, guidance_scale_v=args.cfg_v, height_v=resolution, width_v=resolution, latents_a=latents_a_init, latents_v=latents_v_init)

                    # save gen results
                    video_to_save = (output.videos[0] * 255.0).astype(np.uint8)
                    audio_to_save = output.audios[0]
                    prompt_str = prompt.replace(" ", "-")
                    if len(prompt_str) > 80:
                        prompt_str = prompt_str[:80]
                    wav_file_name = os.path.join(out_dir_gen_wav, f"{str(idx)}_{str(n)}_{prompt_str}.wav")
                    mp4_file_name = os.path.join(out_dir_gen_mp4, f"{str(idx)}_{str(n)}_{prompt_str}.mp4")
                    scipy.io.wavfile.write(wav_file_name, rate=sampling_rate, data=audio_to_save)
                    torchvision.io.write_video(mp4_file_name, video_to_save, fps=fps, video_codec="libx264", audio_array=audio_to_save.reshape((1, -1)), audio_fps=sampling_rate, audio_codec="aac")
                    if args.save_jpeg:
                        dir_jpg = os.path.join(out_dir_gen_jpg, f"{str(idx)}_{str(n)}_{prompt_str}")
                        os.makedirs(dir_jpg, exist_ok=True)
                        for t in range(n_frames_per_sample):
                            img = video_to_save[t][:,:,::-1]
                            cv2.imwrite(os.path.join(dir_jpg, f"{t:03d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    flist.write(f"\"{mp4_file_name}\",\"{wav_file_name}\",\"{prompt}\"\n")
                    if args.max_samples is not None and idx == args.max_samples - 1:
                        break


if __name__ == "__main__":
    main()
