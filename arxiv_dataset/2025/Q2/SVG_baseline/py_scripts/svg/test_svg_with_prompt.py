import numpy as np
import os
import scipy
import argparse

import torch
import torchvision

from va_core.model.svg.pipeline_svg import VideoAudioGenWithSVGPipeline


def main():
    parser = argparse.ArgumentParser(description="A script for testing with a given prompt.")
    parser.add_argument("--model_path", type=str, default=None, help="path to model.")
    parser.add_argument("--prompt", type=str, default=None, help="path to model.")
    parser.add_argument("--n_steps", type=int, default=50, help="num of steps for generation.")
    parser.add_argument("--duration", type=int, default=4, help="length of audio to be generated.")
    parser.add_argument("--fps", type=int, default=4, help="length of audio to be generated.")
    parser.add_argument("--cfg_a", type=float, default=2.5, help="coefficient for classfier-free guidance.")
    parser.add_argument("--cfg_v", type=float, default=7.5, help="coefficient for classfier-free guidance.")
    parser.add_argument("--out_dir", type=str, default="out", help="output filename")
    parser.add_argument("--gamma", type=float, default=1.5, help="gamma.")
    args = parser.parse_args()

    if args.model_path is None or args.prompt is None:
        print("You need to specify both model_path and prompt.")
        return

    pipe = VideoAudioGenWithSVGPipeline.from_pretrained(args.model_path)
    if args.gamma > 0.0:
        pipe.scheduler_v.config.gamma = np.sqrt(args.gamma)
        pipe.scheduler_a.config.gamma = 1. / np.sqrt(args.gamma)

    pipe = pipe.to("cuda", torch.float16)

    # Generate
    prompt_a = args.prompt
    prompt_v = args.prompt
    neg_prompt = "bad quality, worse quality"

    pipe_args = {
        "prompt_a": prompt_a,
        "prompt_v": prompt_v,
        "negative_prompt_a": neg_prompt,
        "negative_prompt_v": neg_prompt,
        "num_inference_steps": args.n_steps,
        "length_in_s": args.duration,
        "frame_rate": args.fps,
        "guidance_scale_a": args.cfg_a,
        "guidance_scale_v": args.cfg_v,
    }
    output = pipe(**pipe_args)

    # Save results
    video_to_save = (output.videos[0] * 255.0).astype(np.uint8)
    audio_to_save = output.audios[0]
    fname = args.prompt.replace(" ", "_")
    torchvision.io.write_video(os.path.join(args.out_dir, fname + ".mp4"), video_to_save, fps=args.fps, video_codec="libx264", audio_array=audio_to_save.reshape((1, -1)), audio_fps=16000, audio_codec="aac")
    scipy.io.wavfile.write(os.path.join(args.out_dir, fname + ".wav"), rate=16000, data=audio_to_save)


if __name__ == "__main__":
    main()
