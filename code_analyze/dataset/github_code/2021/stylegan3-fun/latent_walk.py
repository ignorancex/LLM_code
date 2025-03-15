import click
from typing import Union, List, Optional

from PIL import Image
import numpy as np
import torch
import scipy

import dnnlib
import legacy
from torch_utils import gen_utils

import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import moviepy.editor


# ----------------------------------------------------------------------------


# Helper function for parsing seeds for latent walk
def _parse_path(s: str) -> List[Union[str, os.PathLike]]:
    """
    Input:
        s (str): Comma separated list of names of dlatent vectors to visit
    Output:
        (list): List of names of dlatents to visit
    """
    # Some sanity check
    s = s.replace(' ', '')
    # Split w.r.t. comma
    str_list = s.split(',')
    # Return the elements of s as a list of strings
    return [str(el) for el in str_list]


# ----------------------------------------------------------------------------


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--device', help='Device to use for image generation; using the CPU is much slower than the GPU', type=click.Choice(['cpu', 'cuda']), default='cuda', show_default=True)
@click.option('--cfg', type=click.Choice(['stylegan2', 'stylegan3-t', 'stylegan3-r']), help='Config of the network, used only if you want to use the pretrained models in torch_utils.gen_utils.resume_specs')
# Synthesis options
@click.option('--desired-path', '-path', type=_parse_path, help='Path of the dlatents to visit (in order)', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-start', 'truncation_psi_start', type=float, help='Initial value of pulsating truncation psi', default=None, show_default=True)
@click.option('--trunc-end', 'truncation_psi_end', type=float, help='Maximum/minimum value of pulsating truncation psi', default=None, show_default=True)
@click.option('--global-pulse', 'global_pulsation_trick', is_flag=True, help='If set, the truncation psi will pulsate globally (on all grid cells)')
@click.option('--wave-pulse', 'wave_pulsation_trick', is_flag=True, help='If set, the truncation psi will pulsate in a wave-like fashion from the upper left to the lower right in the grid')
@click.option('--frequency', 'pulsation_frequency', type=int, help='Frequency of the pulsation', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--anchor-latent-space', '-anchor', is_flag=True, help='Anchor the latent space to w_avg to stabilize the video')
# Video options
@click.option('--grid-width', '-gw', type=click.IntRange(min=1), help='Video grid width / number of columns', required=True)
@click.option('--grid-height', '-gh', type=click.IntRange(min=1), help='Video grid height / number of rows', required=True)
@click.option('--dlatent-sec', '-sec', type=gen_utils.float_list, help='Duration length in seconds between each dlatent. Comma-separated values; if one is provided, it will be the same between all dlatents.', default=[5.0], show_default=True)
@click.option('--interp-type', '-interp', type=click.Choice(['linear', 'spherical']), help='Type of interpolation in W', default='spherical', show_default=True)
@click.option('--smooth', is_flag=True, help='Add flag to smooth the transition between dlatents')
@click.option('--smooth-path', is_flag=True, help='Add flag to smooth the whole path; might need fine-tuning!')
@click.option('--fps', type=gen_utils.parse_fps, help='Video FPS.', default=30, show_default=True)
@click.option('--compress', is_flag=True, help='Add flag to compress the final mp4 file via ffmpeg-python (same resolution, lower file size)')
# Extra parameters for saving the results
@click.option('--save-every-frame', '-saveall', is_flag=True, help='Save every frame into as a .png in the outdir')
@click.option('--save-dlatents', is_flag=True, help='Use flag to save individual dlatents (W) for each individual resulting image')
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out', 'latent_walk'), show_default=True, metavar='DIR')
@click.option('--desc', type=str, help='Description name for the directory path to save results', default='latent-walk', show_default=True)
def latent_walk(
        ctx: click.Context,
        network_pkl: Union[str, os.PathLike],
        device: Optional[str],
        cfg: str,
        desired_path: List[str],
        truncation_psi: float,
        truncation_psi_start: Optional[float],
        truncation_psi_end: Optional[float],
        global_pulsation_trick: Optional[bool],
        wave_pulsation_trick: Optional[bool],
        pulsation_frequency: int,
        grid_width: Optional[int],
        grid_height: Optional[int],
        noise_mode: str,
        anchor_latent_space: Optional[bool],
        dlatent_sec: List[float],
        interp_type: str,
        smooth: Optional[bool],
        smooth_path: Optional[bool],
        fps: int,
        compress: Optional[bool],
        save_every_frame: Optional[bool],
        save_dlatents: Optional[bool],
        outdir: Union[str, os.PathLike],
        desc: str,
):
    # Path must visit at least 2 points in W!
    if len(desired_path) == 1:
        ctx.fail('"--desired-path" must have more than one element!')

    # If only one duration is provided, use the same between each dlatent
    if len(dlatent_sec) == 1:
        dlatent_sec = (len(desired_path) - 1) * dlatent_sec

    # Sanity check:
    if len(dlatent_sec) != len(desired_path) - 1:
        ctx.fail('Number of elements in "--dlatent-sec" should be one less than "--desired-path"!')

    dlatent_sec = np.array(dlatent_sec)

    # Number of steps to take between each latent vector
    n_steps = np.rint(dlatent_sec * fps).astype(int)

    # Number of frames in total
    num_frames = n_steps.sum()
    # Duration in seconds
    duration_sec = num_frames / fps

    n_digits = int(np.log10(num_frames)) + 1  # number of digits for naming the .jpg images

    # If model name exists in the gen_utils.resume_specs dictionary, use it instead of the full url
    try:
        network_pkl = gen_utils.resume_specs[cfg][network_pkl]
    except KeyError:
        # Otherwise, it's a local file or an url
        pass

    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device(device)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    # Setup for using CPU
    if device.type == 'cpu':
        gen_utils.use_cpu(G)

    # Stabilize/anchor the latent space
    if anchor_latent_space:
        gen_utils.anchor_latent_space(G)

    # Create the run dir with the given name description
    desc = f'{desc}-smooth_path' if smooth_path else desc
    run_dir = gen_utils.make_run_dir(outdir, desc)

    # Get all the latent vectores from each of the directories in desired_path
    print('Retrieveing W vectors...')
    all_w = np.stack([np.squeeze(gen_utils.parse_all_projected_dlatents(dlatent_dir), axis=1) for dlatent_dir in desired_path])
    src_w = np.empty([0] + list(all_w.shape[1:]), dtype=np.float32)
    for i in range(len(all_w) - 1):
        # We interpolate between each pair of dlatents
        interp = gen_utils.interpolate(all_w[i], all_w[i + 1], n_steps[i], interp_type, smooth)
        # Append it to our source
        src_w = np.append(src_w, interp, axis=0)

    # Smoothen the path?
    if smooth_path:
        # 5/4 is arbitrary, this has to bee fine-tuned to your needs
        src_w = scipy.ndimage.gaussian_filter(src_w, sigma=[(5 / 4) * fps, 0, 0, 0], mode='nearest')

    # For the truncation trick
    w_avg = G.mapping.w_avg

    # Aux function: Frame generation func for moviepy.
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * fps), 0, num_frames - 1))
        # Select the pertinent w dlatent:
        w = torch.from_numpy(src_w[frame_idx]).to(device)
        if None not in (truncation_psi_start, truncation_psi_end):
            # For both, truncation psi will have the general form of a sinusoid: psi = (cos(t) + alpha) / beta
            if global_pulsation_trick:
                # print('Using global pulsating truncation trick...')
                tr = gen_utils.global_pulsate_psi(psi_start=truncation_psi_start,
                                                  psi_end=truncation_psi_end,
                                                  n_steps=num_frames)

            elif wave_pulsation_trick:
                # print('Using wave pulse truncation trick...')
                tr = gen_utils.wave_pulse_truncation_psi(psi_start=truncation_psi_start,
                                                         psi_end=truncation_psi_end,
                                                         n_steps=num_frames,
                                                         grid_shape=(grid_width, grid_height),
                                                         frequency=pulsation_frequency,
                                                         time=frame_idx)
        # Define how to use the truncation psi
        if global_pulsation_trick:
            tr = tr[frame_idx].to(device)
        elif wave_pulsation_trick:
            tr = tr.to(device)
        else:
            tr = truncation_psi

        w = w_avg + (w - w_avg) * tr
        # Run it through Gs to get the image
        image = gen_utils.w_to_img(G, w, noise_mode)
        # Generate the grid for this timestamp:
        grid = gen_utils.create_image_grid(image, (grid_width, grid_height))
        # grayscale => RGB
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2)
        # Save each frame if user wants to
        if save_every_frame and frame_idx % fps == 0:
            frame_name = f'frame-{frame_idx:0{n_digits}d}.png'
            im = Image.fromarray(grid)
            im.save(os.path.join(run_dir, frame_name))
        # Save each dlatent as a .npy file, if user wishes to
        if save_dlatents:
            np.save(os.path.join(run_dir, f'frame-{frame_idx:0{n_digits}d}.npy'), w.unsqueeze(0).cpu().numpy())
        return grid

    # Generate video using make_frame:
    print('Generating latent_walk video...')
    mp4 = "latent_walk"
    videoclip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    videoclip.set_duration(duration_sec)

    # Change the video parameters (codec, bitrate) if you so desire
    final_video = os.path.join(run_dir, f'{mp4}.mp4')
    videoclip.write_videofile(final_video, fps=fps, codec='libx264', bitrate='16M')

    # Compress the video (lower file size, same resolution)
    if compress:
        gen_utils.compress_video(original_video=final_video, original_video_name=mp4, outdir=run_dir, ctx=ctx)

# ----------------------------------------------------------------------------


if __name__ == '__main__':
    latent_walk()
