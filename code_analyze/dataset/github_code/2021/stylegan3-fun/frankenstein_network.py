from torch_utils import gen_utils
import torch
import numpy as np
import moviepy.editor
import os
import scipy.ndimage


# def interpolate_models(model1, model2, alpha):
#     """
#     Interpolates the weights of two models based on the interpolation factor alpha, only for matching keys.
#
#     Args:
#     - model1 (torch.nn.Module): The first model.
#     - model2 (torch.nn.Module): The second model.
#     - alpha (float): Interpolation factor ranging from 0.0 (full model1) to 1.0 (full model2).
#
#     Returns:
#     - A new model with interpolated weights for matching keys.
#     """
#     alpha = min(max(alpha, 0.0), 1.0)  # Clip alpha to the range [0.0, 1.0]
#
#     # Initialize the interpolated model with the same configuration as model1
#     interpolated_model = type(model1)(
#         z_dim=model1.z_dim,
#         c_dim=model1.c_dim,
#         w_dim=model1.w_dim,
#         img_resolution=model1.img_resolution,
#         img_channels=model1.img_channels
#     ).eval().to(next(model1.parameters()).device)
#
#     state_dict1 = model1.state_dict()
#     state_dict2 = model2.state_dict()
#     interpolated_state_dict = {}
#
#     mismatched_keys = []  # Store keys that do not match
#
#     for key in state_dict1:
#         if key in state_dict2:
#             # Only interpolate if the key exists in both models
#             interpolated_state_dict[key] = (1 - alpha) * state_dict1[key] + alpha * state_dict2[key]
#         else:
#             # Log mismatched keys for review
#             mismatched_keys.append(key)
#
#     if mismatched_keys:
#         print("Mismatched keys, not interpolated:", mismatched_keys)
#
#     # Ensure only the interpolated keys are updated in the new model
#     interpolated_model.load_state_dict(interpolated_state_dict, strict=False)
#     return interpolated_model

def interpolate_multiple_models(models, weights=None):
    """
    Interpolates the weights of multiple models based on given weights or equally if no weights are provided.

    Args:
    - models (list of torch.nn.Module): The list of models to interpolate.
    - weights (list of float, optional): A list of weights for each model that sum to 1. Default is None, which assigns equal weight to each model.

    Returns:
    - torch.nn.Module: A new model with interpolated weights.
    """
    if weights is None:
        # If no weights provided, distribute equally
        weights = [1 / len(models)] * len(models)
    else:
        # Validate weights
        if len(weights) != len(models):
            raise ValueError("The number of weights must match the number of models.")
        if not np.isclose(sum(weights), 1):
            raise ValueError("The sum of weights must be 1.")

    # Assume all models are of the same type and have been initialized with the same parameters
    # Use the first model to determine type and reconstruction parameters
    interpolated_model = type(models[0])(
        z_dim=models[0].z_dim,
        c_dim=models[0].c_dim,
        w_dim=models[0].w_dim,
        img_resolution=models[0].img_resolution,
        img_channels=models[0].img_channels
    ).eval().to(next(models[0].parameters()).device)

    # Prepare to average the state dicts
    sum_dict = {}
    for model, weight in zip(models, weights):
        model_dict = model.state_dict()
        for key, value in model_dict.items():
            if key in sum_dict:
                sum_dict[key] += weight * value
            else:
                sum_dict[key] = weight * value

    # Load the weighted average state dict into the new model
    interpolated_model.load_state_dict(sum_dict, strict=False)
    return interpolated_model

# Set device
device = torch.device('cuda')
# Get a vanilla Generator
# G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=1024, img_channels=3).eval().to(device)

# G = gen_utils.load_network('G_ema',
#                            './pretrained/white_veil_5000.pkl',
#                            None, device).eval()
# Get the weights of two different pre-trained networks

G = gen_utils.load_network('G_ema',
                            'mlpony512',
                            'stylegan2', device).eval()

G2 = gen_utils.load_network('G_ema',
                            'fursona512',
                            'stylegan2', device).eval()


# Set the video parameters
grid_size = (1, 1)  # For now
fps = 30
duration_sec = 20.0
num_frames = int(fps * duration_sec)
shape = [num_frames, np.prod(grid_size), G.z_dim]

# Get the latents with the random state
seed = 42
random_state = np.random.RandomState(seed)
z = torch.randn(1, G.z_dim, device=device)
all_latents = random_state.randn(*shape).astype(np.float32)
# Let's smooth out the random latents so that now they form a loop (and are correctly generated in a 512-dim space)
all_latents = scipy.ndimage.gaussian_filter(all_latents, sigma=[3.0 * fps, 0, 0], mode='wrap')
all_latents /= np.sqrt(np.mean(np.square(all_latents)))

# Interpolation parameters
truncation_psi = 0.7

label = torch.zeros([1, G.c_dim], device=device)

def generate_sinusoidal_alphas(total_frames: int) -> np.ndarray:
    """
    Generates alpha values following a sinusoidal pattern from 0.0 to 1.0 and back to 0.0.
    Args:
    - num_frames (int): Total frames of video duration.
    Returns:
    - List of alpha values for each frame in the transition.
    """
    # Generate frame indices
    t = np.linspace(0, 2*np.pi, total_frames, endpoint=False)
    # Calculate sinusoidal alphas:
    alphas = 0.5 * (1 + np.sin(t - np.pi/2))  # Shift up by 1 and right by π/2 to start from 0 and scale to fit 0 to 1
    return alphas

alphas = generate_sinusoidal_alphas(num_frames)

# Generate the video frames
def make_frame(t):
    frame_idx = int(np.clip(np.round(t * fps), 0, num_frames - 1))
    alpha = alphas[frame_idx]

    # Interpolate the generator
    interpolated_generator = interpolate_multiple_models([G, G2], [1 - alpha**2, alpha**2])

    # Get the latents
    z = torch.from_numpy(all_latents[frame_idx]).to(device)
    # Generate the image
    img = gen_utils.z_to_img(interpolated_generator, z, label, truncation_psi=truncation_psi, noise_mode='const')
    grid = gen_utils.create_image_grid(img, (1, 1))

    return grid


desc = 'franken-video'
outdir = os.path.join(os.getcwd(), 'out', 'franken-video')
run_dir = gen_utils.make_run_dir(outdir, desc)

# Generate video using the respective make_frame function
videoclip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
videoclip.set_duration(duration_sec)

mp4_name = 'franken-video'

# Change the video parameters (codec, bitrate) if you so desire
final_video = os.path.join(run_dir, f'{mp4_name}.mp4')
videoclip.write_videofile(final_video, fps=fps, codec='libx264', bitrate='16M')

gen_utils.compress_video(original_video=final_video, original_video_name=mp4_name, outdir=run_dir)


##########

# import numpy as np
# import matplotlib.pyplot as plt
#
# # Define the domain
# x = np.linspace(0, 1, 400)
#
# # Define the functions
# # f = (1/6) * (1 + np.cos(np.pi * x))**2
# # g = (1/6) * (1 + np.cos(2 * np.pi * (x - 0.5)))**2
# # h = (1/6) * (1 + np.cos(np.pi * (x - 1)))**2
#
# # f = 1 - (1 - 4*(x - 0.5) ** 2) - x**2
# f = -(x - 1) ** 3
# g = 1 - 4*(x - 0.5) ** 2
# h = x**2
#
# # Calculate the sum of the functions
# sum_fg_h = f + g + h
#
# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(x, f, label='f(x) = 1/3 (1 + cos(πx))', linestyle='--')
# plt.plot(x, g, label='g(x) = 1/3 (1 + cos(2π(x - 0.5)))', linestyle='-.')
# plt.plot(x, h, label='h(x) = 1/3 (1 + cos(π(x - 1)))', linestyle=':')
# plt.plot(x, sum_fg_h, label='f(x) + g(x) + h(x)', color='black', linewidth=2)
# plt.title('Plot of Functions f(x), g(x), h(x), and Their Sum')
# plt.xlabel('x')
# plt.ylabel('Value')
# # plt.ylim(0, 1.1)  # Extend y-axis slightly above 1 for clarity
# plt.grid(True)
# plt.legend()
# plt.show()
