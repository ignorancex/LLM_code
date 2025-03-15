from pyperlin import FractalPerlin2D
import torch
import numpy as np
import scipy.ndimage
from tqdm import tqdm
from PIL import Image
import os

from torch_utils import gen_utils
from network_features import DiscriminatorFeatures
from discriminator_synthesis import deep_dream, crop_resize_rotate


#  =====================================================================================================================  #


class FractalPerlin3D(object):
    def __init__(self, shape, resolutions, factors, channels: int = 1, generator=torch.random.default_generator,
                 resolutions_xy=None, factors_xy=None):
        self.shape = shape  # Shape should be (depth or timesteps, height, width)
        self.resolutions = resolutions
        self.resolutions_xy = resolutions_xy if resolutions_xy is not None else resolutions
        self.factors = factors
        self.factors_xy = factors_xy if factors_xy is not None else factors
        self.generator = generator
        self.device = generator.device
        self.channels = channels

    def generate_noise_for_plane(self, plane_shape, xy: bool = False):
        # Create a 2D Perlin noise generator for the given plane shape
        noise_gen = FractalPerlin2D(plane_shape, self.resolutions_xy if xy else self.resolutions,
                                    self.factors_xy if xy else self.factors, self.generator)
        # Generate and return the 2D noise
        return noise_gen(batch_size=self.channels)   # Assuming batch_size=1 and getting the first item

    def generate_noise(self):
        # Idea taken from @tntmeijs's Gist: https://gist.github.com/tntmeijs/6a3b4587ff7d38a6fa63e13f9d0ac46d
        # Generate noise for each of the six permutations
        noise_xy = self.generate_noise_for_plane(plane_shape=(self.shape[1], self.shape[2]), xy=True)
        noise_xz = self.generate_noise_for_plane(plane_shape=(self.shape[0], self.shape[2]))
        noise_yz = self.generate_noise_for_plane(plane_shape=(self.shape[0], self.shape[1]))

        # Mirror the noise maps to make them tileable
        noise_yx = self.generate_noise_for_plane(plane_shape=(self.shape[2], self.shape[1]), xy=True).transpose(1, 2)
        noise_zx = self.generate_noise_for_plane(plane_shape=(self.shape[2], self.shape[0])).transpose(1, 2)
        noise_zy = self.generate_noise_for_plane(plane_shape=(self.shape[1], self.shape[0])).transpose(1, 2)

        # Reshape the noise maps to fit into the 3D shape
        noise_xy = noise_xy.unsqueeze(1).expand(-1, self.shape[0], -1, -1)
        noise_xz = noise_xz.unsqueeze(2).expand(-1, -1, self.shape[1], -1)
        noise_yz = noise_yz.unsqueeze(3).expand(-1, -1, -1, self.shape[2])

        noise_yx = noise_yx.unsqueeze(1).expand(-1, self.shape[0], -1, -1)
        noise_zx = noise_zx.unsqueeze(2).expand(-1, -1, self.shape[1], -1)
        noise_zy = noise_zy.unsqueeze(3).expand(-1, -1, -1, self.shape[2])

        # Stack and average the noise maps to simulate 3D noise
        combined_noise = (noise_xy + noise_xz + noise_yz + noise_yx + noise_zx + noise_zy) / 6

        return combined_noise

    def __call__(self):
        return self.generate_noise()


#  =====================================================================================================================  #

seed = 42  # Seed for the random number generator
description = ''  # Description for the run
outdir = os.path.join(os.getcwd(), 'out', 'disc_test_3d')
starting_image_name = 'fractal_perlin3d'
layers = ['b4_conv']  # Layer to maximize the activations

num_frames = 128
fps = 25
resolutions_xy = [(2**i, 2**i) for i in range(1, 8)]
factors_xy = [0.5**i for i in range(7)]

resolutions = [(4**i, 4**i) for i in range(1, 3)]
factors = [0.25**i for i in range(2)]

network_pkl = 'metfaces1024'
cfg = 'stylegan2'

# Set up device and RNG
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
g_cuda = torch.Generator(device=device).manual_seed(seed)

# Load Discriminator
D = gen_utils.load_network('D', network_pkl, cfg, device)

# Get the model resolution (image resizing and getting available layers)
model_resolution = D.img_resolution

print('Generating 3D Perlin noise and smoothing it out...')
noise = FractalPerlin3D((num_frames, model_resolution, model_resolution),
                        resolutions, factors, channels=3, generator=g_cuda,
                        resolutions_xy=resolutions_xy, factors_xy=factors_xy).generate_noise() # [C, *shape]
# noise = (noise - noise.min()) / (noise.max() - noise.min())  # Normalize to [0, 1]
noise = (noise - noise.min(dim=0, keepdim=True).values ) / (noise.max(dim=0, keepdim=True).values - noise.min(dim=0, keepdim=True).values)  # Normalize to [0, 1]
noise = noise.cpu().numpy()
# noise = scipy.ndimage.gaussian_filter(noise, sigma=[0, 5, 0, 0], mode='wrap')
rgb_noise = (255 * noise).astype(np.uint8).transpose(1, 2, 3, 0)  # Normalize to [0, 255], and transpose to (T, H, W, C)


# We will use the features of the Discriminator, on the layer specified by the user
model = DiscriminatorFeatures(D).requires_grad_(False).to(device)

# Make the run dir in the specified output directory
desc = f'discriminator-dream-perlin3d_{seed}'
desc = f'{desc}-{description}' if len(description) != 0 else desc
run_dir = gen_utils.make_run_dir(outdir, desc)

# Number of digits for naming purposes
n_digits = int(np.log10(num_frames)) + 1

for idx, noise in enumerate(tqdm(rgb_noise, desc='Dreaming in 3D...', unit='frame')):
    image = Image.fromarray(noise, 'RGB')  # Create an image from the noise
    image = image.convert('L').convert('RGB')  # Convert to grayscale and back to RGB (small troll)

    # Save starting image
    image.save(os.path.join(run_dir, f'{starting_image_name}_{idx:0{n_digits}d}.jpg'))

    # Extract deep dream image
    dreamed_image = deep_dream(image, model, model_resolution, layers=layers, seed=seed, normed=False,
                               sqrt_normed=False, iterations=10, channels=None,
                               lr=1e-2, octave_scale=1.4, num_octaves=5,
                               unzoom_octave=True, disable_inner_tqdm=True)

    # Save the resulting image and initial image
    filename = f'dreamed_{idx:0{n_digits}d}.jpg'
    Image.fromarray(dreamed_image, 'RGB').save(os.path.join(run_dir, filename))

# Generate video
print('Saving video...')
gen_utils.save_video_from_images(
    run_dir, f'dreamed_%0{n_digits}d.jpg', '3dperlin_interpolation', fps=fps, reverse_video=False
)

gen_utils.save_video_from_images(
    run_dir, f'{starting_image_name}_%0{n_digits}d.jpg', 'perlin3d', fps=fps, reverse_video=False
)