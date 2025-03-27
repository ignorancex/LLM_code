"""airsim dataset."""

import tensorflow_datasets as tfds
from util import airsim
import math
import signal
import subprocess
import time
import haiku as hk
import jax
import scipy

from airsim.types import Pose, Vector3r, Quaternionr
import numpy as np
import os
import stat
from io import BytesIO
from PIL import Image
import tensorflow as tf
import dataclasses
from typing import Tuple

# TODO(airsim): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(airsim): BibTeX citation
_CITATION = """
"""
URL_FORMAT="https://github.com/microsoft/AirSim/releases/download/v1.6.0-linux/{env}.zip"

def render_airsim(state, client, height, width, x_off, y_off, z, theta):
    x = 3*state[0].item() + x_off
    y = 3*state[1].item() + y_off
    z = -z

    pose = Pose(Vector3r(x, y, z), airsim.to_quaternion(0, 0, theta))
    client.simSetVehiclePose(pose, True)
    raw = client.simGetImages([
        airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)
    ])[0]
    arr = np.fromstring(raw.image_data_uint8, dtype=np.uint8)
    arr = arr.reshape(raw.height, raw.width, 3)
    arr[:, :, [0, 1, 2]] = arr[:, :, [2, 1, 0]]
    # take center crop and resize
    dim = min(arr.shape[0], arr.shape[1])
    hdim = dim // 2
    center = (arr.shape[0]//2, arr.shape[1]//2)
    top = int(center[0] - hdim)
    bottom = int(center[0] + hdim)
    left = int(center[1] - hdim)
    right = int(center[1] + hdim)

    crop = arr[top:bottom, left:right]
    return tf.image.resize(crop, (height, width)).numpy().astype(np.uint8)

@dataclasses.dataclass
class AirsimConfig(tfds.core.BuilderConfig):
    env: str = "Blocks"
    img_size: Tuple[int, int] = (64, 64) # height, width
    train_split_size: int = 10000
    test_split_size: int = 1000
    z: float = 15
    x_off: float = 0
    y_off: float = 0
    theta: float = 0.785398
    Prior: np.array = np.eye(4)

class AirsimMetadata(tfds.core.Metadata):
    def load_metadata(self, dir):
        pass

    def save_metadata(self, dir):
        pass

class AirsimDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for airsim dataset."""
    name = "airsim"

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    BUILDER_CONFIGS = [
        AirsimConfig(name='blocks_64x64', env='Blocks', img_size=(64, 64)),
        AirsimConfig(name='zhang_jiajie_64x64', 
            env='ZhangJiajie', img_size=(64, 64), 
            theta=0, z=0, Prior=1.7*np.eye(4)
        ),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(airsim): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(shape=(self.builder_config.img_size[0], self.builder_config.img_size[1], 3)),
                'state': tfds.features.Tensor(shape=(4,), dtype=tf.dtypes.float32)
            }),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        env = self.builder_config.env
        env_path = dl_manager.download_and_extract(URL_FORMAT.format(env=env))
        binary_path = os.path.join(env_path, f'{env}/LinuxNoEditor/{env}/Binaries/Linux/{env}')
        st = os.stat(binary_path)
        os.chmod(binary_path, st.st_mode | stat.S_IEXEC)
        return {
            'train': self._generate_examples(binary_path, self.builder_config.train_split_size),
            'test': self._generate_examples(binary_path, self.builder_config.test_split_size)
        }

    def _generate_examples(self, binary_path, n):
        # launch airsim binary
        with open("/tmp/airsim.log", "w") as log:
            with subprocess.Popen([binary_path], stdout=log, stderr=log, shell=False) as proc:
                try:
                    time.sleep(3)
                    client = airsim.MultirotorClient()
                    render = lambda state: render_airsim(state, client, 
                                            self.builder_config.img_size[0], self.builder_config.img_size[1], 
                                            self.builder_config.x_off, self.builder_config.y_off,
                                            self.builder_config.z, self.builder_config.theta)
                    if not client.ping():
                        raise RuntimeError("Could not connect to airsim")
                    random = jax.random.PRNGKey(516124612321)
                    for i in range(n):
                        random, sk = jax.random.split(random)
                        state = jax.random.multivariate_normal(sk, np.zeros((4,)),
                                    self.builder_config.Prior)
                        img = render(state)
                        yield i, {
                            'image': img,
                            'state': state
                        }
                finally:
                    proc.send_signal(signal.SIGTERM)
                    proc.wait()
            log.flush()


DEFAULT_B = np.array([[0.05, 0], [0, 0.05], [0.3, 0], [0, 0.3]], dtype=np.float32)

@dataclasses.dataclass
class AirsimTrajectoryConfig(tfds.core.BuilderConfig):
    length: int = 5
    env: str = "Blocks"
    img_size: Tuple[int, int] = (64, 64) # height, width
    train_split_size: int = 10000
    test_split_size: int = 1000
    z: float = 15
    x_off: float = 0
    y_off: float = 0
    theta: float = 0.785398

    # dynamics
    A: np.array = np.array([[0.98, 0, 0.2, 0], [0, 0.98, 0, 0.2],
                  [0, 0, 0.999, 0], [0, 0, 0, 0.999]], dtype=np.float32)
    B: np.array = DEFAULT_B
    K: np.array = np.array([[-0.5, 0, -0.05, 0],
                            [0, -0.5, 0, -0.05]], dtype=np.float32)
    Sigma: np.array = 0.001 * np.eye(4)
    u_Sigma: float = 1

    @property
    def Prior(self):
        Q = self.B @ self.B.T * self.u_Sigma + self.Sigma
        A = self.A + self.B @ self.K
        return scipy.linalg.solve_discrete_lyapunov(A, Q)

class AirsimTrajectoryDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for airsim dataset."""
    name = "airsim_trajectory"

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    BUILDER_CONFIGS = [
        AirsimTrajectoryConfig(
            name='blocks_64x64', env='Blocks', img_size=(64, 64)
        ),
        AirsimTrajectoryConfig(
            name='zhang_jiajie_64x64', 
            env='ZhangJiajie', img_size=(64, 64), 
            theta=0, z=0
        )
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(airsim): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'images': tfds.features.Video(shape=(self.builder_config.length,
                    self.builder_config.img_size[0], self.builder_config.img_size[1], 3)),
                'states': tfds.features.Tensor(shape=(self.builder_config.length, 4), dtype=tf.dtypes.float32),
                'inputs': tfds.features.Tensor(shape=(self.builder_config.length - 1, 2), dtype=tf.dtypes.float32)
            }),
            metadata=AirsimMetadata(
                A=self.builder_config.A,
                B=self.builder_config.B,
                Sigma=self.builder_config.Sigma,
                Prior=self.builder_config.Prior
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        env = self.builder_config.env
        env_path = dl_manager.download_and_extract(URL_FORMAT.format(env=env))
        binary_path = os.path.join(env_path, f'{env}/LinuxNoEditor/{env}/Binaries/Linux/{env}')
        st = os.stat(binary_path)
        os.chmod(binary_path, st.st_mode | stat.S_IEXEC)
        return {
            'grid': self._generate_grid(binary_path),
            'train': self._generate_examples(binary_path, self.builder_config.train_split_size),
            'test': self._generate_examples(binary_path, self.builder_config.test_split_size)
        }

    def _generate_grid(self, binary_path):
        with open("/tmp/airsim.log", "w") as log:
            with subprocess.Popen([binary_path], stdout=log, stderr=log, shell=False) as proc:
                try:
                    time.sleep(3)
                    client = airsim.MultirotorClient()
                    render = lambda state: render_airsim(state, client, 
                                            self.builder_config.img_size[0], self.builder_config.img_size[1], 
                                            self.builder_config.x_off, self.builder_config.y_off,
                                            self.builder_config.z, self.builder_config.theta)
                    a = np.arange(-2*self.builder_config.Prior[0, 0], 2*self.builder_config.Prior[0, 0], 0.1)
                    T = self.builder_config.length
                    i = 0
                    for x in a:
                        for y in a:
                            state = np.array([x, y, 0, 0], dtype=np.float32)
                            img = render(state)
                            yield i, {
                                'images': np.tile(np.expand_dims(img, 0), (T, 1, 1, 1)),
                                'states': np.tile(np.expand_dims(state, 0), (T, 1)),
                                'inputs': np.zeros((T - 1, 2), dtype=np.float32)
                            }
                            i = i + 1
                finally:
                    proc.send_signal(signal.SIGTERM)
                    proc.wait()
            log.flush()

    def _generate_examples(self, binary_path, n):
        # launch airsim binary
        with open("/tmp/airsim.log", "w") as log:
            with subprocess.Popen([binary_path], stdout=log, stderr=log, shell=False) as proc:
                try:
                    time.sleep(3)
                    client = airsim.MultirotorClient()
                    render = lambda state: render_airsim(state, client, 
                                            self.builder_config.img_size[0], self.builder_config.img_size[1], 
                                            self.builder_config.x_off, self.builder_config.y_off,
                                            self.builder_config.z, self.builder_config.theta)
                    if not client.ping():
                        raise RuntimeError("Could not connect to airsim")
                    random = hk.PRNGSequence(516124612321)
                    Prior = self.builder_config.Prior
                    for i in range(n):
                        state = jax.random.multivariate_normal(next(random), np.zeros((4,)),
                                    Prior)
                        state = np.expand_dims(state, -1)
                        img = render(state)
                        states = [state]
                        imgs = [img]
                        inputs = []
                        for t in range(1, self.builder_config.length):
                            u_noise = math.sqrt(self.builder_config.u_Sigma) * jax.random.normal(next(random), (2,), dtype=np.float32)
                            u_noise = np.expand_dims(u_noise, -1)
                            u = self.builder_config.K @ state + u_noise
                                    
                            inputs.append(u)
                            noise = jax.random.multivariate_normal(next(random), np.zeros((4,)),
                                        self.builder_config.Sigma)
                            noise = np.expand_dims(noise, -1)
                            state = np.matmul(self.builder_config.A, state) + \
                                        np.matmul(self.builder_config.B, u) + noise
                            img = render(state.squeeze(-1))
                            states.append(state)
                            imgs.append(img)
                        
                        states = np.stack(states).squeeze(-1)
                        imgs = np.stack(imgs)
                        inputs = np.stack(inputs).squeeze(-1)

                        yield i, {
                            'images': imgs,
                            'states': states,
                            'inputs': inputs
                        }
                finally:
                    proc.send_signal(signal.SIGTERM)
                    proc.wait()
            log.flush()