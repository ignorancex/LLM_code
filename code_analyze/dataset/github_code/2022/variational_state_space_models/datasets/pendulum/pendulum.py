"""pendulum dataset."""
import tensorflow_datasets as tfds
import dataclasses
import tensorflow as tf
from cairo import ImageSurface, Context, Format
import jax
import numpy as np
import haiku as hk
import math
from typing import Tuple
from datasets.util import cairo_to_numpy
import scipy

# TODO(pendulum): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""
# TODO(pendulum): BibTeX citation
_CITATION = """
"""

def render_pendulum(surface, ctx, width, height, state):
    ctx.rectangle(0, 0, width, height)
    ctx.set_source_rgb(0.9, 0.9, 0.9)
    ctx.fill()
    ctx.move_to(width/2, height/2)

    radius = 0.7*min(width, height)/2
    ball_radius = 0.1*min(width, height)/2

    # put theta through a tanh to prevent
    # wraparound
    theta = 3.1*math.tanh(state[0]/6)

    x = np.sin(theta)*radius + width/2
    y = np.cos(theta)*radius + height/2

    ctx.set_source_rgb(0.1, 0.1, 0.1)
    ctx.set_line_width(1)
    ctx.line_to(x, y)
    ctx.stroke()

    ctx.set_source_rgb(0.9, 0, 0)
    ctx.arc(x, y, ball_radius, 0, 2*math.pi)
    ctx.fill()
    img = cairo_to_numpy(surface)[:3,:,:]
    # we need to make a copy otherwise it will
    # get overridden the next time we render
    return np.copy(np.transpose(img, (1, 2, 0)))

@dataclasses.dataclass
class PendulumConfig(tfds.core.BuilderConfig):
    img_size: Tuple[int, int] = (64, 64) # height, width
    train_split_size: int = 10000
    test_split_size: int = 1000
    Prior = np.array([[1, 0], [0, 1]], dtype=np.float32)

class PendulumMetadata(tfds.core.Metadata):
    def load_metadata(self, dir):
        pass

    def save_metadata(self, dir):
        pass

class PendulumDataset(tfds.core.GeneratorBasedBuilder):
    name = "pendulum"
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial Release'
    }
    BUILDER_CONFIGS = [
        PendulumConfig(name='32x32', img_size=(32, 32)),
        PendulumConfig(name='64x64', img_size=(64, 64)),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=(self.builder_config.img_size[0], self.builder_config.img_size[1], 3)),
                'state': tfds.features.Tensor(shape=(2,), dtype=tf.dtypes.float32) # theta, theta_dot
            }),
            metadata=PendulumMetadata(
                Prior=self.builder_config.Prior,
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            'train': self._generate_examples(self.builder_config.train_split_size),
            'test': self._generate_examples(self.builder_config.test_split_size)
        }

    def _generate_examples(self, n):
        height = self.builder_config.img_size[0]
        width = self.builder_config.img_size[1]
        surface = ImageSurface(Format.ARGB32, width, height)
        ctx = Context(surface)

        random = hk.PRNGSequence(1234651237365)
        for i in range(n):
            state = jax.random.multivariate_normal(next(random), np.zeros((2,)),
                        self.builder_config.Prior)
            img = render_pendulum(surface, ctx, width, height, state)
            yield i, {
                'image': img,
                'state': state
            }

@dataclasses.dataclass
class PendulumTrajectoryConfig(tfds.core.BuilderConfig):
    img_size: Tuple[int, int] = (64, 64)
    train_split_size: int = 10000
    test_split_size: int = 1000
    length: int = 4
    A: np.array = np.array([[0.99, 0.1], [0, 0.998]], dtype=np.float32)
    B: np.array = np.array([[0], [0.2]], dtype=np.float32)
    K: np.array = np.array([[-0.1, -0.01]], dtype=np.float32)
    u_Sigma: float = 0.5
    Sigma: np.array = np.array([[0.0001, 0],[0, 0.0001]], dtype=np.float32)

    @property
    def Prior(self):
        Q = self.Sigma + self.B @ self.B.T * self.u_Sigma
        # the true A is A + B @ K
        A = self.A + self.B @ self.K
        # solve A X A ^T + Q = X for X
        return scipy.linalg.solve_discrete_lyapunov(A, Q)

class PendulumTrajectoryDataset(tfds.core.GeneratorBasedBuilder):
    name = "pendulum_trajectory"
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial Release'
    }
    BUILDER_CONFIGS = [
        PendulumTrajectoryConfig(name='32x32', img_size=(32, 32)),
        PendulumTrajectoryConfig(name='64x64', img_size=(64, 64)),
        PendulumTrajectoryConfig(name='64x64_lv', 
            A=np.array([[0.99, 0.2], [0, 0.998]], np.float32),
            B=np.array([[0.03], [0.3]], dtype=np.float32),
            K=np.array([[-0.5, -0.05]], np.float32),
            length=5,
            img_size=(64, 64))
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'images': tfds.features.Video(shape=(self.builder_config.length, self.builder_config.img_size[0], self.builder_config.img_size[1], 3)),
                'states': tfds.features.Tensor(shape=(self.builder_config.length, 2), dtype=tf.dtypes.float32), # theta, theta_dot for each timestep
                'inputs': tfds.features.Tensor(shape=(self.builder_config.length - 1, 1), dtype=tf.dtypes.float32) # control input per timestep
            }),
            metadata=PendulumMetadata(
                A=self.builder_config.A,
                B=self.builder_config.B,
                Prior=self.builder_config.Prior,
                Sigma=self.builder_config.Sigma
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            'train': self._generate_examples(self.builder_config.train_split_size),
            'test': self._generate_examples(self.builder_config.test_split_size)
        }

    def _generate_examples(self, n):
        height = self.builder_config.img_size[0]
        width = self.builder_config.img_size[1]
        surface = ImageSurface(Format.ARGB32, width, height)
        ctx = Context(surface)

        random = hk.PRNGSequence(1234651237365)
        Prior = self.builder_config.Prior

        i = 0
        while i < n:
            state = jax.random.multivariate_normal(next(random), 
                    np.zeros((2,), dtype=np.float32), Prior)
            state = np.expand_dims(state, -1)
            img = render_pendulum(surface, ctx, width, height, state.squeeze(-1))
            states = []
            imgs = []
            inputs = []

            for t in range(self.builder_config.length):
                imgs.append(img)
                states.append(state)
                u = self.builder_config.K @ state + \
                        np.sqrt(self.builder_config.u_Sigma) * jax.random.normal(next(random))
                if t < self.builder_config.length - 1:
                    inputs.append(u)
                noise = jax.random.multivariate_normal(next(random), np.zeros((2,), dtype=np.float32), 
                                                        self.builder_config.Sigma) \
                            if np.any(self.builder_config.Sigma) else np.zeros((2,), dtype=np.float32)
                noise = np.expand_dims(noise, -1)
                state = np.matmul(self.builder_config.A, state) + \
                            np.matmul(self.builder_config.B, u) + noise
                img = render_pendulum(surface, ctx, width, height, state.squeeze(-1))
            
            states = np.stack(states).squeeze(-1)
            inputs = np.stack(inputs).squeeze(-1)
            imgs = np.stack(imgs)

            max_theta = np.max(np.abs(states[:,0]))

            yield i, {
                'images': imgs,
                'states': states,
                'inputs': inputs
            }
            i = i + 1