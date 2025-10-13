from functools import partial
from typing import Any, Iterator, Literal, Optional, Tuple

import chex
import jax
from torch.utils.data import DataLoader
import numpy as np
import jax.numpy as jnp
from tqdm.auto import trange

from src.datasets.task_gen.task_generator import PatternTaskGenerator, ArcTrainTaskGenerator
from src.data_utils import data_augmentation_fn


class JAXDataLoader:
    """DataLoader wrapper for JAX since JAX does not support multi-threading."""

    def __init__(
        self,
        task_generator: PatternTaskGenerator | ArcTrainTaskGenerator,
        batch_size: int,
        log_every_n_steps: int,
        num_workers: int,
        worker_timeout: int = 0,
        num_devices: Optional[int] = None,
        online_data_augmentation: bool = True,
        seed: Optional[int] = None,
        return_info: bool = False,
        max_rows: int = 30,
        max_cols: int = 30,
    ):
        numpy_dataloader = DataLoader(
            task_generator,
            batch_size=batch_size * log_every_n_steps,
            num_workers=num_workers,
            collate_fn=partial(
                collate_fn,
                batch_size=batch_size,
                log_every_n_steps=log_every_n_steps,
                num_devices=num_devices,
                return_info=return_info,
                max_rows=max_rows,
                max_cols=max_cols,
            ),
            timeout=worker_timeout if num_workers > 0 else 0,
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )
        self.numpy_dataloader = numpy_dataloader
        self.online_data_augmentation = online_data_augmentation
        self.data_augmentation_fn = jax.jit(data_augmentation_fn, backend="cpu")
        self.key = jax.random.PRNGKey(seed or 0)
        self.return_info = return_info

    def __iter__(self) -> Iterator[Tuple[chex.Array, chex.Array] | Tuple[chex.Array, chex.Array, dict]]:
        for batch in self.numpy_dataloader:
            if self.return_info:
                *batch, info = batch
            batch = jax.tree_util.tree_map(partial(jnp.array, dtype=jnp.uint8), batch)
            if self.online_data_augmentation:
                self.key, augmentation_key = jax.random.split(self.key)
                batch = self.data_augmentation_fn(*batch, augmentation_key)
            if self.return_info:
                yield *batch, info
            else:
                yield batch


def collate_fn(
    batch: list[tuple[list[dict[str, tuple]], dict[str, Any]]],
    batch_size: int,
    log_every_n_steps: int,
    num_devices: Optional[int] = None,
    return_info: bool = False,
    max_rows: int = 30,
    max_cols: int = 30,
) -> tuple[chex.Array, chex.Array] | tuple[chex.Array, chex.Array, dict]:
    tasks, infos = zip(*batch)
    grids, shapes = [], []
    for task in tasks:
        task_input_grids, task_input_shapes, task_output_grids, task_output_shapes = [], [], [], []
        for pair in task:
            input_grid, output_grid = pair["input"], pair["output"]
            input_shape, output_shape = input_grid.shape, output_grid.shape
            input_grid = np.pad(input_grid, ((0, max_rows - input_shape[0]), (0, max_cols - input_shape[1])))
            output_grid = np.pad(
                output_grid, ((0, max_rows - output_shape[0]), (0, max_cols - output_shape[1]))
            )
            task_input_grids.append(input_grid)
            task_input_shapes.append(input_shape)
            task_output_grids.append(output_grid)
            task_output_shapes.append(output_shape)
        task_input_grids = np.stack(task_input_grids)
        task_input_shapes = np.stack(task_input_shapes)
        task_output_grids = np.stack(task_output_grids)
        task_output_shapes = np.stack(task_output_shapes)
        grids.append(np.stack([task_input_grids, task_output_grids], axis=-1))
        shapes.append(np.stack([task_input_shapes, task_output_shapes], axis=-1))
    grids, shapes = np.stack(grids), np.stack(shapes)
    if num_devices is None:
        # Reshape to (log_every_n_steps, batch_size, ...)
        grids = grids.reshape(log_every_n_steps, batch_size, *grids.shape[1:])
        shapes = shapes.reshape(log_every_n_steps, batch_size, *shapes.shape[1:])
    else:
        # Reshape to (num_devices, log_every_n_steps, batch_size // num_devices, ...)
        grids = grids.reshape(num_devices, log_every_n_steps, batch_size // num_devices, *grids.shape[1:])
        shapes = shapes.reshape(num_devices, log_every_n_steps, batch_size // num_devices, *shapes.shape[1:])
    if return_info:
        num_attempts = np.array([info["num_attempts_generate_task"] for info in infos])
        info = {"num_attempts_generate_task": num_attempts}
        if "program_id" in infos[0]:
            program_ids = np.array([info["program_id"] for info in infos])
            info["program_ids"] = program_ids
        if num_devices is None:
            info = jax.tree_util.tree_map(
                lambda x: x.reshape(log_every_n_steps, batch_size, *x.shape[2:]), info
            )
        else:
            info = jax.tree_util.tree_map(
                lambda x: x.reshape(num_devices, log_every_n_steps, batch_size // num_devices, *x.shape[2:]),
                info,
            )
        return grids, shapes, info
    return grids, shapes


def make_task_gen_dataloader(
    batch_size: int,
    log_every_n_steps: int,
    num_workers: int,
    task_generator_class: Literal["PATTERN", "ARC"],
    num_pairs: int,
    worker_timeout: int = 0,
    num_devices: Optional[int] = None,
    online_data_augmentation: bool = True,
    return_info: bool = False,
    seed: Optional[int] = None,
    **task_generator_kwargs,
) -> JAXDataLoader:
    max_rows, max_cols = task_generator_kwargs.get("max_rows", 30), task_generator_kwargs.get("max_cols", 30)
    if task_generator_class == "PATTERN":
        task_generator = PatternTaskGenerator(num_pairs=num_pairs, seed=seed, **task_generator_kwargs)
        max_rows, max_cols = task_generator.num_rows, task_generator.num_cols
    elif task_generator_class == "ARC":
        task_generator = ArcTrainTaskGenerator(num_pairs=num_pairs, seed=seed, **task_generator_kwargs)
    else:
        raise ValueError(f"Invalid task_generator_class: {task_generator_class}")
    jax_dataloader = JAXDataLoader(
        task_generator,
        batch_size,
        log_every_n_steps,
        num_workers,
        worker_timeout,
        num_devices,
        online_data_augmentation,
        seed,
        return_info,
        max_rows,
        max_cols,
    )
    return jax_dataloader


def make_dataset(
    length: int,
    num_pairs: int,
    num_workers: int,
    task_generator_class: Literal["PATTERN", "ARC"],
    online_data_augmentation: bool = True,
    seed: int = 0,
    **task_generator_kwargs,
) -> tuple[chex.Array, chex.Array, chex.Array]:
    dataloader = make_task_gen_dataloader(
        batch_size=1,
        log_every_n_steps=1,
        num_workers=num_workers,
        task_generator_class=task_generator_class,
        num_pairs=num_pairs,
        online_data_augmentation=online_data_augmentation,
        return_info=True,
        seed=seed,
        **task_generator_kwargs,
    )
    dataset_grids, dataset_shapes, program_ids = [], [], []
    for (grids, shapes, info), _ in zip(dataloader, trange(length, desc="Generating dataset")):
        dataset_grids.append(grids[0, 0])
        dataset_shapes.append(shapes[0, 0])
        program_ids.append(info["program_ids"][0, 0] if "program_ids" in info else 0)
    dataset_grids = jnp.stack(dataset_grids)
    dataset_shapes = jnp.stack(dataset_shapes)
    program_ids = jnp.stack(program_ids).astype(jnp.uint32)
    del dataloader
    return dataset_grids, dataset_shapes, program_ids


if __name__ == "__main__":
    import time

    from src.datasets.task_gen.utils import EMA

    dataloader = make_task_gen_dataloader(
        batch_size=100,
        log_every_n_steps=1,
        num_workers=0,
        task_generator_class="ARC",
        num_pairs=4,
        seed=0,
    )
    ema = EMA(start=time.time(), smoothing=0.05, return_inverse=True)
    for (grids, shapes), i in zip(dataloader, range(1000)):
        print("\nBatch", i + 1)
        print(grids.shape)
        print(f"Throughput: {ema(time.time()) * grids.shape[0] * grids.shape[1]:.2f} samples/s")
