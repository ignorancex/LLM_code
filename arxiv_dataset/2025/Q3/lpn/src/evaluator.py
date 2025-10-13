from functools import partial
from typing import Optional

import chex
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm
import numpy as np

from src.models.lpn import LPN
from src.datasets.task_gen.re_arc_generators import ARC_TASK_NAMES


class Evaluator:
    def __init__(
        self, model: LPN, inference_mode: str, inference_mode_kwargs: dict, devices: Optional[dict] = None
    ):
        self.model = model
        self.inference_mode = inference_mode
        self.inference_mode_kwargs = inference_mode_kwargs
        self.max_rows = self.model.encoder.config.max_rows
        self.max_cols = self.model.encoder.config.max_cols
        self.devices = devices or jax.local_devices()
        self.debug_msg = False
        self.pmap_generate_output = jax.pmap(
            partial(
                model.apply,
                dropout_eval=True,
                mode=inference_mode,
                return_two_best=True,
                **inference_mode_kwargs,
                method=model.generate_output,
            ),
            axis_name="devices",
            devices=self.devices[:1],
            donate_argnums=(3, 4),  # donate input and input_grid_shape
        )

    def json_submission(
        self,
        challenges: dict[str, list],
        params: dict,
        only_n_tasks: Optional[int] = None,
        overfit_task: Optional[str] = None,
        progress_bar: bool = False,
        key: Optional[chex.PRNGKey] = None,
        train: bool = False,
    ) -> dict[str, list]:
        # Only use the first device for the forward pass
        single_device_params = jax.tree_util.tree_map(lambda x: x[:1], params)
        if key is None:
            key = jax.random.PRNGKey(0)
        assert only_n_tasks is None or overfit_task is None, "Cannot use both only_n_tasks and overfit_task."
        if overfit_task is not None:
            assert overfit_task in challenges, f"Task {overfit_task} not found in the challenges."
            challenges = {overfit_task: challenges[overfit_task]}
            only_n_tasks = None
        if only_n_tasks is not None:
            num_tasks = min(only_n_tasks, len(challenges))
        else:
            num_tasks = len(challenges)
        if num_tasks < len(challenges):
            task_names = ARC_TASK_NAMES if train else list(challenges.keys())
            challenges = {task_name: challenges[task_name] for task_name in task_names[:num_tasks]}

        results = {}
        # TODO: maybe vectorize the forward to pass the whole dataset at once, splitting it over devices
        for task_id, task in tqdm(
            challenges.items(), total=num_tasks, desc="Generating solutions", disable=not progress_bar
        ):
            pair_list, shape_list = [], []
            for example in task["train"]:
                input = jnp.array(example["input"])
                input_shape = input.shape
                input, input_shape = self.pad_and_crop_json(input, input_shape)
                output = jnp.array(example["output"])
                output_shape = output.shape
                output, output_shape = self.pad_and_crop_json(output, output_shape)
                pair_list.append(jnp.stack([input, output], axis=-1))
                shape_list.append(jnp.stack(jnp.array([input_shape, output_shape]), axis=-1))
            pairs = jnp.stack(pair_list)
            grid_shapes = jnp.stack(shape_list)

            task_outputs = []
            for example in task["test"]:
                input = jnp.array(example["input"])
                input, input_grid_shape = self.pad_and_crop_json(input, input.shape)
                input_grid_shape = jnp.array(input_grid_shape)

                key, sub_key = jax.random.split(key)
                # Add batch dim and duplicate to 1 device (pmap is run on 1 device only).
                b_pairs, b_grid_shapes, b_input, b_input_grid_shape, sub_key = jax.device_put_replicated(
                    (pairs[None], grid_shapes[None], input[None], input_grid_shape[None], sub_key),
                    self.devices[:1],
                )
                *outputs, _ = self.pmap_generate_output(
                    {"params": single_device_params},
                    b_pairs,
                    b_grid_shapes,
                    b_input,
                    b_input_grid_shape,
                    sub_key,
                )
                # Remove batch dim and device dim
                first_output_grid, first_output_grid_shape, second_output_grid, second_output_grid_shape = (
                    jax.tree_util.tree_map(lambda x: x[0, 0], outputs)
                )

                # Crop the output to the predicted shape
                first_num_rows, first_num_cols = first_output_grid_shape
                second_num_rows, second_num_cols = second_output_grid_shape
                attempts = {
                    "attempt_1": first_output_grid[:first_num_rows, :first_num_cols].tolist(),
                    "attempt_2": second_output_grid[:second_num_rows, :second_num_cols].tolist(),
                }
                task_outputs.append(attempts)
            results[task_id] = task_outputs
        return results

    def evaluate_generations(
        self, generations: dict[str, list], solutions: dict[str, list]
    ) -> dict[str, list]:
        top_1_num_correct_tasks, top_2_num_correct_tasks = 0.0, 0.0
        top_1_num_correct_shapes, top_2_num_correct_shapes = 0.0, 0.0
        top_1_pixel_correctness, top_2_pixel_correctness = 0.0, 0.0
        for task_id, generation_outputs in generations.items():
            num_test_grids = len(generation_outputs)
            for generation, solution in zip(generation_outputs, solutions[task_id]):
                attempt_1 = np.array(generation["attempt_1"])
                attempt_2 = np.array(generation["attempt_2"])
                solution = np.array(solution)
                maybe_top_2_num_correct_shapes = 0
                maybe_top_2_num_correct_tasks = 0
                maybe_top_2_pixel_correctness = 0
                if attempt_1.shape == solution.shape:
                    top_1_num_correct_shapes += 1 / num_test_grids
                    top_1_num_correct_tasks += np.array_equal(attempt_1, solution) / num_test_grids
                    top_1_pixel_correctness += np.mean(attempt_1 == solution).item() / num_test_grids
                    maybe_top_2_num_correct_shapes = 1 / num_test_grids
                    maybe_top_2_num_correct_tasks = np.array_equal(attempt_1, solution) / num_test_grids
                    maybe_top_2_pixel_correctness = np.mean(attempt_1 == solution).item() / num_test_grids
                if attempt_2.shape == solution.shape:
                    maybe_top_2_num_correct_shapes = 1 / num_test_grids
                    maybe_top_2_num_correct_tasks = max(
                        np.array_equal(attempt_2, solution) / num_test_grids, maybe_top_2_num_correct_tasks
                    )
                    maybe_top_2_pixel_correctness = max(
                        np.mean(attempt_2 == solution).item() / num_test_grids, maybe_top_2_pixel_correctness
                    )
                top_2_num_correct_shapes += maybe_top_2_num_correct_shapes
                top_2_num_correct_tasks += maybe_top_2_num_correct_tasks
                top_2_pixel_correctness += maybe_top_2_pixel_correctness
        metrics = {
            "top_1_shape_accuracy": top_1_num_correct_shapes / len(generations),
            "top_1_accuracy": top_1_num_correct_tasks / len(generations),
            "top_1_pixel_correctness": top_1_pixel_correctness / len(generations),
            "top_2_shape_accuracy": top_2_num_correct_shapes / len(generations),
            "top_2_accuracy": top_2_num_correct_tasks / len(generations),
            "top_2_pixel_correctness": top_2_pixel_correctness / len(generations),
        }
        return metrics

    def pad_and_crop_json(self, x: chex.Array, x_shape: chex.Array) -> tuple[chex.Array, chex.Array]:
        if x.shape[0] > self.max_rows or x.shape[1] > self.max_cols:
            if not self.debug_msg:
                print(
                    f"WARNING: cropping json grids to {self.max_rows, self.max_cols}. "
                    "The outputs cannot be trusted."
                )
                self.debug_msg = True
            x = x[: self.max_rows, : self.max_cols]
            # clamp the shape to the max values
            x_shape = (min(x_shape[0], self.max_rows), min(x_shape[1], self.max_cols))
        x = jnp.pad(x, ((0, self.max_rows - x.shape[0]), (0, self.max_cols - x.shape[1])))
        return x, x_shape
