"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Create an Orbax CheckpointManager with specified (Async or not) Checkpointer."""
import os
from typing import Optional, Union
from etils import epath
import orbax.checkpoint
from orbax.checkpoint.logging import abstract_logger, cloud_logger, standard_logger, composite_logger
from orbax.checkpoint import pytree_checkpoint_handler, type_handlers
from orbax.checkpoint.checkpoint_manager import CheckpointManager, CheckpointManagerOptions, PyTree
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager
import jax
import numpy as np
import grain.python as grain
from flax.traverse_util import flatten_dict, unflatten_dict

import max_logging
from multihost_dataloading import MultiHostDataLoadIterator
from flax.training import orbax_utils, train_state

PyTreeCheckpointHandler = pytree_checkpoint_handler.PyTreeCheckpointHandler
LocalCheckpointOptions = emergency_checkpoint_manager.LocalCheckpointOptions
PersistentCheckpointOptions = (
    emergency_checkpoint_manager.PersistentCheckpointOptions
)


def create_orbax_checkpoint_manager(config):
  """Returns specified Orbax (async or not) CheckpointManager or None if checkpointing is disabled."""
  if not config.enable_checkpointing:
    max_logging.log("Checkpointing disabled, not creating checkpoint manager.")
    return None
  max_logging.log("Creating checkpoint manager...")
  p = epath.Path(config.checkpoint_dir)

  if config.dataset_type=='c4-array_record':
    item_names = ('state', 'iter')
  else:
    item_names = ('state',)
  load_ocdbt = getattr(config, 'load_ocdbt', False)
  items = {
        "state": orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler(use_ocdbt=load_ocdbt)), # lsp
    }
  mngr = CheckpointManager(
      p,
      items,
      # item_names = item_names, # do not give with items on same time
      options = CheckpointManagerOptions(
          enable_background_delete=getattr(config, 'enable_background_delete', True), # lsp
          max_to_keep=getattr(config, 'max_to_keep', None), # lsp
          create=True,
          save_interval_steps=config.checkpoint_period,
          enable_async_checkpointing=config.async_checkpointing,
          # should_save_fn=checkpoint_should_save_fn,
          keep_period=getattr(config, 'keep_period', 1000), # lsp: step / keep_period would not be deleted
      )
  )
  max_logging.log("Checkpoint manager created!")
  return mngr


def create_orbax_emergency_checkpoint_manager(
    local_checkpoint_dir: str,
    persistent_checkpoint_dir: str,
    global_mesh: jax.sharding.Mesh,
    abstract_state: PyTree,
    local_save_interval_steps: int,
    persistent_save_interval_steps: int,
):
  """Returns an emergency checkpoint."""
  max_logging.log("Creating emergency checkpoint manager...")

  local_registry = type_handlers.create_type_handler_registry(
      (
          jax.Array,
          type_handlers.ArrayHandler(primary_host=None, replica_id=None),
      ),
  )

  local_checkpoint_handler = PyTreeCheckpointHandler(
      use_ocdbt=True,
      use_zarr3=True,
      primary_host=None,
      type_handler_registry=local_registry,
  )

  options = emergency_checkpoint_manager.CheckpointManagerOptions(
      local=LocalCheckpointOptions(
          save_interval_steps=local_save_interval_steps
      ),
      persistent=PersistentCheckpointOptions(
          save_interval_steps=persistent_save_interval_steps
      ),
  )

  emergency_mngr = emergency_checkpoint_manager.CheckpointManager(
      local_checkpoint_dir,
      epath.Path(persistent_checkpoint_dir),
      global_mesh=global_mesh,
      abstract_state=abstract_state,
      options=options,
      local_state_handler=local_checkpoint_handler,
  )

  max_logging.log("Emergency checkpoint manager created!")
  return emergency_mngr


def _find_idx(array: np.ndarray, replica_axis_idx: int):
  """Returns the index along given dimension that the current host belongs to."""
  idx = None
  for idx, val in np.ndenumerate(array):
    if val.process_index == jax.process_index():
      break
  return idx[replica_axis_idx]


def _replica_devices(device_array: np.ndarray, replica_axis_idx: int):
  """Returns the devices from the replica that current host belongs to.

  Replicas are assumed to be restricted to the first axis.

  Args:
    device_array: devices of the mesh that can be obtained by mesh.devices()
    replica_axis_idx: axis dimension along which replica is taken

  Returns:
    devices inside the replica that current host is in
  """
  idx = _find_idx(device_array, replica_axis_idx)
  replica_result = np.take(device_array, idx, axis=replica_axis_idx)
  return np.expand_dims(replica_result, axis=replica_axis_idx)


def create_load_checkpoint_manager(checkpoint_dir, load_ocdbt):
  options = orbax.checkpoint.CheckpointManagerOptions()
  item = {
      "state": orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler(use_ocdbt=load_ocdbt))
  }
  ocdbt_max_mngr = orbax.checkpoint.CheckpointManager(checkpoint_dir, item, options)
  return ocdbt_max_mngr
  
  
def print_state_shape_device(state):
  if not isinstance(state, dict):
    state = state.params
  elif hasattr(state, 'params'):
    state = state.params
  for k, v in flatten_dict(state).items():
    k = '.'.join(k)
    print(k, v.shape)
  is_on_devices = v.devices() if hasattr(v, 'devices') else 'cpu'
  print(f'is_on_devices: {is_on_devices}')
  

def load_state_if_possible(checkpoint_manager: CheckpointManager,
                           data_iterator: Union[MultiHostDataLoadIterator, None],
                           load_parameters_path: str,
                           load_full_state_path: str,
                           abstract_unboxed_pre_state: train_state.TrainState,
                           enable_single_replica_ckpt_restoring: Optional[bool] = False,
                           dataset_type: Optional[str] = "tfds",
                           checkpoint_dir: Optional[str] = None,
                           only_eval: bool = False,
                           load_ocdbt: bool = False,
                           ):

  def extract_path_step(path):
    checkpoint_step = os.path.basename(path)
    try:
      checkpoint_step = int(checkpoint_step)
    except Exception as error:
      error = f'Error: {error}, please check whether ‘load_parameters_path’ endswith step number'
      raise ValueError(error)
    return checkpoint_step
    
  checkpoint_dir = epath.Path(checkpoint_dir)

  checkpoint_manager = create_load_checkpoint_manager(checkpoint_dir, load_ocdbt)

   # 如果存在meta dict且load_full_state_path和load_parameters_path为空则自动加载最新模型
  meta_dict = data_iterator.meta_dict
  checkpoint_step = meta_dict.get('checkpoint_step', None)
    
  if load_full_state_path:
    checkpoint_step = extract_path_step(load_full_state_path)
  elif load_parameters_path:
    checkpoint_step = extract_path_step(load_parameters_path)
   
  if only_eval:
    load_full_state_path = None
    load_parameters_path = checkpoint_dir / str(checkpoint_step)
    max_logging.log(f'only eval mode, start to load parameters. load_parameters_path is ’{load_parameters_path}‘')

  if load_full_state_path:
    max_logging.log(f"restoring state from {load_full_state_path=}")
    state = checkpoint_manager.restore(checkpoint_step, items={"state": abstract_unboxed_pre_state})
    print_state_shape_device(state['state'])
    return state['state'], None

  elif load_parameters_path:
    max_logging.log(f"restoring params from {load_parameters_path=}")
    params_shapedtype = abstract_unboxed_pre_state['params'] if isinstance(abstract_unboxed_pre_state, dict) else abstract_unboxed_pre_state.params
    print(f'params_shapedtype: {params_shapedtype}')
    # 最新版本orbax-checkpoint时，如果模型文件中存在_sharding，当_sharding的shard shape和当前的shard shape不一致时，会报错
    # 有2种解决方案，1、基于当前的mesh shape重新写一个_sharding文件对其进行覆盖；2、手动删除原始_sharding文件
    state = checkpoint_manager.restore(checkpoint_step, items={"state": {"params": params_shapedtype}})
    print_state_shape_device(state['state'])
    return None, state['state']  # params: {'params'} 2

  elif checkpoint_step is not None:
    max_logging.log(f"restoring params from ’{checkpoint_dir}‘ checkpoint_step: {checkpoint_step}")
    checkpoint_dir = checkpoint_dir / str(checkpoint_step) / 'state'
    ckptr = orbax.checkpoint.StandardCheckpointer()
    restored = ckptr.restore(checkpoint_dir, args=orbax.checkpoint.args.StandardRestore(abstract_unboxed_pre_state))
    print_state_shape_device(restored)
    return  {'state': restored}, None
    
  else:
    max_logging.log("No existing checkpoints found, not restoring checkpoint.")
    return None, None


def setup_checkpoint_logger(config) -> composite_logger.CompositeLogger | None:
  """Setup checkpoint logger.
  Args:
    config
  Returns:
    CompositeLogger
  """
  orbax_cloud_logger = None
  orbax_standard_logger = None
  max_logging.log("Setting up checkpoint logger...")
  if config.enable_checkpoint_cloud_logger:
    logger_name = f"checkpoint_{config.run_name}"
    options = cloud_logger.CloudLoggerOptions(
        job_name=config.run_name, logger_name=logger_name
    )
    orbax_cloud_logger = cloud_logger.CloudLogger(options=options)
    max_logging.log("Successfully set up checkpoint cloud logger.")

  if config.enable_checkpoint_standard_logger:
    orbax_standard_logger = standard_logger.StandardLogger()
    max_logging.log("Successfully set up checkpoint standard logger.")

  orbax_logger = None
  if orbax_cloud_logger is not None and orbax_standard_logger is not None:
    orbax_logger = composite_logger.CompositeLogger(
        orbax_cloud_logger, orbax_standard_logger
    )
    max_logging.log("Successfully set up checkpoint composite logger.")

  return orbax_logger


def load_params_from_path(load_parameters_from_path, abstract_unboxed_params):
  """Load decode params from checkpoint at specified path."""
  assert load_parameters_from_path, "load_parameters_from_path is not defined."
  max_logging.log(f"restoring params from {load_parameters_from_path}")
  ckpt = epath.Path(load_parameters_from_path)
  ckptr = orbax.checkpoint.PyTreeCheckpointer()
  # This is a memory optimization. We don't want to restore the entire checkpoint - only the params.
  # Rather than pass the entire abstract state, which could unnecessarily restore opt_state and such and waste
  # memory, we instead specify here that we are just restoring the params field of the checkpoint
  # (which itself may be a dictionary containing a key named 'params').
  restore_args = orbax.checkpoint.checkpoint_utils.construct_restore_args(abstract_unboxed_params)
  restored = ckptr.restore(
    ckpt,
    item={"params": abstract_unboxed_params},
    transforms={},
    restore_args={"params": restore_args}
    )
  return restored["params"]


def save_params_to_path(checkpoint_dir, params):
  """Save decode params in checkpoint at specified path."""
  assert checkpoint_dir, "checkpoint_dir is not defined."
  orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  save_args = orbax_utils.save_args_from_target({"params":params})
  orbax_checkpointer.save(
    checkpoint_dir,
    {"params":params},
    save_args=save_args,
    force=True
    )
  print(f"Quantized params checkpoint saved at: {checkpoint_dir}")