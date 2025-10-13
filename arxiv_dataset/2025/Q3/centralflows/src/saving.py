from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Literal

import h5py
import numpy as np
from torch.utils._pytree import tree_map

from .utils import flatten_dict, to_numpy, unflatten_dict
from .processes import Process

@dataclass
class LoadOptions:
    # path of npz file to load from
    path: Optional[str] = None
    
    # If set to None (default), load each new process from the corresponding 
    # old process (e.g central loads from central, discrete from discrete, etc).
    # But if set to a value, load all processes from that specified old process.
    from_process: Optional[Literal["central", "discrete", "midpoint", "stable"]] = None
    

@dataclass
class Checkpointer:
    """Handles checkpoint saving and loading.
    
    The user can tell the checkpointer to save checkpoints at a certain frequency,
    or can pass in an explicit list of steps to checkpoint at, or both.
    """
    
    frequency: int = -1                              # should save checkpoints at this frequency
    steps: List[int] = field(default_factory=list)   # should save checkpoints on these steps

    def init(self, folder):
        """Initialize the checkpointer.
        
        Args:
          folder (str or Path): the folder to store checkpoints in
        """
        self.folder = Path(folder)

    def maybe_checkpoint(self, step: int, processes: List[Process]):
        """Save a checkpoint, if appropriate.
        
        Args:
          step (int): the step counter of the checkpoint
          processes (List[Process]): the processes to serialize
        """
        if (self.frequency > 0 and step % self.frequency == 0) or step in self.steps:
            self.folder.mkdir(exist_ok=True)
            data_dict = {k: v.to_dict() for k, v in processes.items()}
            data_dict = tree_map(to_numpy, data_dict)
            data_dict = flatten_dict(data_dict, sep=".")
            data_dict["step"] = step
            np.savez(self.folder / f"checkpoint_{step}.npz", **data_dict)
            
    def load(self, load_options: LoadOptions, processes: List[Process]) -> int:
        """Load from a checkpoint.
        
        Args:
          load_options (LoadOptions): options for loading
          processes (List[Process]) processes to load into
          
        Returns:
          int: the step counter of the checkpoint
          
        The processes are modified in place.
        """
        ckpt = dict(np.load(load_options.path))
        step = ckpt.pop("step")
        ckpt = unflatten_dict(ckpt, sep=".")
        for k, p in processes.items():
            # if from_process is none, load from the corresponding process; else log from `from_process``
            if load_options.from_process is None:
                load_from = ckpt[k]
            else:
                load_from = ckpt[load_options.from_process]
            p.from_dict(load_from)
        return step


class DataSaver:
    """Saves experiment data to disk in HDF5 format.
    
    Usage:
      with DataSaver(path, initial_step, steps) as data_saver
        data_saver.save(step, data)
    
    """
    
    def __init__(self, file, initial_step, total_steps, save_every=1):
        self.f = h5py.File(file, "a", libver="latest")
        self.f.swmr_mode = True
        self.total_steps = total_steps
        self.save_every = save_every
        self.step_counter = 0
        self.f.create_dataset("step", data=np.arange(total_steps) + initial_step)

    def save(self, step, data):
        data = tree_map(to_numpy, data)

        def save_value(path, val):
            if isinstance(val, dict):
                for k, v in val.items():
                    save_value(f"{path}/{k}" if path else k, v)
            elif val is not None:
                if path not in self.f:
                    shape = (self.total_steps,) + val.shape
                    dset = self.f.create_dataset(path, shape, dtype=val.dtype)
                    if val.dtype in [np.float32, np.float64]:
                        dset[:] = np.nan
                    dset[step] = val
                else:
                    dset = self.f[path]
                    if val.shape != dset.shape[1:]:
                        # Shape mismatch detected
                        if len(val.shape) > len(dset.shape) - 1:
                            raise ValueError(
                                f"\tNew data has more dimensions than dataset at {path}"
                            )

                        # Create new dataset with larger shape
                        new_shape = (dset.shape[0],) + tuple(
                            max(s1, s2) for s1, s2 in zip(dset.shape[1:], val.shape)
                        )
                        if new_shape != dset.shape:
                            old_data = dset[:]
                            del self.f[path]
                            dset = self.f.create_dataset(
                                path, new_shape, dtype=val.dtype
                            )
                            dset[:] = np.nan
                            slices = (slice(0, step),) + tuple(
                                slice(0, s) for s in old_data.shape[1:]
                            )
                            dset[slices] = old_data[:step]
                        slices = (slice(0, s) for s in val.shape)
                        dset[(step, *slices)] = val
                    else:
                        dset[step] = val

        save_value("", data)

        if self.step_counter % self.save_every == 0:
            self.f.flush()
        self.step_counter += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.flush()
        self.f.close()
