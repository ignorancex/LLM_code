from collections import defaultdict, deque

import torch
import wandb
from wandb.apis.public import Run


class Metrics:
    def __init__(self, max_len: int | None = None):
        self.history = defaultdict(lambda: deque(maxlen=max_len))

    def log(self, **kwargs):
        for k, v in kwargs.items():
            self.history[k].append(v)

    def clear(self):
        self.history.clear()

    def collect(self, *keys: str, clear: bool = True) -> tuple[list, ...]:
        if clear:
            res = tuple(list(self.history.pop(k)) for k in keys)
        else:
            res = tuple(list(self.history[k]) for k in keys)
        return res

    def asdict(self) -> dict[str, list]:
        return {k: list(v) for k, v in self.history.items()}


global_metrics = Metrics()


class CudaTimer:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *exc):
        self.end.record()
        return False

    def elapsed(self):
        torch.cuda.synchronize()
        return self.start.elapsed_time(self.end)


def next_multiple(x: int, base: int) -> int:
    return (x + base - 1) // base * base


def load_runs(entity: str, project: str, run_ids: list[int]) -> list[Run]:
    api = wandb.Api()

    runs = api.runs(
        path=f"{entity}/{project}",
        filters={
            "display_name": {
                "$regex": rf"^[a-zA-Z]+-[a-zA-Z]+-({'|'.join(map(str, run_ids))})$"
            }
        },
    )

    return runs
