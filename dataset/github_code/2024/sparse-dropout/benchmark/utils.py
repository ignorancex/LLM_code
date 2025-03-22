import time
from collections import defaultdict
from typing import Callable, Iterator, TypeAlias

import cuda.cudart
import pandas as pd
import torch

BenchmarkFn: TypeAlias = Callable[[], Iterator[str]]


def make_tensor(*size: int, row_major: bool = True, requires_grad: bool = True):
    if row_major:
        return torch.randn(
            size, device="cuda", dtype=torch.float16, requires_grad=requires_grad
        )
    else:
        return torch.randn(
            size[::-1], device="cuda", dtype=torch.float16, requires_grad=requires_grad
        ).T


class GpuTimer:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()  # type: ignore
        return self

    def __exit__(self, *args):
        self.end.record()  # type: ignore
        torch.cuda.synchronize()

    def elapsed_time(self) -> float:
        return self.start.elapsed_time(self.end) / 1000


class Benchmarker:
    def __init__(
        self, fns: dict[str, BenchmarkFn], warmup_reps: int = 1, duration: float = 1
    ) -> None:
        """Benchmarker for comparing multiple functions.

        Args:
            fns (dict[str, BenchmarkFn]): Dictionary of function names and their corresponding benchmark functions.
                Yields strings representing the name of each breakpoint.
            warmup (int, optional): Number of warmup reps for each function. Defaults to 1.
            rep (float, optional): Number of seconds to run the benchmark. Defaults to 1.
        """
        self.fns = fns
        self.warmup_reps = warmup_reps
        self.duration = duration

        _, l2_size = cuda.cudart.cudaDeviceGetAttribute(
            cuda.cudart.cudaDeviceAttr.cudaDevAttrL2CacheSize, 0
        )
        self.cache = torch.empty(l2_size, dtype=torch.int8, device="cuda")

        self.timings = {}  # fn_name -> breakpoint_name -> list of times
        for name in self.fns:
            self.timings[name] = defaultdict(list)

    def _warmup(self):
        for _ in range(self.warmup_reps):
            for _, fn in self.fns.items():
                self.cache.zero_()
                for _ in fn():
                    pass

    def _benchmark_once(self):
        torch.cuda._sleep(1_000_000)  # give the host some time to saturate the GPU
        timers = defaultdict(dict)
        for fn_name, fn in self.fns.items():
            self.cache.zero_()
            iterator = iter(fn())
            while True:
                try:
                    with GpuTimer() as timer:
                        breakpoint_name = next(iterator)
                    timers[fn_name][breakpoint_name] = timer
                except StopIteration:
                    break
        # Store the measured times (implicitly synchrnoises)
        for fn_name, breakpoints in timers.items():
            for breakpoint_name, timer in breakpoints.items():
                self.timings[fn_name][breakpoint_name].append(timer.elapsed_time())

    def run(self):
        self._warmup()
        start_time = time.time()
        count = 0
        while time.time() - start_time < self.duration:
            self._benchmark_once()
            count += 1
        print(f"Ran {count} iterations")

    def results(self, **agg_fns) -> pd.DataFrame:
        data = []
        for fn_name, breakpoints in self.timings.items():
            for breakpoint_name, times in breakpoints.items():
                data.append(
                    {
                        "method": fn_name,
                        "breakpoint": breakpoint_name,
                        **{
                            agg_name: agg_fn(times)
                            for agg_name, agg_fn in agg_fns.items()
                        },
                    }
                )
        return pd.DataFrame(data)
