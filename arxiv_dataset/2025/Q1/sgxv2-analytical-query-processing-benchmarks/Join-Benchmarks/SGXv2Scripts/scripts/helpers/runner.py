import os
import re
import subprocess
from datetime import datetime
from typing import Tuple, Optional
from dataclasses import dataclass, field

import helpers.commons as commons

TEEBENCH_PHASE_ORDER = ["total", "partition", "partition_1", "partition_r", "partition_s", "partition_2",
                        "partition_2_h", "partition_2_c", "join_total", "build", "probe", "build_miss", "probe_miss"]


def parse_output(stdout):
    phases = {}
    throughput = 0
    for line in stdout.splitlines():
        # find throughput
        if "Throughput" in line:
            throughput = float(re.findall("\d+\.\d+", line)[1])
        # find times
        elif "PHT build LLC-misses" in line:
            phases["build_miss"] = float(re.findall("\d+\.\d+", line)[1])
        elif "PHT probe LLC-misses" in line:
            phases["probe_miss"] = float(re.findall("\d+\.\d+", line)[1])
        else:
            phase = ""
            if "Total Join Time (cycles)" in line:
                phase = "total"
            elif "Partition Overall (cycles)" in line:
                phase = "partition"
            elif "Partition Pass One (cycles)" in line:
                phase = "partition_1"
            elif "Partition One Hist (cycles)" in line:
                phase = "partition_r"
            elif "Partition One Copy (cycles)" in line:
                phase = "partition_s"
            elif "Partition Pass Two (cycles)" in line:
                phase = "partition_2"
            elif "Partition Two Hist (cycles)" in line:
                phase = "partition_2_h"
            elif "Partition Two Copy (cycles)" in line:
                phase = "partition_2_c"
            elif "Build+Join Overall (cycles)" in line:
                phase = "join_total"
            elif "Build (cycles)" in line:
                phase = "build"
            elif "Join (cycles)" in line:
                phase = "probe"
            if phase != "":
                value = int(re.findall(r'\d+', line)[
                                -2])  # yes, this must be -2 because of control sequences, probably for colors.
                phases[phase] = value

    return throughput, phases


def run_join(exe: Tuple[str, str], mode: str, alg: str, size_r: int, size_s: int, threads: int, reps: int,
             filename_detail: str, flags: list[str]):
    lock_string = "spin" if "SPIN_LOCK" in flags else "mutex"
    unroll_string = "unrolled" if "UNROLL" in flags else "scalar"
    sort_string = "simd" if "SIMD_SORT" in flags else "scalar"

    small_partitions = "SMALL_PARTITIONS" in flags
    max_partitions = "MAX_PARTITIONS" in flags
    const = "CONSTANT_RADIX_BITS" in flags

    match (small_partitions, max_partitions, const):
        case (True, False, False):
            partition_string = "small"
        case (False, True, False):
            partition_string = "max"
        case (False, False, True):
            partition_string = "const"
        case (False, False, False):
            partition_string = "normal"
        case _:
            print("Impossible flag combination!")
            return

    with open(filename_detail, 'a') as f_detail:
        settings = ",".join([mode, lock_string, unroll_string, sort_string, partition_string, alg, str(size_r),
                             str(size_s), str(threads)]) + ","

        for i in range(reps):
            try:
                print(str(i + 1) + '/' + str(reps) + ': ' + settings[:-1])
                result = subprocess.run(
                    [f"./{exe[1]}", "-a", alg, "-r", str(size_r), "-s", str(size_s), "-n", str(threads)],
                    cwd=f"../../{exe[0]}/",
                    env=os.environ | {"SGX_DBG_OPTIN": "1"},
                    capture_output=True,
                    text=True,
                    encoding="UTF-8",
                    check=True)

                print(result.stdout)
                throughput, phases = parse_output(result.stdout)

                print('Throughput = ' + str(throughput) + ' M [rec/s]')
                print(phases)

                f_detail.write(settings + 'throughput' + "," + str(round(throughput, 2)) + '\n')
                for phase in TEEBENCH_PHASE_ORDER:
                    if phase in phases:
                        f_detail.write(settings + phase + "," + str(phases[phase]) + '\n')
            except subprocess.CalledProcessError as e:
                print(f"Failed with exit code {e.returncode}! Stdout:")
                print(e.stdout)
                print("stderr:")
                print(e.stderr)


def compile_and_run(config, filename_detail, flags: None | list[str] = None):
    if "flags" in config:
        flags = config["flags"]
    else:
        if flags is None:
            flags = [["SPIN_LOCK"]]
        else:
            flags = [flags]

    commons.remove_file(filename_detail)
    commons.init_file(filename_detail, "mode,lock,unroll,sort,part,alg,size_r,size_s,threads,measurement,value\n")
    last_enclave_size = "0"
    last_flag_set = None
    exe: Optional[Tuple[str, str]] = None
    for mode in config['modes']:
        for flag_set in flags:
            if mode == "native":
                exe = commons.compile_app(mode, flags=flag_set)
                if not exe:
                    return
            for size in config["sizes"]:
                if mode == "sgx":
                    # recompile with another enclave size if necessary
                    enclave_size = commons.find_fitting_enclave(size, "MWAY" in config["algorithms"], False)
                    if enclave_size != last_enclave_size or flag_set != last_flag_set:
                        exe = commons.compile_app(mode, enclave_size=enclave_size, flags=flag_set)
                        if not exe:
                            return
                        last_enclave_size = enclave_size
                        last_flag_set = flag_set
                for algorithm in config["algorithms"]:
                    for threads in config["threads"]:
                        if not exe:
                            raise ValueError("Trying to run a non-existent executable")
                        run_join(exe, mode, algorithm, size[0], size[1], threads, config['reps'], filename_detail,
                                 flag_set)


@dataclass
class ExperimentConfig:
    modes: list[str]
    flags: list[list[str]]
    sizes: list[Tuple[int, int]]
    algorithms: list[str]
    threads: list[int]
    materialize: list[bool]
    repetitions: int
    init_core: list[int] = field(default_factory=lambda: list(range(1)))
    dynamic_enclave: list[bool] = field(default_factory=lambda: [False].copy())
    mitigation: list[bool] = field(default_factory=lambda: [False].copy())
    skew: list[float] = field(default_factory=lambda: list(range(1)))

    def run_count(self):
        return (len(self.modes) * len(self.flags) * len(self.sizes) * len(self.algorithms) * len(self.threads)
                * len(self.materialize) * self.repetitions * len(self.init_core) * len(self.dynamic_enclave)
                * len(self.mitigation) * len(self.skew))


@dataclass
class JoinRunConfig:
    mode: str
    flags: list[str]
    alg: str
    size_r: int
    size_s: int
    threads: int
    materialize: bool
    init_core: int = 0
    dynamic_enclave: bool = False
    mitigation: bool = False
    skew: float = 0.0

    @classmethod
    def header(cls) -> str:
        return "mode,flags,alg,materialize,threads,size_r,size_s,init_core,dynamic_enclave,mitigation,skew"

    def as_comma_separated(self) -> str:
        return ",".join([self.mode, " ".join(self.flags), self.alg, str(self.materialize), str(self.threads),
                         str(self.size_r), str(self.size_s), str(self.init_core), str(self.dynamic_enclave),
                         str(self.mitigation), str(self.skew)])


def run_join_simple_flags(exe: Tuple[str, str], config: JoinRunConfig, reps: int, filename_detail: str,
                          total_runs: int = 0, total_runs_start: int = 0):
    with open(filename_detail, 'a') as f_detail, open(f"{filename_detail}-full.txt", 'a') as f_full:
        settings = config.as_comma_separated()

        materialize_flag = ["-m"] if config.materialize else []
        mitigation_flag = ["--mitigation"] if config.mitigation else []

        for i in range(reps):
            try:
                time = datetime.now().astimezone().replace(microsecond=0).isoformat()
                total_run_counter_string = f" {total_runs_start + i + 1}/{total_runs} " if total_runs else " "
                local_run_counter_string = f"{i + 1}/{reps}"
                experiment_setting_announcement = f"[{time}]{total_run_counter_string}{local_run_counter_string}: {settings}"
                print(experiment_setting_announcement)
                f_full.write(experiment_setting_announcement + '\n')
                result = subprocess.run(
                    [f"./{exe[1]}", "-a", config.alg, "-r", str(config.size_r), "-s", str(config.size_s),
                     "-n", str(config.threads), "-c", str(config.init_core), "-z",
                     str(config.skew)] + materialize_flag + mitigation_flag,
                    cwd=f"../../{exe[0]}/",
                    env=os.environ | {"SGX_DBG_OPTIN": "1"},
                    capture_output=True,
                    text=True,
                    encoding="UTF-8",
                    check=True)

                print(result.stdout)
                f_full.write(result.stdout + '\n')
                throughput, phases = parse_output(result.stdout)

                print('Throughput = ' + str(throughput) + ' M [rec/s]')
                f_full.write('Throughput = ' + str(throughput) + ' M [rec/s]' + '\n')
                print(phases)
                f_full.write(repr(phases) + '\n')

                f_detail.write(",".join([settings, 'throughput', str(round(throughput, 4))]) + "\n")
                for phase in TEEBENCH_PHASE_ORDER:
                    if phase in phases:
                        f_detail.write(",".join([settings, phase, str(phases[phase])]) + '\n')
            except subprocess.CalledProcessError as e:
                print(f"Failed with exit code {e.returncode}! Stdout:")
                print(e.stdout)
                print("stderr:")
                print(e.stderr)
                f_full.write(f"Failed with exit code {e.returncode}! Stdout:\n")
                f_full.write(e.stdout + '\n')
                f_full.write(f"stderr:\n")
                f_full.write(e.stderr + '\n')


def compile_and_run_simple_flags(config: ExperimentConfig, filename_detail: str):
    run_count = config.run_count()
    print(f"Running {run_count} configurations.")
    flags = config.flags if config.flags else [[]]
    cpms = 2200 if os.uname()[1].startswith("sgx06") else None

    commons.remove_file(filename_detail)
    commons.remove_file(f"{filename_detail}-full.txt")
    commons.init_file(filename_detail, f"{JoinRunConfig.header()},measurement,value\n")

    total_run_counter = 0

    last_enclave_size = "0"
    last_flag_set = None
    last_dyn_enclave = None
    exe: Optional[Tuple[str, str]] = None
    for flag_set in flags:
        for mode in config.modes:
            if mode == "native":
                exe = commons.compile_app(mode, flags=flag_set, cpms=cpms)
                if not exe:
                    return
            for size in config.sizes:
                for dyn_enclave in config.dynamic_enclave:
                    if mode == "sgx":
                        # recompile with another enclave size if necessary
                        enclave_size = commons.find_fitting_enclave(size, "MWAY" in config.algorithms,
                                                                    any(config.materialize))
                        if enclave_size != last_enclave_size or flag_set != last_flag_set or dyn_enclave != last_dyn_enclave:
                            exe = commons.compile_app(mode, enclave_size=enclave_size, flags=flag_set, cpms=cpms,
                                                      dynamic_enclave=dyn_enclave)
                            if not exe:
                                return
                            last_enclave_size = enclave_size
                            last_flag_set = flag_set
                    for mitigation in config.mitigation:
                        for algorithm in config.algorithms:
                            for threads in config.threads:
                                for skew in config.skew:
                                    for materialize in config.materialize:
                                        for init_core in config.init_core:
                                            if not exe:
                                                raise ValueError("Trying to run a non-existent executable")
                                            run_config = JoinRunConfig(mode, flag_set, algorithm, size[0], size[1],
                                                                       threads, materialize, init_core, dyn_enclave,
                                                                       mitigation, skew)
                                            run_join_simple_flags(exe, run_config, config.repetitions, filename_detail,
                                                                  run_count, total_run_counter)
                                            total_run_counter += config.repetitions
