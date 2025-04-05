import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional

import helpers.commons as commons

TPCH_PHASE_ORDER = ["total", "selection", "selection1", "selection2", "selection3", "join", "copy", "join1", "join2",
                    "join3"]


def parse_tpch_output(stdout):
    measurements = {}
    for line in stdout.splitlines():
        measurement = ""
        if "QueryTimeTotal (us)" in line:
            measurement = "total"
        elif "QueryTimeSelection (us)" in line:
            measurement = "selection"
        elif "QueryTimeSelection 1 (us)" in line:
            measurement = "selection1"
        elif "QueryTimeSelection 2 (us)" in line:
            measurement = "selection2"
        elif "QueryTimeSelection 3 (us)" in line:
            measurement = "selection3"
        elif "QueryTimeJoin (us)" in line:
            measurement = "join"
        elif "QueryTimeCopy (us)" in line:
            measurement = "copy"
        elif "QueryTimeJoin 1 (us)" in line:
            measurement = "join1"
        elif "QueryTimeJoin 2 (us)" in line:
            measurement = "join2"
        elif "QueryTimeJoin 3 (us)" in line:
            measurement = "join3"
        elif "QueryThroughput (M rec/s)" in line:
            measurement = "throughput"
        if measurement == "throughput":
            measurements[measurement] = float(re.findall("\d+\.\d+", line)[1])
        elif measurement in ["join1", "join2", "join3", "selection1", "selection2", "selection3"]:
            measurements[measurement] = int(re.findall(r'\d+', line)[4])
        elif measurement != "":
            measurements[measurement] = int(re.findall(r'\d+', line)[3])
    return measurements


def run_tpch(exe: Tuple[str, str], alg: str, query: int, sf: int, threads: int, reps: int, filename_detail: str,
             flags: list[str]):
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
        settings = ",".join([str(query), str(sf), lock_string, unroll_string, sort_string, partition_string, alg,
                             str(threads)]) + ","

        for i in range(reps):
            try:
                print(str(i + 1) + '/' + str(reps) + ': ' + settings[:-1])
                result = subprocess.run([f"./{exe[1]}", "-a", alg, "-q", str(query), "-s", str(sf), "-n", str(threads)],
                                        cwd=f"../../{exe[0]}/",
                                        env=os.environ | {"SGX_DBG_OPTIN": "1"},
                                        capture_output=True,
                                        text=True,
                                        encoding="UTF-8",
                                        check=True)

                print(result.stdout)
                measurements = parse_tpch_output(result.stdout)
                print(measurements)

                f_detail.write(settings + 'throughput' + "," + str(round(measurements["throughput"], 4)) + '\n')
                for phase in TPCH_PHASE_ORDER:
                    if phase in measurements:
                        f_detail.write(settings + phase + "," + str(measurements[phase]) + '\n')
            except subprocess.CalledProcessError as e:
                print(f"Failed with exit code {e.returncode}! Stdout:")
                print(e.stdout)
                print("stderr:")
                print(e.stderr)


def compile_and_run_tpch(config, filename_detail, flags: None | list[str] = None):
    if "flags" in config:
        flags = config["flags"]
    else:
        if flags is None:
            flags = [["SPIN_LOCK"]]
        else:
            flags = [flags]

    commons.remove_file(filename_detail)
    commons.init_file(filename_detail, "query,sf,lock,unroll,sort,part,alg,threads,measurement,value\n")
    for flag_set in flags:
        enclave_size = "4GB"
        exe = commons.compile_app("tpch", enclave_size=enclave_size, flags=flag_set)
        if not exe:
            return
        for size in config["sizes"]:
            for query in config["queries"]:
                for algorithm in config["algorithms"]:
                    for threads in config["threads"]:
                        run_tpch(exe, algorithm, query, size, threads, config['reps'], filename_detail, flag_set)


@dataclass
class ExperimentConfig:
    modes: list[str]
    flags: list[list[str]]
    queries: list[int]
    scale_factors: list[int]
    algorithms: list[str]
    threads: list[int]
    repetitions: int

    def run_count(self):
        return (len(self.modes) * len(self.flags) * len(self.queries) * len(self.scale_factors) * len(self.algorithms)
                * len(self.threads) * self.repetitions)


@dataclass
class TPCHRunConfig:
    mode: str
    flags: list[str]
    query: int
    scale_factor: int
    alg: str
    threads: int

    @classmethod
    def header(cls) -> str:
        return "mode,flags,query,scale_factor,alg,threads"

    def as_comma_separated(self) -> str:
        return ",".join([self.mode, " ".join(self.flags), str(self.query), str(self.scale_factor), self.alg,
                         str(self.threads)])


def run_tpch_simple_flags(exe: Tuple[str, str], config: TPCHRunConfig, reps: int, filename_detail: str):
    environment_vars = {"SGX_DBG_OPTIN": "1", "MALLOC_ARENA_MAX": "16", "MALLOC_TOP_PAD_": "4294967296",
                        "MALLOC_TRIM_THRESHOLD_": "-1"}
    with open(filename_detail, 'a') as f_detail, open(f"{filename_detail}-full.txt", 'a') as f_full:
        settings = config.as_comma_separated()

        for i in range(reps):
            try:
                time = datetime.now().astimezone().replace(microsecond=0).isoformat()
                experiment_setting_announcement = f"[{time}] {i + 1}/{reps}: {settings}"
                print(experiment_setting_announcement)
                f_full.write(experiment_setting_announcement + '\n')
                result = subprocess.run(
                    [f"./{exe[1]}", "-a", config.alg, "-q", str(config.query), "-s", str(config.scale_factor),
                     "-n", str(config.threads)],
                    cwd=f"../../{exe[0]}/",
                    env=os.environ | environment_vars,
                    capture_output=True,
                    text=True,
                    encoding="UTF-8",
                    check=True)

                print(result.stdout)
                f_full.write(result.stdout)
                measurements = parse_tpch_output(result.stdout)
                print(measurements)
                f_full.write(repr(measurements) + '\n')

                f_detail.write(",".join([settings, 'throughput', str(round(measurements["throughput"], 4))]) + "\n")
                for phase in TPCH_PHASE_ORDER:
                    if phase in measurements:
                        f_detail.write(",".join([settings, phase, str(measurements[phase])]) + '\n')
            except subprocess.CalledProcessError as e:
                print(f"Failed with exit code {e.returncode}! Stdout:")
                print(e.stdout)
                print("stderr:")
                print(e.stderr)
                f_full.write(f"Failed with exit code {e.returncode}! Stdout:\n")
                f_full.write(e.stdout)
                f_full.write(f"stderr:\n")
                f_full.write(e.stderr)


def compile_and_run_simple_flags(config: ExperimentConfig, filename_detail: str):
    print(f"Running {config.run_count()} configurations.")
    flags = config.flags if config.flags else [[]]
    cpms = 2200 if os.uname()[1].startswith("sgx06") else None

    commons.remove_file(filename_detail)
    commons.remove_file(f"{filename_detail}-full.txt")
    commons.init_file(filename_detail, f"{TPCHRunConfig.header()},measurement,value\n")

    for flag_set in flags:
        for mode in config.modes:
            for scale_factor in config.scale_factors:
                enclave_size = "4GB" if scale_factor <= 10 else "16GB"
                exe = commons.compile_app(mode, enclave_size=enclave_size, flags=flag_set, cpms=cpms)
                if not exe:
                    return
                for algorithm in config.algorithms:
                    for query in config.queries:
                        for threads in config.threads:
                            run_config = TPCHRunConfig(
                                mode,
                                flag_set,
                                query,
                                scale_factor,
                                algorithm,
                                threads
                            )
                            run_tpch_simple_flags(exe, run_config, config.repetitions, filename_detail)
