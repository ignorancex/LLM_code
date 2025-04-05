#!/usr/bin/python3
import os
import subprocess
import sys

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import helpers.commons as commons
from helpers.runner import parse_output

phase_order = ["total", "partition", "partition_1", "partition_r", "partition_s", "partition_2", "join_total", "build",
               "probe"]
MB = 2 ** 20
TUPLE_SIZE = 8
TUPLE_PER_MB = MB/TUPLE_SIZE


def plot_throughput(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)

    data = data[(data["measurement"] == "throughput")]

    data["Throughput in $10^6$ rows/s"] = data["value"]
    data["Divisor"] = data["part"]
    data["Size"] = data["size_r"]

    # c_palette_1 = ["#D56257", "#8e8e8e", "#29292b", "#176582"]
    sns.set_style("ticks")
    sns.set_context("notebook")
    # sns.set_palette(sns.color_palette(c_palette_1))
    sns.set_palette([sns.color_palette("deep")[0], sns.color_palette("deep")[2]])

    plt.figure(figsize=(6, 3.25))
    sns.barplot(data, y="Throughput in $10^6$ rows/s", x="Size", hue="Divisor", errorbar="sd")

    plt.ylim(bottom=0)
    plt.grid(axis="y")
    plt.tight_layout()

    plot_filename = f"../img/{experiment_name}.pdf"
    plt.savefig(plot_filename, dpi=300)
    # plt.show()


def run_join(mode, alg, size_r, size_s, threads, reps, filename_detail, flags: list[str]):
    lock_string = "spin" if "SPIN_LOCK" in flags else "mutex"
    unroll_string = "unrolled" if "UNROLL" in flags else "scalar"

    divisor_string = flags[-1]
    partition_string = divisor_string[divisor_string.index("=")+1:]

    with open(filename_detail, 'a') as f_detail:
        settings = ",".join([mode, lock_string, unroll_string, partition_string, alg, str(size_r), str(size_s), str(threads)]) + ","

        for i in range(reps):
            try:
                print(str(i + 1) + '/' + str(reps) + ': ' + settings[:-1])

                result = subprocess.run(["./app", "-a", alg, "-r", str(size_r), "-s", str(size_s),
                                         "-n", str(threads)],
                                        cwd="../../",
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
                for phase in phase_order:
                    if phase in phases:
                        f_detail.write(settings + phase + "," + str(phases[phase]) + '\n')
            except subprocess.CalledProcessError as e:
                print(f"Failed with exit code {e.returncode}! Stdout:")
                print(e.stdout)
                print("stderr:")
                print(e.stderr)


def compile_and_run(config, filename_detail, flags: None | list[str] = None):
    commons.remove_file(filename_detail)
    commons.init_file(filename_detail, "mode,lock,unroll,part,alg,size_r,size_s,threads,measurement,value\n")
    last_enclave_size = "0"
    last_flag_set = None
    for mode in config['modes']:
        for flag_set in flags:
            if mode == "native":
                commons.compile_app(mode, enclave_config_file=f'Enclave/Enclave1GB.config.xml',
                                    flags=flag_set)
            for size in config["sizes"]:
                if mode == "sgx":
                    # recompile with another enclave size if necessary
                    enclave_size = commons.find_fitting_enclave(size, "MWAY" in config["algorithms"])
                    if enclave_size != last_enclave_size or flag_set != last_flag_set:
                        commons.compile_app(mode, enclave_config_file=f'Enclave/Enclave{enclave_size}.config.xml',
                                            flags=flag_set)
                        last_enclave_size = enclave_size
                        last_flag_set = flag_set
                for algorithm in config["algorithms"]:
                    for threads in config["threads"]:
                        run_join(mode, algorithm, size[0], size[1], threads, config['reps'], filename_detail, flag_set)


def main():
    experiment_name = "partition-size-exp"

    if len(sys.argv) < 2:
        print("Please specify run/plot/both")
        exit(-1)

    execution_command = sys.argv[1]

    if len(sys.argv) < 3:
        print(f"Experiment name defaults to {experiment_name}.")
    else:
        experiment_name = sys.argv[2]

    filename_detail = f"../data/{experiment_name}.csv"

    config = dict()
    config["experiment"] = True
    config["threads"] = [16]
    config["algorithms"] = ["RHO"]
    config["reps"] = 3
    config["modes"] = ["sgx"]
    config["flags"] = [["UNROLL", "FORCE_2_PHASES", "SPIN_LOCK"] + [f"CACHE_DIVISOR={div}"] for div in
                       [1, 2, 4, 8, 16]]

    config["sizes"] = [(100 * TUPLE_PER_MB, 400 * TUPLE_PER_MB),
                       (200 * TUPLE_PER_MB, 400 * TUPLE_PER_MB),
                       (300 * TUPLE_PER_MB, 400 * TUPLE_PER_MB),
                       (400 * TUPLE_PER_MB, 400 * TUPLE_PER_MB)]

    if execution_command in ["run", "both"]:
        compile_and_run(config, filename_detail, config["flags"])
    if execution_command in ["plot", "both"]:
        pass
        # plot_phases(filename_detail, experiment_name, "RHO", 16)
        plot_throughput(filename_detail, experiment_name)


if __name__ == '__main__':
    main()
