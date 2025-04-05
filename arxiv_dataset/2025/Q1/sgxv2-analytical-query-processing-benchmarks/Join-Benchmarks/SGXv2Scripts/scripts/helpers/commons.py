import errno
import os
import re
import subprocess
from itertools import chain, combinations
from pathlib import Path
from shutil import rmtree
from typing import Optional, Tuple, TypeVar

import matplotlib.pyplot as plt

MB = 2 ** 20
TUPLE_SIZE = 8
TUPLE_PER_MB = MB / TUPLE_SIZE


def escape_ansi(line):
    ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    return ansi_escape.sub('', line)


def configure(flags=None, enclave_size=None, debug=False, cpms=2900, dynamic_enclave=False) -> Optional[str]:
    if flags is None:
        flags = []

    cflags = ";".join(flags)

    if not enclave_size:
        enclave_string = '-DENCLAVE_CONFIG_FILE=Enclave/Enclave.config.xml'
    else:
        enclave_string = f'-DENCLAVE_CONFIG_FILE=Enclave/Enclave{enclave_size}{"Dyn" if dynamic_enclave else ""}.config.xml'

    if debug:
        build_type = "Debug"
    else:
        build_type = "Release"

    build_dir = f"cmake-build-exp-{build_type}-{enclave_size}-{''.join(flags)}{'-dyn' if dynamic_enclave else ''}"

    if not Path(f"../../{build_dir}").is_dir():
        try:
            arguments = ["cmake",
                         "-G", "Ninja",
                         "-DCMAKE_MAKE_PROGRAM=ninja",
                         "-DCMAKE_C_COMPILER=gcc-12",
                         "-DCMAKE_CXX_COMPILER=g++-12",
                         f"-DCMAKE_BUILD_TYPE={build_type}",
                         f"-DCFLAGS={cflags}",
                         f"-DCPMS={cpms}",
                         enclave_string,
                         "-B", build_dir]

            print(f"Configure CMAKE: {' '.join(arguments)}")
            subprocess.check_output(arguments, cwd="../../", stderr=subprocess.STDOUT)
            return build_dir
        except subprocess.CalledProcessError as e:
            print(f"Failed with exit code {e.returncode}! Stdout:")
            print(e.stdout.decode("utf-8"))
            return None
    else:
        print(f"Skipped configuring {build_dir}. Already exists.")
        return build_dir


def compile_app(mode: str, flags=None, enclave_size=None, debug=False, cpms=None, dynamic_enclave=False) -> Optional[
    Tuple[str, str]]:
    if mode == "sgx":
        target = "teebench"
    elif mode == "native":
        target = "native"
    elif mode == "tpch":
        target = "tpch"
    elif mode == "tpch-native":
        target = "tpch-native"
    else:
        raise ValueError("Unknown mode")

    if not cpms:
        cpms = 2900

    build_dir = configure(flags=flags, enclave_size=enclave_size, debug=debug, cpms=cpms,
                          dynamic_enclave=dynamic_enclave)
    if not build_dir:
        return None

    return compile_configured_app(target, build_dir)


def compile_configured_app(target: str, build_dir: str) -> Optional[Tuple[str, str]]:
    try:
        print(f"Building {build_dir}/{target}")
        subprocess.check_output(["cmake", "--build", build_dir, "--target", target],
                                cwd="../../", stderr=subprocess.STDOUT)
        return build_dir, target
    except subprocess.CalledProcessError as e:
        print(f"Failed with exit code {e.returncode}! Stdout:")
        print(e.stdout.decode("utf-8"))
        return None


def clean_all_builds():
    print("Cleaning all builds")
    build_directories = list(Path("../../").glob("cmake-build-exp*"))

    for build_dir in build_directories:
        try:
            subprocess.check_output(["cmake", "--build", build_dir, "--target", "clean"],
                                    cwd="../../", stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"Failed to clean build directory {build_dir} with error code {e.returncode}! Stdout:")
            print(e.stdout.decode("utf-8"))


def delete_all_configurations():
    print("Deleting all configurations")
    build_directories = list(Path("../../").glob("cmake-build-exp*"))
    for build_dir in build_directories:
        try:
            rmtree(build_dir)
        except:
            print("Deleting build directories failed!")


def remove_file(fname):
    try:
        os.remove(fname)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise


def init_file(fname, first_line):
    f = open(fname, "x")
    print(first_line, end="")
    f.write(first_line)
    f.close()


def find_fitting_enclave(sizes: (int, int), large_size_factor: bool, materialize: bool) -> str:
    """Find the smallest enclave that fits the data.

    :param sizes: number of tuples in (R, S)
    :return: Enclave size string
    """
    TUPLE_SIZE = 8
    SECURITY_FACTOR = 5 if large_size_factor else 3  # MWAY needs more than data_size * 4. *5 for enough space
    if materialize:
        SECURITY_FACTOR *= 2
    ENCLAVE_SIZES = [(2 ** 27, "128MB"),
                     (3 * 2 ** 26, "192MB"),
                     (2 ** 30, "1GB"),
                     (2 ** 31, "2GB"),
                     (2 ** 32, "4GB"),
                     (2 ** 32 + 2 ** 31, "6GB"),
                     (2 ** 33, "8GB"),
                     (2 ** 33 + 2 ** 31, "10GB"),
                     (2 ** 33 + 3 * 2 ** 30, "11GB"),
                     # (3 * 2 ** 33, "12GB"),
                     (2 ** 34, "16GB"),
                     (2 ** 35, "32GB")]

    num_tuples = sum(sizes)
    data_size = num_tuples * TUPLE_SIZE
    min_enclave_size = data_size * SECURITY_FACTOR

    for size, name in ENCLAVE_SIZES:
        if size >= min_enclave_size:
            return name

    raise ValueError("No enclave large enough!")


##############
# colors:
#
# 30s
# mint: bcd9be
# yellow brock road: e5cf3c
# poppy field: b33122
# powder blue: 9fb0c7
# egyptian blue: 455d95
# jade: 51725b
#
# 20s
# tuscan sun: e3b22b
# antique gold: a39244
# champagne: decd9d
# cadmium red: ad2726
# ultramarine: 393c99
# deco silver: a5a99f
#
# 80s
# acid wash: 3b6bc6
# purple rain: c77dd4
# miami: fd69a5
# pacman: f9e840
# tron turqouise: 28e7e1
# powersuit: fd2455
#
# 2010s
# millennial pink: eeb5be
# polished copper: c66d51
# quartz: dcdcdc
# one million likes: fc3e70
# succulent: 39d36e
# social bubble: 0084f9
#############


def color_alg(alg):
    # colors = {"CHT":"#e068a4", "PHT":"#829e58", "RHO":"#5ba0d0"}
    colors = {"CHT": "#b44b20",  # burnt sienna
              "PHT": "#7b8b3d",  # avocado
              "PSM": "#c7b186",  # natural
              "RHT": "#885a20",  # teak
              "RHO": "#e6ab48",  # harvest gold
              "RSM": "#4c748a",  # blue mustang
              "OJ": "#fd2455",
              "OPAQ": "#c77dd4",
              "OBLI": "#3b6bc6",
              "INL": '#7620b4',
              'NL': '#20b438',
              'MWAY': '#fd2455',
              'GHJ': 'black',
              'RHO_atomic': 'deeppink',
              'CRKJ': 'dodgerblue',
              'CRKJ_CHT': 'deeppink',
              'oldCRKJ': "#7b8b3d",  # avocado
              'CRKJ_static': "#4c748a",  # blue mustang

              'RHO-sgx': '#e6ab48',
              'RHO-sgx-affinity': 'g',
              'RHO-lockless': 'deeppink',
              'RHO_atomic-sgx': 'deeppink'}
    # colors = {"CHT":"g", "PHT":"deeppink", "RHO":"dodgerblue"}
    return colors[alg]


def marker_alg(alg):
    markers = {
        "CHT": 'o',
        "PHT": 'v',
        "PSM": 'P',
        "RHT": 's',
        "RHO": 'X',
        "RSM": 'D',
        "INL": '^',
        'MWAY': '*',
        'RHO_atomic': 'P',
        'CRKJ': '>',
        'CRKJ_CHT': '^',
        'oldCRKJ': 'X',
        'CRKJ_static': 'D',

        'RHO-sgx': 'X',
        'RHO-lockless': 'h',
        'RHO_atomic-sgx': 'h'
    }
    return markers[alg]


def savefig(filename, font_size=15, tight_layout=True):
    plt.rcParams.update({'font.size': font_size})
    if tight_layout:
        plt.tight_layout()
    plt.savefig(filename, transparent=False, bbox_inches='tight', pad_inches=0.1, dpi=200)
    print("Saved image file: " + filename)


def draw_vertical_lines(plt, x_values, linestyle='--', color='#209bb4', linewidth=1):
    for x in x_values:
        plt.axvline(x=x, linestyle=linestyle, color=color, linewidth=linewidth)


T = TypeVar("T")


def powerset(l: list[T]) -> list[list[T]]:
    return list(chain.from_iterable([combinations(l, r) for r in range(len(l) + 1)]))
