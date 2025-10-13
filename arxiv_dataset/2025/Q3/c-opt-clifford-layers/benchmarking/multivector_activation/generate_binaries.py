import os
import itertools
import subprocess

# Define the parameter spaces
K_vals = [4, 8]
modes = {
    0: 'Linear',
    1: 'Sum',
    2: 'Mean'
}
versions = {
    'baseline': 'multivector_act.c',
    'opt1': 'multivector_act_opt1.c',
    'opt2': 'multivector_act_opt2.c'
}

# Compiler settings
CC = "gcc"
CFLAGS = "-O3 -march=native -ffast-math -Wall"
SRC = "multivector_act_benchmark_x86_new.c"
COMMON_SRC = "../../clib/multivector_activation"
HEADER = "tsc_x86.h"
OUT_DIR = "build"

os.makedirs(OUT_DIR, exist_ok=True)

for K, mode, version in itertools.product(K_vals, modes.keys(), versions.keys()):
    target_name = f"multivector_act_benchmark_x86_K={K}_{version}_{modes[mode]}"
    output_path = os.path.join(OUT_DIR, target_name)

    compile_cmd = [
        CC,
        CFLAGS,
        f"-DMODE={mode}",
        f"-DK_VAL={K}",
        f"-DVERSION_{version.upper()}",  # e.g., -DVERSION_BASELINE
        "-o", output_path,
        SRC,
        os.path.join(COMMON_SRC, versions[version]),
        HEADER,
        "-lm"
    ]

    print(f"Compiling {target_name} ...")
    try:
        subprocess.run(" ".join(compile_cmd), check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to compile {target_name}: {e}")

print("All binaries built.")
