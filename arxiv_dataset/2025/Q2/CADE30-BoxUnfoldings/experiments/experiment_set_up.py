##Sets up different folders corresponding to different pairs of representatives
import os
import subprocess
import shutil

##ASSUMES this is run in the same directory as the file: run_all_experiments.py


def run_command(command, input=""):
    """Runs a command with stdin input and returns stdout"""
    try:
        logs = subprocess.run(command, input=input, shell=True, check=True, stdout=subprocess.PIPE, text=True).stdout
    except subprocess.CalledProcessError as e:
        logs = e.stdout

    return logs

def dim_2_str(dims):
    return f"{dims[0]}x{dims[1]}x{dims[2]}"

def pair_2_str(pair):
    return "_".join(str(x) for x in pair)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Sets up experiments 2-unfoldings')

    # Add arguments

    ##Required dimensions
    parser.add_argument('-d', '--dimensions', nargs=6, type=int,
                        help='Dimensions (x, y, z, a, b, c) corresponding to the boxes (x, y, z), (a, b, c)', required=True)
    parser.add_argument('-enc', '--encoder', type=str, help='Path to encoder', required=True)
    parser.add_argument('-o', '--output', type=str, help='Path for the creation of the experiments folder', required=True)
    parser.add_argument('-pair', '--pair_generator', type=str, help='Path to generator of pairs of representatives', required=True)
    parser.add_argument('--orient2', type=int, help='Orientation of starting cell on second box. Using 5 generates all 4 orientations', required=True)

    EXPERIMENT_RUNNER_PATH = "run_all_experiments.py"

    args = parser.parse_args()
    all_dims = args.dimensions
    encoder_path = args.encoder.replace(" ", r"\ ")
    output_path = args.output.replace(" ", r"\ ")
    pair_generator_path = args.pair_generator.replace(" ", r"\ ")
    orient2 = args.orient2

    assert orient2 in [0, 1, 2, 3, 5], "orient2 invalid, must be in [0, 1, 2, 3, 5]"

    dimension_list = [[all_dims[0], all_dims[1], all_dims[2]], all_dims[3:]]

    orients = [0, 1, 2, 3] if orient2 == 5 else [orient2]
    gen_pairs_cmd = f"python3 {pair_generator_path} -f 0 -d {dimension_list[1][0]} {dimension_list[1][1]} {dimension_list[1][2]}"
    base_path = output_path + "/" + dim_2_str(dimension_list[0]) + "_" + dim_2_str(dimension_list[1]) + f"_orient2_{orient2}"

    for ori in orients:
        encoder_cmd = f"python3 {encoder_path} -n 2 -f -1 -m 1 --orient2={ori} -d {dimension_list[0][0]} {dimension_list[0][1]} {dimension_list[0][2]} {dimension_list[1][0]} {dimension_list[1][1]} {dimension_list[1][2]} -s1 0 0 0 -t1 1 0 0"

        out = run_command(gen_pairs_cmd).split("\n")[:-1]
        cnt = out[0]
        pairs = [[int(x) for x in s.split()] for s in out[1:]]

        print(f"Total pairs: {cnt}")

        os.makedirs(base_path, exist_ok=True)
        for pair in pairs:
            cur_path = base_path + "/" + pair_2_str(pair) + f"_orient2_{ori}"
            os.makedirs(cur_path, exist_ok=True)
            ##Create encoding
            run_command(encoder_cmd + f" -s2 {pair[0]} {pair[1]} {pair[2]} -t2 {pair[3]} {pair[4]} {pair[5]} -o {cur_path}/encoding.cnf")
