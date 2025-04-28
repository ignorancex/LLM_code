import pathlib
import sys

from gravann import validator_runner

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise IOError("No input provided!")
    name = sys.argv[1]
    cuda_devices = sys.argv[2]
    input_directory = pathlib.Path(f"./results/{name}")
    output_directory = pathlib.Path(f"./results/re-validation-noise/{name}")
    output_directory.mkdir(parents=True, exist_ok=True)

    validator_runner.run(
        input_directory,
        output_directory,
        N=10000,
        sampling_altitudes=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
        integration_points=300000,
        cuda_devices=cuda_devices,
        with_constant_noise=True
    )
