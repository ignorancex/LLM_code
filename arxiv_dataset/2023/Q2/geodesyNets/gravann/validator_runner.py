import os

import pandas as pd

from gravann.input import model_reader
from gravann.network import encodings
from gravann.training import validator
from gravann.util import deep_get, enableCUDA


def run(
        input_directory: os.PathLike,
        output_directory: os.PathLike,
        ground_truth: str = 'polyhedral',
        **kwargs
) -> pd.DataFrame:
    """Reads all configuration files and models root_directory in the root_directory. Populates the models and re-runs
    the validation.

    Args:
        input_directory: the directory to recursively search for models and their configuration files
        output_directory: the directory where to store the validation results as csv file
        ground_truth: the ground truth for the validation (either 'polyhedral' or 'mascon')
        **kwargs: passed to validator.validation(..)

    Keyword Args:
        integration_points (int, optional): Number of integrations points to use. Defaults to 500000.
        N (int, optional): Number of evaluations per altitude. Defaults to 5000.
        sampling_altitudes (np.array, optional): Altitude to sample at for validation. Defaults to [0.05, 0.1, 0.25].
        with_constant_noise (bool): include noise in the ground truth for constant bias validation
        ...

    Returns:
        pandas DataFrame containing the validation results

    """
    # Cuda Init
    os.environ["CUDA_VISIBLE_DEVICES"] = kwargs.get("cuda_devices", "0")
    enableCUDA()

    # Start validation
    results_df = pd.DataFrame()
    print("##Loading models...")
    all_models = model_reader.read_models(input_directory)
    total_number = len(all_models)
    print(f"##Loaded {total_number} models")
    for it, (model, config) in enumerate(all_models):
        # Read
        print(f"#### Starting with re-validation {it + 1}/{total_number}")
        encoding = encodings.get_encoding(deep_get(config, ["Encoding", "encoding"]))
        sample = deep_get(config, ["Sample", "sample"])
        use_acc = deep_get(config, ["use_acc", "Type"], True)
        # The format changed between the different runs, string is legacy format, simple boolean is new format
        if isinstance(use_acc, str) and use_acc == "ACC":
            use_acc = True
        elif isinstance(use_acc, str) and use_acc == "U":
            use_acc = False

        # If with_noise is given and set to True + the noise includes a constant bias
        noise_config = None
        if kwargs.get("with_constant_noise", False):
            noise_config = config["noise"]

        # Validate
        validation_results = validator.validate(
            model, encoding, sample, ground_truth, use_acc,
            progressbar=False,
            noise=noise_config,
            **kwargs
        )

        # Output
        val_res = validator.validation_results_unpack_df(validation_results)
        config_val_df = pd.concat([pd.DataFrame([config]), val_res], axis=1)
        results_df = results_df.append(config_val_df, ignore_index=True)
        results_df.to_csv(os.path.join(output_directory, f"it{it:04d}_val_checkpoint.csv"), index=False)

    results_df.to_csv(os.path.join(output_directory, f"validation_final_result.csv"), index=False)
    return results_df
