import os
import re
import pandas as pd


def main():
    """Main function to process experiment logs and save the processed data."""
    EXP = int(input('Enter the experiment number (1, 2, 3, 4, or 5): '))
    # EXP = 3
    # Get experiment settings
    targets, plot_directory, experiments, ks, model_types, bits_list_per_target = get_experiment_settings(EXP)

    # Base directory for the log files
    base_dir = '/scratch/qilong3/transferattack/results'

    # Read and parse logs
    df = read_and_parse_logs(base_dir, targets, experiments, bits_list_per_target, model_types, ks, EXP)

    # Process dataframe
    df = process_dataframe(df)

    # Save the DataFrame to a CSV file for later use
    df.to_csv(f'processed_data_exp{EXP}.csv', index=False)
    print(f"Processed data has been saved to 'processed_data_exp{EXP}.csv'.")


def get_experiment_settings(EXP):
    """
    Returns the experiment settings based on the experiment number.

    Parameters:
    EXP (int): Experiment number (1, 2, 3, or 4)

    Returns:
    tuple: (targets, plot_directory, experiments, ks, model_types, bits_list_per_target)
    """
    # Define the targets, ks, and experiments based on EXP
    if EXP == 1 or EXP == 5:
        targets = ['hidden', 'mbrs', 'stega', 'RivaGAN']
        plot_directory = f'plots_{EXP}'
        experiments = ['optimization']
        ks = [1, 2, 5, 10, 20, 30, 40, 50]
        model_types = ['cnn']
    elif EXP == 2:
        targets = ['hidden']
        plot_directory = 'plots_2'
        experiments = [
            'mean',
            'mean_budget_0.25_clamp',
            'optimization'
        ]
        ks = [1, 2, 5, 10, 20, 30, 40, 50]
        model_types = ['cnn', 'resnet']
    elif EXP == 3:
        targets = ['hidden']
        plot_directory = 'plots_3'
        experiments = ['optimization'] + [f'mean_model_{i}' for i in range(2, 12)] + \
                      [f'mean_budget_0.25_clamp_model_{i}' for i in range(2, 12)]
        ks = [1, 2, 5, 10, 20, 30, 40, 50]
        model_types = ['cnn', 'resnet']
    elif EXP == 4:
        targets = ['hidden', 'mbrs', 'stega', 'RivaGAN']
        plot_directory = 'plots_4'
        experiments = [
            'mean',
            'mean_budget_0.25_clamp',
        ]
        ks = [1]
        model_types = ['cnn']
    else:
        raise ValueError('EXP must be 1, 2, 3, 4 or 5.')

    # Initialize bits_list_per_target
    bits_list_per_target = {}
    for target in targets:
        # Define bits_list for each target
        if target == 'hidden':
            bits_list = [30] if EXP in [1, 4, 5] else [20, 30, 64]
        elif target == 'mbrs':
            bits_list = [64, 256]
        elif target == 'stega':
            bits_list = [100]
        elif target == 'RivaGAN':
            bits_list = [32]
        else:
            raise ValueError(f'Unknown target: {target}')
        bits_list_per_target[target] = bits_list

    return targets, plot_directory, experiments, ks, model_types, bits_list_per_target


def read_and_parse_logs(base_dir, targets, experiments, bits_list_per_target, model_types, ks, EXP):
    """
    Reads and parses log files based on the given parameters.

    Parameters:
    base_dir (str): Base directory where the log files are stored.
    targets (list): List of target models.
    experiments (list): List of experiments.
    bits_list_per_target (dict): Dictionary mapping targets to their bits list.
    model_types (list): List of model types.
    ks (list): List of k values.
    EXP (int): Experiment number.

    Returns:
    pd.DataFrame: DataFrame containing the parsed data.
    """
    # Initialize an empty list to collect the data
    data = []
    source_bits = 30  # Assuming source bits is 30

    # Get the list of sources based on EXP
    sources_list = get_sources_list(EXP)

    # Loop over parameters to construct file paths and parse logs
    for target in targets:
        bits_list = bits_list_per_target[target]
        for bits in bits_list:
            for model_type in model_types:
                for experiment in experiments:
                    ks_to_process = get_ks_to_process(EXP, experiment, ks)
                    for k in ks_to_process:
                        for source in sources_list:
                            # Adjust base directory based on EXP and experiment
                            base_dir_for_experiment = get_base_dir_for_experiment(base_dir, EXP, experiment)
                            # Construct file path
                            filepath = construct_file_path(base_dir_for_experiment, EXP, experiment, target,
                                                           model_type, source_bits, bits, k, source)
                            # Read and parse log file if it exists
                            if os.path.exists(filepath):
                                metrics = parse_log_file(filepath, target, k, experiment, bits, model_type, source)
                                data.append(metrics)
                            else:
                                print(f'File {filepath} does not exist.')

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Ensure that the 'k' column is numeric and sort the DataFrame
    df['k'] = pd.to_numeric(df['k'])
    df = df.sort_values(by=['target', 'experiment', 'bits', 'model_type', 'k', 'source'])

    return df


def get_sources_list(EXP):
    """
    Returns the list of sources based on the experiment number.

    Parameters:
    EXP (int): Experiment number.

    Returns:
    list: List of sources.
    """
    return ['stable_diffusion'] if EXP in [1, 4, 5] else ['stable_diffusion', 'midjourney']


def get_ks_to_process(EXP, experiment, ks):
    """
    Returns the list of k values to process based on EXP and experiment.

    Parameters:
    EXP (int): Experiment number.
    experiment (str): Experiment name.
    ks (list): Original list of k values.

    Returns:
    list: List of k values to process.
    """
    return [1] if EXP == 3 and experiment != 'optimization' else ks


def get_base_dir_for_experiment(base_dir, EXP, experiment):
    """
    Determines the base directory for a given experiment.

    Parameters:
    base_dir (str): The base directory path.
    EXP (int): The experiment identifier.
    experiment (str): The name of the experiment.

    Returns:
    str: The resolved base directory for the experiment.

    Raises:
    ValueError: If an invalid EXP value is provided.
    """
    if EXP in [1, 5]:
        # For experiments 1 and 5, return the base directory as-is.
        return base_dir
    elif EXP == 2:
        # For experiment 2, append '0.25' only if the experiment is 'optimization'.
        if experiment == 'optimization':
            return os.path.join(base_dir, '0.25')
        return base_dir
    elif EXP == 3:
        # For experiment 3, always return the base directory with '0.25'.
        return os.path.join(base_dir, '0.25')
    elif EXP == 4:
        # For experiment 4, always append '0.25'.
        return os.path.join(base_dir, '0.25')
    else:
        # Raise an error for invalid EXP values.
        raise ValueError(f"Invalid EXP value: {EXP}. Expected values are 1, 2, 3, 4, or 5.")



def construct_file_path(base_dir_for_experiment, EXP, experiment, target, model_type, source_bits, bits, k, source):
    """
    Constructs the file path for the log file based on parameters.

    Parameters:
    base_dir_for_experiment (str): Base directory for the experiment.
    EXP (int): Experiment number.
    experiment (str): Experiment name.
    target (str): Target model.
    model_type (str): Model type.
    source_bits (int): Number of bits in source.
    bits (int): Number of bits in target.
    k (int): Number of models.
    source (str): Source of the images.

    Returns:
    str: Full file path to the log file.
    """
    # Construct base filename
    base_filename = f'hidden_to_{target}_{model_type}_{source_bits}_to_{bits}bitsAT_{k}models'
    if source == 'midjourney':
        base_filename += '_midjourney'

    if EXP == 4 or EXP == 3:
        # For EXP==4, logs are at base_filename/results.log
        if experiment == 'optimization':
            pass
            # base_filename += '.log'
        else:
            base_filename += f'_no_optimization_{experiment}'
        filename = 'results.log'
    else:
        if experiment == 'optimization':
            filename = base_filename + '.log'
        else:
            filename = base_filename + f'_no_optimization_{experiment}.log'
        base_filename = ''

    filepath = os.path.join(base_dir_for_experiment, base_filename, filename)
    return filepath


def parse_log_file(filepath, target, k, experiment, bits, model_type, source):
    """
    Parses the log file and extracts metrics.

    Parameters:
    filepath (str): Path to the log file.
    target (str): Target model.
    k (int): Number of models.
    experiment (str): Experiment name.
    bits (int): Number of bits.
    model_type (str): Model type.
    source (str): Source of images.

    Returns:
    dict: Dictionary of extracted metrics.
    """
    with open(filepath, 'r') as f:
        content = f.readlines()
    metrics = {
        'target': target,
        'k': k,
        'experiment': experiment,
        'bits': bits,
        'model_type': model_type,
        'source': 'Midjourney' if source == 'midjourney' else 'Stable Diffusion'
    }
    for line in content:
        # Use regex to extract metric name and value
        match = re.match(r'INFO:root:(.+?)\s+([0-9\.]+)', line)
        if match:
            metric_name = match.group(1).strip()
            value = float(match.group(2))
            # Sanitize the metric name to be used as a DataFrame column
            metric_name = sanitize_metric_name(metric_name)
            metrics[metric_name] = value
    return metrics


def sanitize_metric_name(metric_name):
    """
    Sanitizes the metric name to be used as a DataFrame column.

    Parameters:
    metric_name (str): Original metric name.

    Returns:
    str: Sanitized metric name.
    """
    return re.sub(r'[\s\-\(\)]', '_', metric_name)


def process_dataframe(df):
    """
    Processes the DataFrame by computing additional metrics.

    Parameters:
    df (pd.DataFrame): Original DataFrame.

    Returns:
    pd.DataFrame: Processed DataFrame.
    """
    # Compute Evasion Rate and Evasion Rate (Unattacked)
    df['Evasion_Rate'] = 1 - df['tdr_attk']
    df['Evasion_Rate_Unattacked'] = 1 - df['tdr']

    return df


if __name__ == '__main__':
    main()
