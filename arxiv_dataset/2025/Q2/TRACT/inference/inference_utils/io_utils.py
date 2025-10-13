import os
import pandas as pd
from typing import List, Dict, Tuple, Union
import logging
from colorama import Fore, Style

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def save_results(
    output_dir: str,
    all_correlations: Union[List, Tuple],
    args: Dict,
):
    """Save results to a CSV file.
    If the output file already exists, will directly load the file and append the new results.
    Args:
        output_dir (str): The path to the output directory.
        all_correlations (Union[List, Tuple]): A list or tuple of correlations.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "results.csv")

    # Load the csv as a DataFrame if it already exists
    if os.path.exists(output_file):
        results = pd.read_csv(output_file)
    else:
        results = pd.DataFrame()

    # Write the new results to the dataframe
    # Check 
    new_row_to_add = args | all_correlations

    if results.empty:
        results = pd.DataFrame([new_row_to_add])
    else:
        results = pd.concat([results, pd.DataFrame([new_row_to_add])], ignore_index = True)
    
    # Save the results to the output file
    results.to_csv(output_file, index = False)
    logger.info(f"Savd the results to {Fore.GREEN}{output_file}{Style.RESET_ALL}.")
