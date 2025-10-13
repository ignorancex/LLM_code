from blind_delta import calc_individual
from itertools import product
import json
import sys
import csv

DEFAULT_OUTPUT_FILENAME = 'blind_delta_output.csv'
DEFAULT_DEPTH = 2000
DEFAULT_P = 2
DEFAULT_PRECISION = 100_000
DEFAULT_NOT_CALCULATED_MARKER = -1010
DEFAULT_RATIONAL_MARKER = -2020
DEFAULT_LIMIT_CONSTANT = 1e10


def blind_delta_multi_pcf_wrapper(coefficents_ranges, coefficients_lengths, depth=DEFAULT_DEPTH, p=DEFAULT_P,
    precision=DEFAULT_PRECISION, not_calculated_marker=DEFAULT_NOT_CALCULATED_MARKER,
    rational_marker=DEFAULT_RATIONAL_MARKER, limit_constant=DEFAULT_LIMIT_CONSTANT):
    """
    A wrapper function from that executes blind_delta.calc_individual (which executes the blind delta algorithm on a 
    single pcf), over a set of continued fractions.
    
    Parameters:
    - coefficents_ranges (list): Coefficient ranges to scan. Every element should contain a tuple with the minimal and
      maximal value allowd for the coefficent. For example, [(1, 2), (3, 4)] entails that blind delta will be executed on
      the paramters (1,3), (1,4), (2,3), (2,4).
    The rest of the parameters are identical to those of blind_delta.calc_individual.
    - coefficients_lengths (list): Lengths (or the degree+1) of a_n and b_n polynomials.
    - depth (int): Calculation depth.
    - p (int): The relation between the calculation depth and the point where the blind delta is sampled.
    - precision (int): Precision for calculations.
    - not_calculated_marker: Marker for not calculated values.
    - rational_marker: Marker for rational values.
    - LIMIT_CONSTANT: A constant mimicing infinity.
    """
    # Expand coefficents_ranges to a list of coefficent combinations in the range.
    combinations = product(*[range(min_val, max_val+1) for min_val, max_val in coefficents_ranges])

    # Discard cases where one of the polynomials is strictly zero
    filtered_combinations = []
    for coefs in combinations:
        an_coefs = coefs[:coefficients_lengths[0]]
        bn_coefs = coefs[coefficients_lengths[0]:]

        if all(c == 0 for c in an_coefs) or all(c == 0 for c in bn_coefs):
            # Bad pcf 
            continue

        filtered_combinations.append(coefs)

    # Run blind delta on all PCFs in the filtered set.
    results = []
    for coefs in filtered_combinations:
        print(coefs)
        results.append(calc_individual(
            coefficients=coefs, 
            coefficients_lengths=coefficients_lengths, 
            depth=depth, 
            p=p, 
            precision=precision, 
            not_calculated_marker=not_calculated_marker, 
            rational_marker=rational_marker, 
            LIMIT_CONSTANT=limit_constant))

    return results


def main():
    if len(sys.argv) not in (2, 3) or sys.argv[1] in ('-h', '--help', '-?', '/?'):
        print('Usage:')
        print('execute_from_json.py delta2_job.json [output_filename.csv]')

        exit()

    job_config_filename = sys.argv[1]
    output_filename = sys.argv[2] if len(sys.argv) == 3 else DEFAULT_OUTPUT_FILENAME

    with open(job_config_filename, 'r') as f:
        job_data = json.load(f)
    results = blind_delta_multi_pcf_wrapper(**job_data)
    
    with open(output_filename, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        fields = results[0].keys()
        csvwriter.writerow(fields)

        for result in results:
            csvwriter.writerow(result.values())


if __name__ == '__main__':
    main()