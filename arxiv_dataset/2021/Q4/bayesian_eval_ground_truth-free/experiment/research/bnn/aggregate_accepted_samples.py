"""Script that aggregates the accepted BNN sampled weights from BNN MCMC into a
single json.
"""
import argparse
import logging

from experiment import io
from experiment.research.bnn import bnn_mcmc_fwd


parser = argparse.ArgumentParser(description=' '.join([
    'Aggregates the accepted BNN sampled weights from BNN MCMC into a single',
    'JSON.',
]))

parser.add_argument(
    'input_dir',
    help='The directory of BNN sampled weights to recursively crawl.',
)

parser.add_argument(
    'output_filename',
    default=None,
    help='The output file path of converted json.',
)

args = parser.parse_args()

weights_sets = bnn_mcmc_fwd.load_sample_weights(args.input_dir)

logging.info('Total accepted BNN MCMC samples = %d', weights_sets[0].shape[0])

io.save_json(args.output_filename, weights_sets)
