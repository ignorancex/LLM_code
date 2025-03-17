
from fractions import Fraction
from argparse import ArgumentParser
from dhnamlib.pylib.filesys import extended_json_load  # , pandas_tsv_load

from meta_configuration import set_default_domain_name
set_default_domain_name('overnight')  # Call `set_default_domain_name` before the `configuration` module is loaded.
# from configuration import config


# def get_domain_size_dict(split):
#     if split == 'val':
#         dataset = config.encoded_val_set
#     else:
#         assert split == 'test'
#         dataset = config.encoded_test_set

#     domain_size_dict = {}
#     for example in dataset:
#         domain_size_dict[example['domain']] = domain_size_dict.get(example['domain'], 0) + 1
#     return domain_size_dict

def main():
    parser = ArgumentParser(description='Merging test results',)
    # parser.add_argument('split', choices=['val', 'test'])
    parser.add_argument('extra_performance_paths', nargs='+')

    args = parser.parse_args()

    # domain_size_dict = get_domain_size_dict(args.split)
    # total_size = sum(domain_size_dict.values())

    accumulated = Fraction()
    domains = set()
    total_num_correct = 0
    total_num_examples = 0

    for extra_performance_file_path in args.extra_performance_paths:
        extra_performance = extended_json_load(extra_performance_file_path)
        assert len(extra_performance) == 1
        [[domain, performance]] = extra_performance.items()
        assert domain not in domains
        domains.add(domain)
        total_num_correct += performance['num_correct']
        total_num_examples += performance['num_examples']

    assert len(domains) == 8

    print('Merged accuracy: {}'.format(total_num_correct / total_num_examples))
    print('Merged accuracy fraction: {} / {}'.format(total_num_correct, total_num_examples))

"""
Example
  BASE_DIR_PATH=model-test/overnight/some-result
  DECODING_TYPE=full-constraints:best  # full-constraints:best, no-arg-candidate:best, no-ac-no-dut:best, no-constrained-decoding:best
  python -m domain.overnight.merge_result $BASE_DIR_PATH/*/$DECODING_TYPE/extra_performance.json
"""


if __name__ == '__main__':
    main()
