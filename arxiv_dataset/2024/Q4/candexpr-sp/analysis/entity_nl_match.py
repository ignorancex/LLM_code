
import argparse
from itertools import chain

from dhnamlib.pylib.filesys import json_load, json_save_pretty


def analyze_entities(input_path, output_path):
    raw_dataset = json_load(input_path)
    entity_to_info = {}
    for example in raw_dataset:
        question = example['question']
        for entity in chain(*(func_call['inputs'] for func_call in example['program'])):
            info = entity_to_info.setdefault(entity, {})
            info['freq'] = info.get('freq', 0) + 1
            info['matched'] = info.get('matched', 0) + int(entity in question)
            info['question'] = question
    info_list = sorted(entity_to_info.items(),
                       key=lambda x: x[1]['matched'] - x[1]['freq'])

    print('#entities: {}'.format(len(entity_to_info)))
    print('#unmatched: {}'.format(sum(1 for entity, info in entity_to_info.items()
                                      if info['matched'] != info['freq'])))
    print('#entitity count: {}'.format(sum(info['freq'] for info in entity_to_info.values())))
    print('#unmatched count: {}'.format(sum(info['freq'] - info['matched'] for entity, info in entity_to_info.items()
                                            if info['matched'] != info['freq'])))
    print('#fully unmatched count: {}'.format(sum(info['freq'] - info['matched'] for entity, info in entity_to_info.items()
                                                  if info['matched'] == 0)))
    json_save_pretty(info_list, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')

    args = parser.parse_args()

    analyze_entities(args.input, args.output)
