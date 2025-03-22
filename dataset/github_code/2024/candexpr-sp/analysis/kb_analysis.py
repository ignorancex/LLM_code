
import re
from collections import defaultdict
from transformers import BartTokenizer
from tqdm import tqdm

from dhnamlib.pylib.decoration import fcache

numeric_types = ('quantity', 'year', 'date')

@fcache
def analyze_kb(kb):
    assert set(kb) == {'concepts', 'entities'}

    # string_attribute_keys = set()
    # number_attribute_keys = set()
    # time_attribute_keys = set()

    attribute_key_to_types = defaultdict(set)
    type_to_tokens = defaultdict(set)
    qualifier_key_to_types = defaultdict(set)
    relations = set()

    tokenizer = BartTokenizer.from_pretrained('./pretrained/bart-base')

    def get_tokens(text):
        return tokenizer.convert_ids_to_tokens(tokenizer(text)['input_ids'])[1:-1]

    def update_type_to_tokens(value_info):
        if value_info['type'] in numeric_types:
            type_to_tokens[value_info['type']].update(get_tokens(str(value_info['value'])))
            if value_info['type'] == 'quantity' and value_info['unit'] != '1':
                type_to_tokens['unit'].update(get_tokens(value_info['unit']))

    def update_type_to_tokens_from_qualifiers(qualifiers):
        for qk, qv in qualifiers.items():
            for elem in qv:
                qualifier_key_to_types[qk].add(elem['type'])
                update_type_to_tokens(elem)

    for entity in tqdm(kb['entities'].values()):
        for attribute in entity['attributes']:
            ak = attribute['key']
            av = attribute['value']

            attribute_key_to_types[ak].add(av['type'])
            update_type_to_tokens(av)

            update_type_to_tokens_from_qualifiers(attribute['qualifiers'])

        for relation in entity['relations']:
            relations.add(relation['relation'])
            update_type_to_tokens_from_qualifiers(relation['qualifiers'])

    return dict(attribute_key_to_types=attribute_key_to_types,
                type_to_tokens=type_to_tokens,
                qualifier_key_to_types=qualifier_key_to_types,
                relations=relations)


def iter_values(kb, valid=lambda value: True):
    for entity in tqdm(kb['entities'].values()):
        for attribute in entity['attributes']:
            ak = attribute['key']
            av = attribute['value']

            if valid(av['type']):
                yield av['value']

            for qk, qv in attribute['qualifiers'].items():
                for elem in qv:
                    if valid(elem['type']):
                        yield elem['value']

        for relation in entity['relations']:
            for qk, qv in relation['qualifiers'].items():
                for elem in qv:
                    if valid(elem['type']):
                        yield elem['value']


def is_digit(s):
    return re.match(r'^Ġ?[0-9]+$', s)


if __name__ == '__main__':
    from dhnamlib.pylib.filesys import json_load
    from itertools import chain
    # import re

    # Test 1
    kb = json_load('./_tmp_data-indented/indented-kb.json')
    kb_analysis = analyze_kb(kb, file_cache_path='_tmp_cache/kb-analysis.json')

    type_to_special_tokens = defaultdict(set)
    for typ in ('quantity', 'year', 'date'):
        for token in kb_analysis['type_to_tokens'][typ]:
            if not is_digit(token):
                type_to_special_tokens[typ].add(token)
    # type_to_special_tokens == {'quantity': {'.', '-', 'e'}, 'year': {'-'}, 'date': {'/'}}

    for qualifier, types in chain(kb_analysis['attribute_key_to_types'].items(),
                                  kb_analysis['qualifier_key_to_types'].items()):
        if len(types) >= 2:
            assert types == {'year', 'date'}

    # Test 2
    large_or_small_values = []  # [1.2e-06, 9e-06, 4.3e-06]
    negative_values = []
    float_values = []
    for v in set(iter_values(kb, valid=lambda v: v == 'quantity')):
        if 'e' in str(v):
            large_or_small_values.append(v)
        elif '-' in str(v):
            negative_values.append(v)
        elif '.' in str(v):
            float_values.append(v)

    special_years = []
    for v in set(iter_values(kb, valid=lambda v: v == 'year')):
        if len(str(v)) != 4:
            special_years.append(v)
    # special_years == [-84000000000, 1, 2, 4, 5, 7, 27, 30, 33, 36, 43, -2000, 50, 67, -4000, 100, 106, -76, 140, -8000, 200, -1812, -72000, 250, -4540000000, 300, 301, -69, 330, 332, -57000, 350, 354, 374, 395, 400, 401, -10500, 461, -63, 479, 482, 500, 501, -1534, -1500, 553, 570, 580, 600, 601, 610, 632, 634, 635, 636, 681, -54, 697, 700, 701, 729, 742, 751, 752, 762, 766, 780, 794, 800, 810, 841, 843, -1200, 850, 854, 861, 862, 870, 871, 880, 882, 890, -66000000, 30000000, 900, 901, 907, 927, 929, -7250, -44, 944, 950, 960, 962, 966, 969, 971, 985, 988, -42000, -1000, -3000, -5000, -7000, -9000, -4900, -13000, -15000, -4570000000, -4500, -2500, -4527000000, -4000000, -45, 802701, -27, -30, -16, -15, -170000, -12, -7, -2]

    for v in set(iter_values(kb, valid=lambda v: v == 'date')):
        if not re.match(r'^[0-9]?[0-9]?[0-9]?[0-9]/[0-9]?[0-9]/[0-9]?[0-9]$', v):
            breakpoint()
            raise Exception("Date type mismatch")

    breakpoint()
    # a = 100
