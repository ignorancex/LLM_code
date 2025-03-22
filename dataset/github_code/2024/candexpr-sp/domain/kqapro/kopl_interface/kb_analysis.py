
import re
from datetime import date
from collections import defaultdict
from splogic.utility.trie import SequenceTrie
# from itertools import chain
# from tqdm import tqdm

# from dhnamlib.pylib.decoration import fcache
from dhnamlib.pylib.typeutil import creatable
from dhnamlib.pylib.iteration import unique, distinct_pairs
from dhnamlib.pylib.decoration import construct, id_cache
from dhnamlib.pylib.function import compose


numeric_types = ('quantity', 'year', 'date')


# @fcache
@id_cache
def extract_kb_info(kb):
    assert set(kb) == {'concepts', 'entities'}

    concepts = set()
    entities = set()
    relations = set()
    attribute_to_types = defaultdict(set)
    qualifier_to_types = defaultdict(set)
    units = set()

    def add_type(type_set, value):
        type_set.add(value['type'])
        if value['type'] == 'quantity' and value['unit'] != '1':
            units.add(value['unit'])

    def update_from_qualifiers(qualifiers):
        for qk, qv in qualifiers.items():
            for elem in qv:
                add_type(qualifier_to_types[qk], elem)

    # for concept in tqdm(kb['concepts'].values()):
    for concept in kb['concepts'].values():
        concepts.add(concept['name'])

    # for entity in tqdm(kb['entities'].values()):
    for entity in kb['entities'].values():
        entities.add(entity['name'])

        for attribute in entity['attributes']:
            add_type(attribute_to_types[attribute['key']], attribute['value'])
            update_from_qualifiers(attribute['qualifiers'])

        for relation in entity['relations']:
            relations.add(relation['relation'])
            update_from_qualifiers(relation['qualifiers'])

    def normalize_type(types):
        if len(types) >= 2:
            assert types == {'year', 'date'}
            return 'time'
        else:
            return unique(types)

    def make_type_to_keys(key_to_types):
        type_to_keys = dict()
        for attr, types in key_to_types.items():
            type_to_keys.setdefault(normalize_type(types), set()).add(attr)
        return type_to_keys

    return dict(concepts=concepts,
                entities=entities,
                relations=relations,
                type_to_attributes=make_type_to_keys(attribute_to_types),
                type_to_qualifiers=make_type_to_keys(qualifier_to_types),
                units=units)


def make_trie(kw_set, lf_tokenizer, end_of_seq):
    trie = SequenceTrie(tokenizer=lf_tokenizer, end_of_seq=end_of_seq)
    for kw in kw_set:
        trie.add_text(kw)
    return trie


@construct(compose(dict, distinct_pairs))
def make_trie_dict(kb_info, lf_tokenizer, end_of_seq):
    def _make_trie(kw_set):
        return make_trie(kw_set, lf_tokenizer, end_of_seq)

    for key, coll in kb_info.items():
        if isinstance(coll, dict):
            new_coll = dict([typ, _make_trie(typed_coll)] for typ, typed_coll in coll.items())
        else:
            new_coll = _make_trie(coll)
        yield key, new_coll


def make_kw_to_type_dict(kb_info):
    @construct(dict)
    def _make_kw_to_type_dict(type_to_kw):
        for typ, kws in kb_info[type_to_kw].items():
            for kw in kws:
                yield kw, typ

    return dict(
        attribute=_make_kw_to_type_dict('type_to_attributes'),
        qualifier=_make_kw_to_type_dict('type_to_qualifiers'))


quantity_prefix_regex = re.compile(r'^[+-]?[0-9]*(\.[0-9]*)?(e([+-][0-9]*)?)?$')
year_prefix_regex = re.compile(r'^[+-]?[0-9]*$')
date_prefix_regex = re.compile(r'^[0-9]{1,4}(-([0-9]{1,2}(-([0-9]{1,2})?)?)?)?$')


def is_quantity_prefix(text):
    return quantity_prefix_regex.match(text)


def is_date_prefix(text):
    return date_prefix_regex.match(text)


def is_year_prefix(text):
    return year_prefix_regex.match(text)


def is_value_type(value, typ):
    def parse_date(value):
        ymd_tuple = tuple(map(int, value.split('-')))
        if len(ymd_tuple) == 3:
            return date(*ymd_tuple)
        else:
            raise ValueError('datetime.date requires 3 int type arguments')

    cls_dict = dict([['string', lambda *args: None],
                     ['quantity', float],
                     ['date', parse_date],
                     ['year', int]])

    return creatable(cls_dict[typ], value, exception_cls=ValueError)


if __name__ == '__main__':
    from transformers import BartTokenizer
    from dhnamlib.pylib.filesys import json_load
    # import re

    # Test 1
    kb = json_load('./_tmp_data-indented/indented-kb.json')
    # keyword_info = extract_kb_info(kb, file_cache_path='_tmp_cache/kb-analysis.json')
    kb_info = extract_kb_info(kb)
    
    tokenizer = BartTokenizer.from_pretrained(
        './pretrained/bart-base',
        add_prefix_space=True)
    reduce_token = '<reduce>'
    tokenizer.add_tokens([reduce_token], special_tokens=True)
    trie_dict = make_trie_dict(kb_info, tokenizer, reduce_token)
