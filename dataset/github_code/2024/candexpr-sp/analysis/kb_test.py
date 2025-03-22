
from collections import defaultdict
import argparse
from itertools import chain

from dhnamlib.pylib.filesys import json_load, json_save_pretty


def analyze_kb(input_path, output_path):
    kb = json_load(input_path)
    assert set(kb) == {'concepts', 'entities'}

    concepts = []

    for concept in kb['concepts'].values():
        concepts.append(concept['name'])

    entities = []
    attributes = []
    attribute_qualifiers = defaultdict(lambda: defaultdict(int))
    relations = []
    relation_qualifiers = defaultdict(lambda: defaultdict(int))

    for entity in kb['entities'].values():
        entities.append(entity['name'])
        for attribute in entity['attributes']:
            attributes.append(attribute['key'])
            for k, v in attribute['qualifiers'].items():
                for elem in v:
                    attribute_qualifiers[k][elem['type']] += 1
        for relation in entity['relations']:
            relations.append(relation['relation'])
            for k, v in relation['qualifiers'].items():
                for elem in v:
                    relation_qualifiers[k][elem['type']] += 1

    info = dict(
        entities=entities,
        attributes=attributes,
        attribute_qualifiers=attribute_qualifiers,
        relations=relations,
        relation_qualifiers=relation_qualifiers)

    # qualifier_types = ['string', 'quantity', 'year', 'date']

    breakpoint()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    # parser.add_argument('output')

    args = parser.parse_args()

    # analyze_kb(args.input, args.output)
    analyze_kb(args.input, None)

    pass


# concept, entity, attribute, relation, qualifier
# concept names
# entity names
# attribute key
# relation

# pygtrie.StringTrie._root.children.iteritems()

# pygtrie._get_node


import pygtrie
from transformers import BartTokenizer


class TokenTrie:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.trie = pygtrie.Trie()

    def _add_token_ids(self, ids):
        self.trie[ids] = True

    add = _add_token_ids

    def add_text(self, text):
        assert self.tokenizer is not None
        token_ids, attention_mask = self.tokenizer(text, add_special_tokens=False)
        self._add_token_ids(token_ids)

    def __contains__(self, key):
        # "key" is a sequence of token ids
        # return self.trie.get(key, False)
        return key in self.trie

    def candidate_token_ids(self, prefix):
        # "prefix" is a sequence of token ids of a prefix part of an entire sequence of a key
        prefix_node, path = self.trie._get_node(prefix)
        return (token_id for token_id, node in prefix_node.children.iteritems())

    def __iter__(self):
        return iter(self.trie)
