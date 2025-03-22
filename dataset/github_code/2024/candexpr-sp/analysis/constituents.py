
from collections import defaultdict
from itertools import chain

from dhnamlib.pylib.structure import get_recursive_dict, AttrDict


qualifier_types = ['string', 'quantity', 'year', 'date']


def get_kb_constituents(kb):
    assert set(kb) == {'concepts', 'entities'}

    concepts = []

    for concept in kb['concepts'].values():
        concepts.append(concept['name'])

    entity_names = []
    attribute_keys = []
    attribute_qualifier_typed_values = defaultdict(lambda: defaultdict(set))
    relation_names = []
    relation_qualifier_typed_values = defaultdict(lambda: defaultdict(set))

    for entity in kb['entities'].values():
        entity_names.append(entity['name'])
        for attribute in entity['attributes']:
            attribute_keys.append(attribute['key'])
            for k, v in attribute['qualifiers'].items():
                for elem in v:
                    attribute_qualifier_typed_values[k][elem['type']].add(elem['value'])
        for relation in entity['relations']:
            relation_names.append(relation['relation'])
            for k, v in relation['qualifiers'].items():
                for elem in v:
                    relation_qualifier_typed_values[k][elem['type']].add(elem['value'])

    qualifier_typed_values = defaultdict(lambda: defaultdict(set))
    for qualifier, typed_values in chain(attribute_qualifier_typed_values.items(),
                                         relation_qualifier_typed_values.items()):
        for typ, values in typed_values.items():
            qualifier_typed_values[qualifier][typ].update(values)

    constituent_dict = dict(
        entity_names=entity_names,
        attribute_keys=attribute_keys,
        attribute_qualifier_typed_values=attribute_qualifier_typed_values,
        relation_names=relation_names,
        relation_qualifier_typed_values=relation_qualifier_typed_values,
        qualifier_typed_values=qualifier_typed_values)

    return get_recursive_dict(constituent_dict, AttrDict)


if __name__ == '__main__':
    from dhnamlib.pylib.filesys import json_load

    constituents = get_kb_constituents(json_load('./_tmp_data-indented/indented-kb.json'))
    breakpoint()
    a = 100
