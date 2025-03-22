from dhnamlib.pylib.filesys import json_load
from kopl.data import KB


def compare_recursively(kb_json_original, kb_json_modified):
    if type(kb_json_original) != type(kb_json_modified):
        breakpoint()
    assert type(kb_json_original) == type(kb_json_modified)
    if isinstance(kb_json_original, (tuple, list)):
        if len(kb_json_original) != len(kb_json_modified):
            breakpoint()
        assert len(kb_json_original) == len(kb_json_modified)
        for x, y in zip(kb_json_original, kb_json_modified):
            compare_recursively(x, y)
    elif isinstance(kb_json_original, dict):
        if len(kb_json_original) != len(kb_json_modified):
            breakpoint()
        assert len(kb_json_original) == len(kb_json_modified)
        for (xk, xv), (yk, yv) in zip(kb_json_original.items(), kb_json_modified.items()):
            compare_recursively(xk, yk)
            compare_recursively(xv, yv)
    else:
        assert kb_json_original == kb_json_modified, f'{kb_json_original} != {kb_json_modified}'


if __name__ == '__main__':
    '''
    Usage:
        $ python -m analysis.kb_identification
    '''
    kb_json_1 = json_load('./dataset/kqapro/kb.json')
    kb_json_2 = json_load('./dataset/kqapro/kb.json')
    kb_obj = KB(kb_json_2)

    compare_recursively(kb_json_1, kb_json_2)
    print('No error during running.')
