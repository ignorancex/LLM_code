
from itertools import chain
from pprint import pprint as pp


def parse_program_into_tree_form(labeled_program):
    # TODO: QFilter* input == output of Relate or Filter*

    def parse(func_call_idx):
        func_call = labeled_program[func_call_idx]
        return dict(function=func_call['function'],
                    inputs=func_call['inputs'],
                    dependencies=list(map(parse, func_call['dependencies'])))

    return parse(len(labeled_program) - 1)


def tree_form_repr(tree_form):
    if isinstance(tree_form, dict):
        return '({} {})'.format(tree_form['function'], ' '.join(
            map(tree_form_repr, chain(tree_form['inputs'], tree_form['dependencies']))))
    elif isinstance(tree_form, list):
        return '{}'.format(' '.join(map(tree_form_repr, tree_form)))
    elif isinstance(tree_form, str):
        return repr(tree_form)
    else:
        return tree_form


# q_filter_dependency = {}
dependencies = {}

def check_dependencies(labeled_program):
    def parse(func_call_idx):
        func_call = labeled_program[func_call_idx]

        # if func_call['function'].startswith('QFilter'):
        for idx in func_call['dependencies']:
            dependencies.setdefault(func_call['function'], set()).add(
                labeled_program[idx]['function'])

        return dict(function=func_call['function'],
                    inputs=func_call['inputs'],
                    dependencies=list(map(parse, func_call['dependencies'])))

    return parse(len(labeled_program) - 1)


if __name__ == '__main__':
    from dhnamlib.pylib.filesys import json_load

    raw_dataset = json_load('./_tmp_data-indented/indented-train.json')
    tree_forms = []
    tree_form_reprs = []
    # for example in raw_dataset:
    #     # tree_forms.append(parse_program_into_tree_form(example['program']))
    #     tree_form_reprs.append(tree_form_repr(parse_program_into_tree_form(example['program'])))
    #     breakpoint()

    tree_form_reprs = tuple(map(check_dependencies, (example['program'] for example in raw_dataset)))
    pp(dependencies)

    output_functions = set(example['program'][-1]['function'] for example in raw_dataset)
    pp(output_functions)

    breakpoint()
    a = 100
