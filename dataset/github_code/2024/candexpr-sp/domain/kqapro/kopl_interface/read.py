
import re
from kopl.kopl import KoPLEngine

from dhnamlib.pylib.decoration import variable, construct


def kopl_to_recursive_form(labeled_kopl_program):
    def parse(func_call_idx):
        func_call = labeled_kopl_program[func_call_idx]
        return dict(function=func_call['function'],
                    inputs=func_call['inputs'],
                    dependencies=list(map(parse, func_call['dependencies'])))

    return parse(len(labeled_kopl_program) - 1)


_kopl_function_regex = re.compile(r"context\.([a-zA-Z]+)")


def extract_kopl_func(expr):
    # e.g. expr == "(context.VerifyDate @2 @0 @1)"
    match_obj = _kopl_function_regex.search(expr)
    if match_obj:
        kopl_func =  match_obj.group(1)
        assert kopl_func in kopl_function_names
        return kopl_func
    else:
        assert match_obj is None
        return None


def is_camel_case(s):
    # https://stackoverflow.com/a/10182901
    return s != s.lower() and s != s.upper() and "_" not in s


@variable
@construct(set)
def kopl_function_names():
    for attr in dir(KoPLEngine):
        if (
                is_camel_case(attr) and
                attr[0].upper() == attr[0] and
                callable(getattr(KoPLEngine, attr))
            ):
            yield attr


def number_to_quantity_and_unit(number):
    # this is the same code from kopl.kopl.KoPLEngine._parse_key_value

    if ' ' in number:
        splits = number.split()
        quantity = splits[0]
        unit = ' '.join(splits[1:])
    else:
        quantity = number
        unit = '1'
    return quantity, unit


def classify_value_type(context, key, value):
    # Modified from "kopl.kopl.KoPLEngine._parse_key_value"
    typ = context.kb.key_type[key]

    if typ == 'string':
        return 'string'
    elif typ == 'quantity':
        return 'number'
    else:
        assert typ == 'date'
        if '/' in value or ('-' in value and '-' != value[0]):
            return 'date'
        else:
            return 'year'
