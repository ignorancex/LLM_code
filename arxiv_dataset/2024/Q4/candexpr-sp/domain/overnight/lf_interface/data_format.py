
import re


delimiter_regex = re.compile(r'([() ])')

def parse_expr(expr):
    tokens = tuple(
        token for token in delimiter_regex.split(expr)
        if token != '')

    def recurse(index):
        parse = []
        while index < len(tokens):
            token = tokens[index]
            index += 1
            if token == '(':
                index, sub_parse = recurse(index)
                parse.append(sub_parse)
            elif token == ')':
                return index, tuple(parse)
            elif token == ' ':
                pass
            else:
                parse.append(token)
        breakpoint()
        raise Exception('Parentheses are not matched')

    start_index = 0
    while tokens[start_index] == ' ':
        start_index += 1

    assert tokens[start_index] == '('

    index, parse = recurse(start_index + 1)
    return parse

