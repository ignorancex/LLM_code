
import re
from transformers import BartTokenizer


def is_digit(s):
    return re.match(r'^Ġ?[0-9]+$', s)

def analyze_tokens():
    tokenizer = BartTokenizer.from_pretrained('./pretrained/bart-base')
    all_tokens = tuple(map(tokenizer.convert_ids_to_tokens, range(tokenizer.vocab_size)))

    digit_tokens = tuple(filter(is_digit, all_tokens))
    print(len(digit_tokens))


special_tokens = {'quantity': {'.', '-', 'e'},
                  'year': {'-'},
                  'date': {'/'}}


# quantity examples: [1.2e-06, 9e-06, 4.3e-06]
quantity_prefix_regex = r'^[+-]?[0-9]*(\.[0-9]*)?(e([+-][0-9]*)?)?$'
year_prefix_regex = r'^[+-]?[0-9]*$'
date_prefix_regex_with_slash = r'^[0-9]{1,4}(/([0-9]{1,2}(/([0-9]{1,2})?)?)?)?$'


def check_prefix(regex, text, expected=True):
    if expected:
        assert all(re.match(regex, text[:length])
                   for length in range(1, len(text) + 1))
    else:
        assert any(not re.match(regex, text[:length])
                   for length in range(1, len(text) + 1))


if __name__ == '__main__':
    analyze_tokens()

    check_prefix(quantity_prefix_regex, '4.3e-06')
    check_prefix(quantity_prefix_regex, '.3e+10')
    check_prefix(quantity_prefix_regex, '20e+10')
    check_prefix(quantity_prefix_regex, '-20e+10')
    check_prefix(quantity_prefix_regex, '-.3e-2')
    check_prefix(quantity_prefix_regex, '123')
    # check_prefix(quantity_prefix_regex, 'e-2', False)

    check_prefix(year_prefix_regex, '-1')
    check_prefix(year_prefix_regex, '-12')
    check_prefix(year_prefix_regex, '-11231233432')
    check_prefix(year_prefix_regex, '1')
    check_prefix(year_prefix_regex, '123123123')
    check_prefix(year_prefix_regex, '+12321')
    check_prefix(year_prefix_regex, '1999')
    check_prefix(year_prefix_regex, '1999.0', False)

    check_prefix(date_prefix_regex_with_slash, '1/1/1')
    check_prefix(date_prefix_regex_with_slash, '1/01/01')
    check_prefix(date_prefix_regex_with_slash, '11/12/13')
    check_prefix(date_prefix_regex_with_slash, '2222/12/13')
    check_prefix(date_prefix_regex_with_slash, '2222/12/13')
    check_prefix(date_prefix_regex_with_slash, '2222/12/133', False)
    check_prefix(date_prefix_regex_with_slash, '2222/012/13', False)
