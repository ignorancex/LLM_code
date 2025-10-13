import re


def split_string_with_escapes(s:str)->list[str]:
    """
    Split a string into a list of tokens, where each token is a character or an escape sequence.
    :param s: a given string
    :return: a list of tokens
    """
    pattern = r'(\\[abfnrtv\'"\\]|.)'
    tokens = re.findall(pattern, s)
    return tokens


def escape_string(s: str) -> str:
    """
    Escape
    :param s: The string to escape.
    :return: The escaped string.
    """
    return s.encode('unicode_escape').decode('utf-8')

def unescape_string(s: str) -> str:
    """
    Unescape
    :param s: The string to unescape.
    :return: The unescaped string.
    """
    return s.encode('utf-8').decode('unicode_escape')

def extract_code_block(s: str) -> str:
    """
    Extract code from Markdown code block (```code```) from a string.
    :param s: The string to extract code from.
    :return: Code
    """
    match = re.search(r"```(?:\w+\n|\n|)(.*?)```", s, re.DOTALL)
    if match:
        return match.group(1)
    else:
        raise ValueError('No code block found in the given string.')