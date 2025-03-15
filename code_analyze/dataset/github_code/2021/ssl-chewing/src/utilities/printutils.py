from typing import NoReturn, List


def _get_line_length(s: str) -> int:
    assert isinstance(s, str)

    lines: List[str] = s.splitlines()

    return max([len(line) for line in lines])


def print_with_header(message: str, header: str = '', spaces: int = 3) -> NoReturn:
    assert isinstance(message, str)
    assert isinstance(header, str)
    assert isinstance(spaces, int)

    n: int = 2 * spaces + len(message)
    m: int = (n - len(header) - 2)
    m1: int = int(m / 2)
    m2: int = int(m - m1)

    print(' ')
    print('-' * m1 + ' ' + header + ' ' + '-' * m2)
    print(' ' * spaces + message + ' ' * spaces)


def get_tabulate_header(tbl: str, caption: str, toprule: bool = False, midrule: bool = True) -> str:
    assert isinstance(tbl, str)
    assert isinstance(caption, str)
    assert isinstance(toprule, bool)
    assert isinstance(midrule, bool)

    tbl_lines: List[str] = tbl.splitlines()
    length0: int = max([len(line) for line in tbl_lines])
    length: int = max([length0, len(caption)])

    m: int = length - len(caption)
    m1: int = int(m / 2)
    m2: int = int(m - m1)

    header: str = (' ' * m1) + caption + (' ' * m2)

    if toprule:
        header = ("#" * length) + "\n" + header

    if midrule:
        header += "\n" + ("-" * length)

    return header


def pretty_lopo_header(ids: List, i: int) -> str:
    assert isinstance(ids, List)
    assert isinstance(i, int)

    return 'ID = ' + str(ids[i]) + ' (' + str(i + 1) + '/' + str(len(ids)) + ')'
