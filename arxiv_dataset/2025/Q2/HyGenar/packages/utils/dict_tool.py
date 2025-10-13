def merge_dict_by_range(d: dict[int, int], start: int, end: int) -> int:
    """
    Merge the values of the dictionary within the range of start and end.
    :param d: a dictionary with integer keys and integer values
    :param start: start key (inclusive)
    :param end: end key (inclusive)
    :return: merged value
    """
    if not d:
        return 0
    result = sum(value for key, value in d.items() if start <= key <= end)
    return result
