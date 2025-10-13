from collections import defaultdict


def flatten(xss):
    # https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
    return [x for xs in xss for x in xs]


def recursive_default_dict():
    return defaultdict(recursive_default_dict)
