# Written by Chun Kit Wong and CIRC under MIT license:
# https://github.com/wong-ck/DeepSegment/blob/master/LICENSE

import logging


def create_or_get_logger(
    name="deepsegment",
    level="INFO",
    format="%(asctime)-15s [%(levelname)-8s] %(name)-16s: %(message)s"
):
    _logger = logging.getLogger(name)

    if len(_logger.handlers) == 0:
        _logger = _create_stream_logger(name=name, level=level, format=format)

    return _logger


def _create_stream_logger(name, level, format):
    # create logger
    _logger = logging.getLogger(name)
    _logger.setLevel(level)

    # create stream handler
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # create formatter and add it to the handler
    formatter = logging.Formatter(format)
    ch.setFormatter(formatter)

    # add the handler to the logger
    _logger.addHandler(ch)

    return _logger


def print_dict(dict_in, logger, level=logging.INFO, indent=0):
    # determine length of longest key
    max_keylen = max([len(str(k)) for k in dict_in.keys()])

    # define message template
    msg = "  " * indent + "- {:}: {:}"

    # loop over each item in dictionary
    for k, v in dict_in.items():
        # if value is another dictionary, call function recursively;
        # else, just print
        if isinstance(v, dict):
            logger.log(level, msg.format(str(k).ljust(max_keylen, ' '), ''))
            print_dict(v, logger, level, indent + 1)
        else:
            logger.log(level, msg.format(str(k).ljust(max_keylen, ' '), v))

    return
