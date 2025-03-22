
class UndefinedValueError(Exception):
    pass


def UVE(msg=None):
    args = (msg,) if msg is not None else ()
    raise UndefinedValueError(*args)
