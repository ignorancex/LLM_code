# Decorators
from functools import wraps


def my_timer(orig_func):
    """
    Decorator to measure the time spent on the function
    Parameters
    ----------
        orig_func:
            FUnction to be analysed

    Returns
    -------

    """
    import time

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        if not kwargs["optimize"]:
            print("{} ran in: {} sec".format(orig_func.__name__, t2))
        return result

    return wrapper
