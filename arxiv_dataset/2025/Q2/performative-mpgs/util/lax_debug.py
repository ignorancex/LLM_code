from typing import Callable, TypeVar

T = TypeVar("T")


def fori_loop(lower: int, upper: int, body_fun: Callable[[int, T], T], init_val: T) -> T:
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def while_loop(cond_fun: Callable[[T], bool], body_fun: Callable[[T], T], init_val: T) -> T:
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val


def jit(func: Callable, **kwargs) -> Callable:
    return func
