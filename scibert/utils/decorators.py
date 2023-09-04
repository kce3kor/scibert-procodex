"""Custom decorators utility

"""

import time
import functools


def debug(func):
    """Print the function signature and return value"""

    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]  # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)  # 3
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")  # 4
        return value

    return wrapper_debug


def timer(func):
    """Decorator function for timming funcitons"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start = time.perf_counter()
        value = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Time Taken :: {end - start:0.4f} seconds")
        return value

    return wrapper_timer
