import os
import pickle
import psutil
import logging

from typing import Callable, List, Any, Tuple
from time import time
from functools import wraps


BASE_LOGGING_LEVEL = logging.WARNING


def grouped(iterable: List[Any], n: int):
    """Groups subsequent values according to the formula: `(s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ...`

    Args:
        iterable (list): List of objects.
        n (int): Number of elements in group.
    """
    return zip(*[iter(iterable)] * n)


def flatten(x: List[Any]) -> List[Any]:
    """Returns flattened list.

    Args:
        x (list): Nested Python list.

    Returns:
        list: Flattened Python list.
    """
    return [i for sl in x for i in sl]


def save_object(obj: Any, filename: str) -> None:
    """Saves object in Pickle format.

    Args:
        obj : Python object to save.
        filename (str): Save path.
    """
    with open(f"{filename}.pickle", "wb") as handle:
        pickle.dump(obj, handle)


def read_object(filename: str) -> Any:
    """Reads object from pickle format.

    Args:
        filename (str): Reading path.
    """
    with open(f"{filename}.pickle", "rb") as handle:
        return pickle.load(handle)


def timing(f: Callable) -> Tuple[Any, float]:
    """Decorator that measure s execution time of a selected function.

    Args:
        f (Callable): Function.

    Returns:
        Tuple[Any, float]: Returns tuple with original function output and the time in seconds as a floating point number.
    """

    @wraps(f)
    def wrapper(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        return result, te - ts

    return wrapper


def print_memory_usage() -> None:
    """Function shows actual RAM usage."""
    print(psutil.Process(os.getpid()).memory_info().rss / 1024**2)
