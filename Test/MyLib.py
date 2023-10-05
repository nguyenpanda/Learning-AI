import time
from typing import Callable

def timeChecking(func: Callable, *args: any, **kwargs: dict) -> float:
    tic = time.time()
    func(*args, **kwargs)
    toc = time.time()
    return toc - tic