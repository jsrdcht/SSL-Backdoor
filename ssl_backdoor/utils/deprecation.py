import functools
import warnings
from typing import Callable, ParamSpec, TypeVar, Type

P = ParamSpec("P")
R = TypeVar("R")


def deprecated(
    message: str,
    *,
    category: Type[Warning] = DeprecationWarning,
    stacklevel: int = 2,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Marks the function as deprecated, issues a warning on call, and keeps the original signature/documentation via wraps.

    Note: `DeprecationWarning` is often hidden under default warning filters.
    Use -Wd or set PYTHONWARNINGS=default to show warnings in the command line.
    """

    def deco(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            warnings.warn(message, category=category, stacklevel=stacklevel)
            return func(*args, **kwargs)

        return wrapper

    return deco

