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
    标注函数已弃用：调用时发出 warning，并保持原函数签名/文档（wraps）。

    注意：`DeprecationWarning` 在默认过滤规则下常常不会显示；
    如需在命令行看到，可使用 `-Wd` 或设置 `PYTHONWARNINGS=default`。
    """

    def deco(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            warnings.warn(message, category=category, stacklevel=stacklevel)
            return func(*args, **kwargs)

        return wrapper

    return deco

