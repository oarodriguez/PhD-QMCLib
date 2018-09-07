import os
from datetime import datetime
from itertools import product
from typing import Dict, ItemsView, Iterable, Sequence, Tuple, TypeVar, Union

from decorator import contextmanager
from numba.config import reload_config
from tzlocal import get_localzone


def now(local: bool = True) -> datetime:
    """

    :param local: Whether or not to use the local OS timezone.
        Defaults to ``True``.
    :return:
    """
    tz = None if local is None else get_localzone()
    return datetime.now(tz)


def now_to_string(fmt: str, local: bool = True) -> str:
    """Returns the current date and time as a string
    in the specified format.

    :param fmt: Format string for ``strftime`` datetime method.
    :param local: Whether or not to use the local OS timezone.
        Defaults to ``True``.
    :return: A datetime string.
    """
    return now(local).strftime(fmt)


def dated_path(date: datetime,
               roots: Tuple[str] = None):
    """

    :param date:
    :param roots:
    :return:
    """
    roots = roots or ()
    assert '..' not in roots

    y, m, d = date.strftime('%Y-%m-%d').split('-')
    dp = os.path.join(*roots, y, m, d)
    return dp


def make_dated_data_dir(date):
    """

    :param date:
    :return:
    """
    base = os.getenv('SCI_PROJECT_DATA_PATH')
    dp = dated_path(date, roots=(base,))
    return os.makedirs(dp)


KT = TypeVar('KT', str, Tuple[str, ...])
VT = TypeVar('VT', Iterable, Tuple[Iterable, ...])


def items_to_mesh(items: Union[Sequence[Tuple[KT, VT]],
                               ItemsView[KT, VT]],
                  fixed: Dict[str, object] = None):
    """

    :param items:
    :param fixed:
    :return:
    """
    names, ranges = zip(*items)  # type:
    fixed = fixed or {}

    assert len(names) == len(ranges)

    # Work over the ordered fixed names.
    # TODO: Check for duplicated names in names.
    fixed_names = tuple(fixed.keys())
    ranges = tuple(ranges) + tuple((fixed[k],) for k in fixed_names)
    names = tuple(names) + fixed_names

    elements = product(*ranges)
    for e in elements:
        e_items = []
        for i, ev in enumerate(e):
            ek = names[i]
            if isinstance(ev, Tuple) and ek not in fixed.keys():
                e_items.extend([(ek[j], esv) for j, esv in enumerate(ev)])
            else:
                e_items.append((ek, ev))

        yield e_items


def mapping_to_mesh(mapping: Dict[KT, VT],
                    fixed: Dict[str, object] = None):
    """

    :param mapping:
    :param fixed:
    :return:
    """
    yield from items_to_mesh(mapping.items(), fixed=fixed)


@contextmanager
def numba_env_vars(**env_vars):
    """Context manager that updates the **numba** environment variables
    with the keyword dictionary ``env_vars``. Any keyword that starts
    with ``NUMBA_`` will override the corresponding key in the system
    environment dictionary. It resets the environment variables to its
    original values upon exit.

    :param env_vars: Keyword dictionary.
    :return:
    """
    os_environ = os.environ
    old_env_vars = {}
    new_env_vars = {}

    for key, value in env_vars.items():
        if key.startswith('NUMBA_'):
            new_env_vars[key] = str(value)
            env_var = os_environ.get(key, None)
            if env_var is not None:
                old_env_vars[key] = env_var

    os_environ.update(new_env_vars)
    reload_config()
    #
    yield
    #
    for key in new_env_vars.keys():
        del os_environ[key]
    os_environ.update(old_env_vars)
    reload_config()
