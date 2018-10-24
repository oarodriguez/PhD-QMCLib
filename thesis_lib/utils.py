import os
from abc import ABCMeta
from datetime import datetime
from itertools import product
from multiprocessing import current_process
from time import time
from typing import (
    Dict, ItemsView, Iterable, Mapping, MutableMapping, Sequence, Tuple,
    TypeVar, Union
)

import numpy as np
from decorator import contextmanager
from numba.config import reload_config
from tzlocal import get_localzone


class CachedProperty(object):
    """An object that acts as a property attribute with its value cached."""
    # Do not inherit from property...

    # TODO: Add __repr__

    def __init__(self, func, name=None):
        """

        :param name:
        :param func:
        """
        self.__name__ = name or func.__name__
        self.func = func
        self.__doc__ = func.__doc__

    def __get__(self, obj: 'Cached', type_=None):
        """

        :param obj:
        :param type_:
        :return:
        """
        if obj is None:
            return self

        name = self.__name__
        func = self.func

        if not isinstance(obj, Cached):
            raise TypeError

        if name not in obj.__cached_properties__:
            msg = "object has no property '{}'".format(name)
            raise AttributeError(msg)

        if name not in obj._cached_data:
            prop_value = func(obj)
            obj._cached_data[name] = prop_value
            return prop_value

        return obj._cached_data[name]

    def __set__(self, obj, value):
        """Make read-only data descriptor."""
        raise AttributeError("can't set attribute")

    def __delete__(self, obj):
        """Make read-only data descriptor."""
        raise AttributeError("can't delete attribute")


def cached_property(func, name=None):
    """Decorates a method with a single self argument to behave as a
    property attribute, i.e., a :class:`CachedProperty` instance.

    :param func:
    :param name:
    :return:
    """
    return CachedProperty(func, name)


class CachedMeta(ABCMeta):
    """Metaclass for the :class:`Cached` type. It picks every attribute
    that is a :class:`CachedProperty` instance and stores a reference to
    each of them in a special class attribute.
    """

    def __new__(mcs, name, bases, namespace):
        """

        :param name:
        :param bases:
        :param namespace:
        :return:
        """
        cls = super().__new__(mcs, name, bases, namespace)

        properties = {}
        # NOTE: Whats is the difference between using the namespace?
        for name_ in dir(cls):
            obj = getattr(cls, name_)
            if isinstance(obj, CachedProperty):
                properties[obj.__name__] = obj

        cls.__cached_properties__ = properties
        return cls


class Cached(metaclass=CachedMeta):
    """Type that adds support for attribute caching."""

    __slots__ = ('_cached_data',)

    def __init__(self, *args, **kwargs):
        """Mapping to hold cached data"""
        self._cached_data = {}


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


def get_random_rng_seed():
    """Generates a random integer to be used as seed for a RNG. It uses
    the current process id (integer) plus the current time in milliseconds
    to seed the ``numpy`` RNG, and pick a random integer from a uniform
    probability distribution as a new seed. This way the integer is
    different for every process when working with the ``multiprocessing``
    module.

    :return: A random integer
    """
    # TODO: This function needs more analysis :-\
    i32_max = np.iinfo(np.int32).max
    np.random.seed(
            int(current_process().pid + int(time() * 1000) % i32_max)
    )

    return np.random.randint(0, high=i32_max - 1, dtype=np.int64)


def missing_message(missing: Iterable):
    """Gives a nice "missing ..." formatted string with the corresponding
    values in ``missing``.
    """
    missing = ["'{}'".format(value) for value in missing]
    len_missing = len(missing)
    joined_missing = ' and '.join(missing)
    if len_missing == 1:
        message_fmt = 'missing {} required item: {}'
    else:
        message_fmt = 'missing {} required items: {}'
    # NOTE: KeyError will display the message in quotes :|
    return message_fmt.format(len_missing, joined_missing)


def strict_update(obj: MutableMapping,
                  mapping: Union[Mapping, Sequence, ItemsView],
                  full: bool = False):
    """Updates ``obj`` with the items in ``mapping``. Raises ``KeyError``
    if an item in ``mapping`` is not in ``obj``. If ``full = True``, the
    update process expect that all the items of ``obj`` are in ``mapping``.
    """
    items = {}
    mapping = dict(mapping)
    for name in mapping:
        if name not in obj:
            raise KeyError("unexpected item: '{}'".format(name))
        else:
            items[name] = mapping[name]
    missing = obj.keys() - mapping.keys()
    if missing and full:
        raise KeyError(missing_message(missing))

    obj.update(items)
