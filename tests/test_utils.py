import os

import pytest

from phd_qmclib.utils import strict_update


def test_strict_update():
    """

    :return:
    """
    items = {'PATH', 'TMP', 'OS'}
    obj = {item: os.environ[item] for item in items}

    with pytest.raises(KeyError):
        # Unknown items.
        strict_update(obj, {'UNKNOWN_KEY': 1})

    with pytest.raises(KeyError):
        # Known item, but we want to update all
        strict_update(obj, {'PATH': 1}, full=True)

    other = {'PATH': 99, 'TMP': 88, 'OS': 77}
    strict_update(obj, other, full=True)
    strict_update(obj, other.items(), full=True)

    other_lc = [(name, value) for (name, value) in other.items()]
    strict_update(obj, other_lc, full=True)
    strict_update(obj, {'TMP': 10})
