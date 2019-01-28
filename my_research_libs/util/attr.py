"""
    my_research_libs.util.attr
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    Module with symbols and routines used with the attrs library.
"""
from pathlib import Path
from typing import Any

import attr
import numpy as np

#  Common validators.

float_validator = attr.validators.instance_of((float, int))
int_validator = attr.validators.instance_of(int)
str_validator = attr.validators.instance_of(str)
bool_validator = attr.validators.instance_of(bool)

opt_float_validator = attr.validators.optional(float_validator)
opt_int_validator = attr.validators.optional(int_validator)
opt_str_validator = attr.validators.optional(str_validator)
opt_bool_validator = attr.validators.optional(bool_validator)

path_validator = attr.validators.instance_of(Path)
opt_path_validator = attr.validators.optional(path_validator)

# These are the possible types of the data read from HDF5 files.
_valid_int_types = \
    (int, np.int, np.int8, np.int16, np.int32, np.int64)


# TODO: More tests for this function...
def int_converter(value: Any):
    """Converter for data that may come from an HDF5 file.

    :param value:
    :return:
    """
    # Data read from HDF5 files with h5py has a numpy dtype. Before using
    # as an attribute, we have to convert it to a python integer.
    if isinstance(value, _valid_int_types):
        return int(value)

    return value


# These are the possible types of the data read from HDF5 files.
_valid_bool_types = bool, np.bool_


def bool_converter(value: Any):
    """Converter for data that may come from an HDF5 file.

    :param value:
    :return:
    """
    # Data read from HDF5 files with h5py has a numpy dtype. Before using
    # as an attribute, we have to convert it to a python bool.
    if isinstance(value, _valid_bool_types):
        return bool(value)

    return value


opt_int_converter = attr.converters.optional(int_converter)
opt_float_converter = attr.converters.optional(float)
opt_bool_converter = attr.converters.optional(bool_converter)
