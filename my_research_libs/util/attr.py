"""
    my_research_libs.util.attr
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    Module with symbols and routines used with the attrs library.
"""
import typing as t
from abc import ABCMeta
from pathlib import Path

import attr
import numpy as np

#  Common validators and converters.

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

seq_validator = attr.validators.instance_of(t.Sequence)

# These are the possible types of the data read from HDF5 files.
_valid_int_types = \
    (int, np.int, np.int8, np.int16, np.int32, np.int64)


# TODO: More tests for this function...
def int_converter(value: t.Any):
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


def bool_converter(value: t.Any):
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


@attr.s(auto_attribs=True)
class Record(metaclass=ABCMeta):
    """The parameters of the model."""

    @classmethod
    def get_dtype(cls):
        """Build the numpy dtype for the params object."""
        return np.dtype(cls.get_dtype_fields())

    @classmethod
    def get_dtype_fields(cls) -> t.Sequence[t.Tuple[str, np.dtype]]:
        """Retrieve the fields of the numpy dtype."""
        return [(field.name, field.type) for field in attr.fields(cls)]

    # NOTE: Mask the return type as a Params instance.
    def as_record(self) -> 'Record':
        """Return the params as a numpy structured array."""
        # NOTE 1: Return the first element of the array to simplify the
        #  access to the elements of the parameters.
        # NOTE 2: Should we return an instance of numpy.rec.array? These
        #  arrays are intended to be used in code compiled in nopython
        #  mode after all.
        return np.array([attr.astuple(self)], dtype=self.get_dtype())[0]
