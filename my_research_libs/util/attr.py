"""
    my_research_libs.util.attr
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    Module with symbols and routines used with the attrs library.
"""

import attr

#  Common validators.

float_validator = attr.validators.instance_of((float, int))
int_validator = attr.validators.instance_of(int)
str_validator = attr.validators.instance_of(str)
bool_validator = attr.validators.instance_of(bool)

opt_float_validator = attr.validators.optional(float_validator)
opt_int_validator = attr.validators.optional(int_validator)
opt_str_validator = attr.validators.optional(str_validator)
opt_bool_validator = attr.validators.optional(bool_validator)
