from abc import ABCMeta, abstractmethod
from collections import Mapping
from enum import Enum
from typing import NamedTuple, Type

from my_research_libs.utils import strict_update

__all__ = [
    'CoreFuncs',
    'CoreFuncsMeta',
    'Spec',
    'SpecMeta',
    'SpecNT',
    'ParamNameEnum',
    'ParamsSet'
]


class ParamNameEnum(str, Enum):
    """Base class for parameter enumerations.

    These parameters (the Enum elements) behave as strings. They can
    be used directly as their ``value`` attribute. In addition, they
    have a ``loc`` attribute that indicates the order in which they
    where defined.
    """

    def __new__(cls, value):
        """
        :param value:
        :return:
        """
        loc = len(cls.__members__)
        param = super().__new__(cls, value)
        param._loc_ = loc
        return param

    @property
    def loc(self):
        return self._loc_


class ParamsSet(Mapping):
    """Base class for parameters. Implements a read-only mapping
    interface.
    """
    # Enum with the allowed parameters.
    names: Type[Enum] = None

    # Enum with the default values (if necessary) of the parameters.
    defaults: Type[Enum] = None

    # Names are important as they restrict the set of
    # allowed parameters.
    __slots__ = (
        '_ord_names',
        '_data',
    )

    def __init__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        """
        self_names = self.names
        if self_names is None:
            raise TypeError("'names' attribute must not be None")
        elif not issubclass(self_names, Enum):
            raise TypeError("'names' attribute must be an enumeration")

        defaults = self.defaults
        if defaults is None:
            ext_data = dict(*args, **kwargs)
        else:
            ext_data = {}
            for default in defaults:
                default_name = default.name
                if default_name not in self_names.__members__:
                    raise KeyError("unexpected default: "
                                   "'{}'".format(default_name))
                name = self_names[default_name]
                default = defaults[name.name]
                ext_data[name.value] = default.value
            ext_data.update(*args, **kwargs)

        ord_names = [name.value for name in self_names]
        self_data = dict([(name.value, None) for name in self_names])
        strict_update(self_data, ext_data, full=True)

        # The order in which the mapping will be iterated.
        self._ord_names = tuple(ord_names)
        self._data = self_data

    def __getitem__(self, name):
        # Items come from the attributes.
        return self._data[name]

    def __len__(self):
        """"""
        return len(self._data)

    def __iter__(self):
        """"""
        return iter(self._ord_names)


class SpecNT(NamedTuple):
    """The parameters of the model."""
    pass


class SpecMeta(ABCMeta):
    """Metaclass for :class:`Spec` abstract base class."""
    pass


class Spec(metaclass=SpecMeta):
    """ABC for a QMC model specification.

    Represents the specification of a Quantum Monte Carlo model for a
    physical quantum system. This abstract base class that defines the
    most common methods/functions used in a QMC simulation to estimate
    the properties of a physical system.
    """
    __slots__ = ()

    @property
    @abstractmethod
    def boundaries(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def sys_conf_shape(self):
        pass

    @abstractmethod
    def get_sys_conf_buffer(self):
        pass

    @abstractmethod
    def init_get_sys_conf(self):
        pass

    @property
    @abstractmethod
    def as_nt(self):
        pass

    @property
    @abstractmethod
    def cfc_spec_nt(self):
        """Tuple to be used as part of the arguments of the functions
        in the corresponding :class:`CoreFuncs` instance of the model
        (:attr:`Spec.core_funcs` attribute).
        """
        pass


class CoreFuncsMeta(ABCMeta):
    """Metaclass for :class:`CoreFuncs` abstract base class."""
    pass


class CoreFuncs(metaclass=CoreFuncsMeta):
    """"""
    __slots__ = ()

    @property
    @abstractmethod
    def real_distance(self):
        pass

    @property
    @abstractmethod
    def wf_abs_log(self):
        pass

    @property
    @abstractmethod
    def wf_abs(self):
        pass

    @property
    @abstractmethod
    def delta_wf_abs_log_kth_move(self):
        pass

    @property
    @abstractmethod
    def ith_drift(self):
        pass

    @property
    @abstractmethod
    def delta_ith_drift_kth_move(self):
        pass

    @property
    @abstractmethod
    def ith_energy(self):
        pass

    @property
    @abstractmethod
    def energy(self):
        pass

    @property
    @abstractmethod
    def ith_energy_and_drift(self):
        pass

    @property
    @abstractmethod
    def one_body_density(self):
        pass

    @property
    @abstractmethod
    def structure_factor(self):
        pass
