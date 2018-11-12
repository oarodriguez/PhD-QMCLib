from abc import ABCMeta, abstractmethod
from collections import Callable, Mapping
from enum import Enum
from typing import Callable as TCallable, NamedTuple, Sequence, Type

import numpy as np

from my_research_libs.utils import strict_update

__all__ = [
    'CoreFuncs',
    'CoreFuncsMeta',
    'GUFunc',
    'GUFuncMeta',
    'Spec',
    'SpecMeta',
    'SpecNT',
    'ParamNameEnum',
    'ParamsSet',
    'QMCFuncsNames'
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

    @property
    @abstractmethod
    def core_funcs(self):
        """Performance-critical (JIT-compiled) implementations of the basic
        QMC functions associated with the model.
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


# NOTE: What are the implications of mixing str and Enum?
# NOTE: Do we need this enum?
class QMCFuncsNames(str, Enum):
    """`Enum` with the most common function names (possibly)
    available in a Quantum Monte Carlo Kernel.
    """

    WF_ABS = 'wf_abs'
    I_WF_ABS_LOG = 'ith_wf_abs_log'
    WF_ABS_LOG = 'wf_abs_log'
    D_WF_ABS_LOG_K_M = 'delta_wf_abs_log_kth_move'
    D_I_DRIFT_K_M = 'delta_ith_drift_kth_move'

    ADV_CFG = 'advance_conf_func'
    ADV_DIFF_CFG = 'advance_diffuse_conf_func'

    I_L_E = 'ith_energy'
    L_E = 'energy'
    I_L_E_DRIFT = 'ith_energy_and_drift'
    L_E_DRIFT = 'energy_and_drift'

    I_L_OBD = 'ith_one_body_density'
    L_OBD = 'one_body_density'

    L_SF = 'structure_factor'

    L_TBC = 'two_body_correlation_func'


class GUFuncMeta(ABCMeta):
    """Metaclass for :class:`GUFunc` abstract base class."""
    pass


class GUFunc(Callable, metaclass=GUFuncMeta):
    """Interface to implement a callable object that behaves as
    a ``numpy`` **generalized universal function**.
    """
    #
    signatures: Sequence[str] = []
    layout: str = ''

    def __init__(self, base_func: TCallable[..., float],
                 target: str = None):
        """Initializer.

        :param base_func:
        :param target:
        """
        super().__init__()
        self._base_func = base_func
        self.target = target or 'parallel'

    @property
    @abstractmethod
    def as_elem_func_args(self):
        """Numba compiled function to get the elementary function arguments
        from an array. It returns one or more objects compatible with
        the ``elem_func`` signature.
        """
        # NOTE: Beware of array boundaries within a jit-function.
        # The gufunc will not raise any error if we access elements
        # outside the array with target='parallel'. In 'cpu' target it
        # is possible.
        pass

    @property
    @abstractmethod
    def elem_func(self):
        """Wrapper over the elementary function that should be vectorized.
        This wrapper is necessary as numba vectorized functions only accept
        numpy arrays as arguments.
        """
        pass

    @property
    @abstractmethod
    def core_func(self):
        """The internal generalized universal function."""
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> np.ndarray:
        """"""
        pass
