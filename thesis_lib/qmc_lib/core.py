from abc import abstractmethod
from collections import Callable, Mapping
from enum import Enum
from typing import Callable as TCallable, Sequence, Type

from thesis_lib.utils import Cached, CachedMeta, strict_update

__all__ = [
    'GUFunc',
    'GUFuncMeta',
    'Model',
    'ModelFuncs',
    'ModelFuncsMeta',
    'ModelMeta',
    'ParamNameEnum',
    'ParamsSet',
    'QMCFuncsNames'
]


class ParamNameEnum(str, Enum):
    """Base class for parameter enumerations.

    These parameters (the Enum elements) behave as strings. They can
    be used directly as their ``value`` attribute. In addition, they
    have a ``slot`` attribute that indicates the order in which they
    where defined.
    """

    def __new__(cls, value):
        """
        :param value:
        :return:
        """
        slot = len(cls.__members__)
        param = super().__new__(cls, value)
        param._slot_ = slot
        return param

    @property
    def slot(self):
        return self._slot_


class ParamsSet(Mapping):
    """Base class for parameters. Implements a read-only mapping
    interface.
    """
    #
    names: Type[Enum] = None

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
        if self.names is None:
            raise TypeError("'names' attribute must not be None")
        elif not issubclass(self.names, Enum):
            raise TypeError("'names' attribute must be an enumeration")

        ord_names = [name.value for name in self.names]
        data = dict([(name, None) for name in ord_names])
        strict_update(data, dict(*args, **kwargs), full=True)

        # The order in which the mapping will be iterated.
        self._ord_names = tuple(ord_names)
        self._data = data

    def __getitem__(self, name):
        # Items come from the attributes.
        return self._data[name]

    def __len__(self):
        """"""
        return len(self._data)

    def __iter__(self):
        """"""
        return iter(self._ord_names)


class ModelMeta(CachedMeta):
    """Metaclass for :class:`Model` abstract base class."""
    pass


class Model(Cached, metaclass=ModelMeta):
    """Represents a Quantum Monte Carlo model for a physical quantum
    system. This abstract base class that defines the most common
    methods/functions used in a QMC simulation to estimate the properties
    of a physical system.
    """

    @property
    @abstractmethod
    def boson_number(self):
        pass

    @property
    @abstractmethod
    def supercell_size(self):
        pass

    @property
    @abstractmethod
    def boundaries(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def params(self):
        pass

    @property
    @abstractmethod
    def args(self):
        pass

    @property
    @abstractmethod
    def var_params(self):
        return

    @property
    @abstractmethod
    def var_args(self):
        pass

    @property
    @abstractmethod
    def num_sys_conf_slots(self):
        pass

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
    def wf_args(self):
        pass

    @property
    @abstractmethod
    def var_params_bounds(self):
        pass


class ModelFuncsMeta(CachedMeta):
    """Metaclass for :class:`ModelFuncs` abstract base class."""
    pass


class ModelFuncs(Cached, metaclass=ModelFuncsMeta):
    """"""

    @property
    @abstractmethod
    def boson_number(self):
        pass

    @property
    @abstractmethod
    def supercell_size(self):
        pass

    @property
    @abstractmethod
    def boundaries(self):
        raise NotImplementedError

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
    def drift(self):
        pass

    @property
    @abstractmethod
    def delta_ith_drift_kth_move(self):
        pass

    @property
    @abstractmethod
    def energy(self):
        pass

    @property
    @abstractmethod
    def energy_and_drift(self):
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


class GUFuncMeta(CachedMeta):
    """Metaclass for :class:`GUFunc` abstract base class."""
    pass


class GUFunc(Cached, Callable, metaclass=GUFuncMeta):
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
    def __call__(self, *args, **kwargs):
        """"""
        pass
