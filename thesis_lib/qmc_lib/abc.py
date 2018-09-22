from abc import abstractmethod
from collections import Callable
from enum import Enum
from typing import Callable as TCallable, Sequence

from thesis_lib.utils import Cached, CachedMeta

__all__ = [
    'GUFuncBase',
    'GUFuncBaseMeta',
    'QMCFuncsBase',
    'QMCFuncsMeta',
    'QMCFuncsNames',
    'QMCModelBase',
    'QMCModelMeta'
]


class QMCModelMeta(CachedMeta):
    """Metaclass for :class:`QMCModelBase` abstract base class."""
    pass


class QMCModelBase(Cached, metaclass=QMCModelMeta):
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
    def var_params(self):
        return

    @property
    @abstractmethod
    def num_boson_conf_slots(self):
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
    def wf_params(self):
        pass

    @property
    @abstractmethod
    def var_params_bounds(self):
        pass


class QMCFuncsMeta(CachedMeta):
    """Metaclass for :class:`QMCFuncsBase` abstract base class."""
    pass


class QMCFuncsBase(Cached, metaclass=QMCFuncsMeta):
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
    def delta_ith_drift_kth_move(self):
        pass

    @property
    @abstractmethod
    def energy(self):
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
    L_E_BUFFER = 'energy_to_buffer'

    I_L_OBD = 'ith_one_body_density'
    L_OBD = 'one_body_density'
    L_OBD_BUFFER = 'one_body_density_to_buffer'
    L_OBD_GUV = 'one_body_density_guv_func'

    L_SF = 'structure_factor'
    L_SF_GUV = 'structure_factor_func_guv'

    L_TBC = 'two_body_correlation_func'


class GUFuncBaseMeta(CachedMeta):
    """Metaclass for :class:`GUFuncBase` abstract base class."""
    pass


class GUFuncBase(Cached, Callable, metaclass=GUFuncBaseMeta):
    """Interface to implement a callable object that behaves as
    a ``numpy`` **generalized universal function**.
    """

    def __init__(self, base_func: TCallable[..., float],
                 signatures: Sequence[str],
                 layout: str, target: str = None):
        """Initializer.

        :param base_func:
        :param target:
        """
        super().__init__()
        self._base_func = base_func
        self.signatures = signatures
        self.layout = layout
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
