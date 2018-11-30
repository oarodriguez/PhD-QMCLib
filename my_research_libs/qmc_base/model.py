from abc import ABCMeta, abstractmethod
from typing import NamedTuple, Optional

import numpy as np

__all__ = [
    'CoreFuncs',
    'CoreFuncsMeta',
    'PhysicalFuncs',
    'Spec',
    'SpecMeta',
    'SpecNT',
    'WFOptimizer'
]


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

    #: Functions to calculate the main physical properties of a model.
    phys_funcs: 'PhysicalFuncs'

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


class PhysicalFuncs(metaclass=ABCMeta):
    """Functions to calculate the main physical properties of a model."""

    __slots__ = ()

    #: The model spec these functions correspond to.
    spec: Spec

    #:  The core functions of the model.
    core_funcs: CoreFuncs

    @property
    @abstractmethod
    def wf_abs_log(self):
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


class WFOptimizer(metaclass=ABCMeta):
    """Class to optimize the trial-wave function of a model."""

    __slots__ = ()

    #: The spec of the model.
    spec: Spec

    #: The system configurations used for the minimization process.
    sys_conf_set: np.ndarray

    #: The initial wave function values. Used to calculate the weights.
    ini_wf_abs_log_set: np.ndarray

    #: The energy of reference to minimize the variance of the local energy.
    ref_energy: Optional[float]

    @abstractmethod
    def update_spec(self, *args, **kwargs):
        """Updates the initial model spec."""
        pass

    @property
    @abstractmethod
    def principal_function_bounds(self):
        pass

    @abstractmethod
    def principal_function(self, *args, **kwargs):
        """Evaluates the quantity to minimize (optimize)."""
        pass

    @abstractmethod
    def exec(self):
        """Starts the minimization process."""
        pass
