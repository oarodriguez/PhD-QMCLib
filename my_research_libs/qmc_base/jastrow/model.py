import typing as t
from abc import ABCMeta, abstractmethod
from enum import Enum, IntEnum, unique
from math import (cos, exp, fabs, log, sin)

import numpy as np
from cached_property import cached_property
from numba import guvectorize, jit

from .. import model
from ..utils import sign

__all__ = [
    'CFCSpecNT',
    'CoreFuncs',
    'CSWFOptimizer',
    'OBFParams',
    'Params',
    'PhysicalFuncs',
    'Spec',
    'SysConfSlot',
    'SysConfDistType',
    'TBFParams'
]


@unique
class SysConfSlot(IntEnum):
    """Slots to store the configuration of a single particle."""

    #: Position slot.
    pos: int = 0

    # Drift slot.
    drift: int = 1


class SysConfDistType(Enum):
    """The ways to arrange the positions of the system configuration."""
    RANDOM = 'random'
    REGULAR = 'regular'


DIST_RAND = SysConfDistType.RANDOM
DIST_REGULAR = SysConfDistType.REGULAR


class Params(model.Params, metaclass=ABCMeta):
    """The common fields a Jastrow model spec should implement."""
    boson_number: int
    supercell_size: float
    is_free: bool
    is_ideal: bool


class OBFParams(model.Params, metaclass=ABCMeta):
    """Fields of the one-body function spec.

    We declare this class to help with typing and nothing more. A concrete
    spec should be implemented for a concrete model. It is recommended
    to subclass this class to keep a logical structure in the code.
    """
    pass


class TBFParams(model.Params, metaclass=ABCMeta):
    """Fields of the two-body function spec.

    We declare this class to help with typing and nothing more. A concrete
    spec should be implemented for a concrete model. It is recommended
    to subclass this class to keep a logical structure in the code.
    """
    pass


class CFCSpecNT(t.NamedTuple):
    """The common structure of the spec of a core function."""
    model_params: Params
    obf_params: OBFParams
    tbf_params: TBFParams


class Spec(model.Spec):
    """Abstract Base Class that represents a Quantum Monte Carlo model
    with a trial-wave function of the Bijl-Jastrow type.
    """
    __slots__ = ()

    #: The number of bosons.
    boson_number: int

    #: The size of the QMC simulation box.
    supercell_size: float

    @property
    @abstractmethod
    def is_free(self) -> bool:
        """Tests if the spec represents a free system."""
        pass

    @property
    @abstractmethod
    def is_ideal(self) -> bool:
        """Tests if the spec represents an ideal system."""
        pass

    def get_sys_conf_buffer(self):
        """Creates an empty array/buffer to store the configuration
        of the particles of the system.

        :return: A ``np.ndarray`` with zero-filled entries.
        """
        sc_shape = self.sys_conf_shape
        return np.zeros(sc_shape, dtype=np.float64)

    @property
    @abstractmethod
    def params(self) -> Params:
        """"""
        pass

    @property
    @abstractmethod
    def obf_params(self) -> OBFParams:
        pass

    @property
    @abstractmethod
    def tbf_params(self) -> TBFParams:
        pass

    @property
    def cfc_spec_nt(self):
        """"""
        self_params = self.params.as_record()
        obf_params = self.obf_params.as_record()
        tbf_params = self.tbf_params.as_record()
        return CFCSpecNT(self_params, obf_params, tbf_params)


# Stubs to help with static type checking.
# noinspection PyUnusedLocal
def _one_body_func_stub(z: float, obf_params: OBFParams) -> float:
    pass


# noinspection PyUnusedLocal
def _two_body_func_stub(rz: float, tbf_params: TBFParams) -> float:
    pass


# noinspection PyUnusedLocal
def _potential_stub(z: float, params: Params) -> float:
    pass


class CoreFuncs(model.CoreFuncs):
    """Abstract Base Class that groups core, JIT-compiled, performance-critical
    functions to realize a Quantum Monte Carlo calculation for a QMC model
    with a trial-wave function of the Bijl-Jastrow type.
    """
    #
    sys_conf_slots = SysConfSlot

    @property
    @abstractmethod
    def one_body_func(self):
        """The one-body function definition."""
        return _one_body_func_stub

    @property
    @abstractmethod
    def two_body_func(self):
        """The two-body function definition."""
        return _two_body_func_stub

    @property
    @abstractmethod
    def one_body_func_log_dz(self):
        """One-body function logarithmic derivative."""
        return _one_body_func_stub

    @property
    @abstractmethod
    def two_body_func_log_dz(self):
        """Two-body function logarithmic derivative."""
        return _two_body_func_stub

    @property
    @abstractmethod
    def one_body_func_log_dz2(self):
        """One-body function second logarithmic derivative."""
        return _one_body_func_stub

    @property
    @abstractmethod
    def two_body_func_log_dz2(self):
        """Two-body function second logarithmic derivative."""
        return _two_body_func_stub

    @property
    @abstractmethod
    def potential(self):
        """The external potential."""
        return _potential_stub

    @cached_property
    def ith_wf_abs_log(self):
        """

        :return:
        """
        real_distance = self.real_distance
        pos_slot = int(SysConfSlot.pos)

        one_body_func = self.one_body_func
        two_body_func = self.two_body_func

        @jit(nopython=True)
        def _ith_wf_abs_log(i_: int, sys_conf: np.ndarray,
                            cfc_spec: CFCSpecNT):
            """Computes the variational wave function of a system of bosons in
            a specific configuration.

            :param i_:
            :param sys_conf:
            :param cfc_spec:
            :return:
            """
            model_params = cfc_spec.model_params
            obf_params = cfc_spec.obf_params
            tbf_params = cfc_spec.tbf_params
            ith_wf_abs_log = 0.

            if not model_params.is_free:
                # Gas subject to external potential.
                z_i = sys_conf[pos_slot, i_]
                obv = one_body_func(z_i, obf_params)
                ith_wf_abs_log += log(fabs(obv))

            if not model_params.is_ideal:
                # Gas with interactions.
                z_i = sys_conf[pos_slot, i_]
                nop = model_params.boson_number
                for j_ in range(i_ + 1, nop):
                    z_j = sys_conf[pos_slot, j_]
                    z_ij = real_distance(z_i, z_j, model_params)
                    tbv = two_body_func(fabs(z_ij), tbf_params)
                    ith_wf_abs_log += log(fabs(tbv))

            return ith_wf_abs_log

        return _ith_wf_abs_log

    @cached_property
    def wf_abs_log(self):
        """

        :return:
        """
        ith_wf_abs_log = self.ith_wf_abs_log

        @jit(nopython=True, nogil=True)
        def _wf_abs_log(sys_conf: np.ndarray,
                        cfc_spec: CFCSpecNT):
            """Computes the variational wave function of a system of bosons in
            a specific configuration.

            :param sys_conf:
            :param cfc_spec:
            :return:
            """
            model_params = cfc_spec.model_params
            wf_abs_log = 0.

            if model_params.is_free and model_params.is_ideal:
                return wf_abs_log

            nop = model_params.boson_number
            for i_ in range(nop):
                wf_abs_log += ith_wf_abs_log(i_, sys_conf, cfc_spec)
            return wf_abs_log

        return _wf_abs_log

    @cached_property
    def wf_abs(self):
        """

        :return:
        """
        wf_abs_log = self.wf_abs_log

        @jit(nopython=True, nogil=True)
        def _wf_abs(sys_conf: np.ndarray,
                    cfc_spec: CFCSpecNT):
            """Computes the variational wave function of a system of
            bosons in a specific configuration.

            :param sys_conf:
            :param cfc_spec:
            :return:
            """
            wf_abs_log_ = wf_abs_log(sys_conf, cfc_spec)
            return exp(wf_abs_log_)

        return _wf_abs

    @cached_property
    def delta_wf_abs_log_kth_move(self):
        """

        :return:
        """
        real_distance = self.real_distance
        pos_slot = int(SysConfSlot.pos)

        one_body_func = self.one_body_func
        two_body_func = self.two_body_func

        @jit(nopython=True)
        def _delta_wf_abs_log_kth_move(k_: int,
                                       z_k_delta: float,
                                       sys_conf: np.ndarray,
                                       cfc_spec: CFCSpecNT):
            """Computes the change of the logarithm of the wave function
            after displacing the `k-th` particle by a distance ``z_k_delta``.

            :param k_:
            :param z_k_delta:
            :param sys_conf:
            :param cfc_spec:
            :return:
            """
            model_params = cfc_spec.model_params
            obf_params = cfc_spec.obf_params
            tbf_params = cfc_spec.tbf_params
            delta_wf_abs_log = 0.

            # NOTE: Is it better to use this conditional from external loops?
            if model_params.is_free and model_params.is_ideal:
                return delta_wf_abs_log

            # k-nth particle position.
            z_k = sys_conf[pos_slot, k_]
            z_k_upd = z_k + z_k_delta

            if not model_params.is_free:
                # Gas subject to external potential.
                obv = one_body_func(z_k, obf_params)
                obv_upd = one_body_func(z_k_upd, obf_params)
                delta_wf_abs_log += log(fabs(obv_upd / obv))

            if not model_params.is_ideal:
                # Gas with interactions.
                nop = model_params.boson_number
                for i_ in range(nop):
                    if i_ == k_:
                        continue

                    z_i = sys_conf[pos_slot, i_]
                    r_ki = fabs(real_distance(z_k, z_i, model_params))
                    r_ki_upd = fabs(real_distance(z_k_upd, z_i, model_params))

                    tbv = two_body_func(r_ki, tbf_params)
                    tbv_upd = two_body_func(r_ki_upd, tbf_params)
                    delta_wf_abs_log += log(fabs(tbv_upd / tbv))

            return delta_wf_abs_log

        return _delta_wf_abs_log_kth_move

    @cached_property
    def ith_drift(self):
        """

        :return:
        """
        real_distance = self.real_distance
        pos_slot = int(SysConfSlot.pos)

        one_body_func_log_dz = self.one_body_func_log_dz
        two_body_func_log_dz = self.two_body_func_log_dz

        @jit(nopython=True)
        def _ith_drift(i_: int,
                       sys_conf: np.ndarray,
                       cfc_spec: CFCSpecNT):
            """Computes the local energy for a given configuration of the
            position of the bodies. The kinetic energy of the hamiltonian is
            computed through central finite differences.

            :param i_:
            :param sys_conf: The current configuration of the positions of the
                   particles.
            :param cfc_spec:
            :return: The local energy.
            """
            model_params = cfc_spec.model_params
            obf_params = cfc_spec.obf_params
            tbf_params = cfc_spec.tbf_params
            ith_drift = 0.

            # NOTE: Is it better to use this conditional from external loops?
            if model_params.is_free and model_params.is_ideal:
                return ith_drift

            # i-nth particle position.
            z_i = sys_conf[pos_slot, i_]

            if not model_params.is_free:
                # Case with external potential.
                ob_fn_ldz = one_body_func_log_dz(z_i, obf_params)
                ith_drift += ob_fn_ldz

            if not model_params.is_ideal:
                # Case with interactions.
                nop = model_params.boson_number
                for j_ in range(nop):
                    # Do not account diagonal terms.
                    if j_ == i_:
                        continue
                    z_j = sys_conf[pos_slot, j_]
                    z_ij = real_distance(z_i, z_j, model_params)
                    sgn = sign(z_ij)
                    tb_fn_ldz = two_body_func_log_dz(fabs(z_ij),
                                                     tbf_params) * sgn

                    ith_drift += tb_fn_ldz

            # Accumulate to the drift velocity squared magnitude.
            return ith_drift

        return _ith_drift

    @cached_property
    def drift(self):
        """

        :return:
        """
        # TODO: Rename to drift
        pos_slot = int(SysConfSlot.pos)
        drift_slot = int(SysConfSlot.drift)
        ith_drift = self.ith_drift

        @jit(nopython=True)
        def _drift(sys_conf: np.ndarray,
                   cfc_spec: CFCSpecNT,
                   result: np.ndarray = None):
            """

            :param sys_conf:
            :param cfc_spec:
            :param result:
            :return:
            """
            if result is None:
                result = np.zeros_like(sys_conf)

            # Set the position first, then the drift.
            result[pos_slot, :] = sys_conf[pos_slot, :]

            nop = cfc_spec.model_params.boson_number
            for i_ in range(nop):
                result[drift_slot, i_] = ith_drift(i_, sys_conf, cfc_spec)

            return result

        return _drift

    @cached_property
    def delta_ith_drift_kth_move(self):
        """

        :return:
        """
        real_distance = self.real_distance
        pos_slot = int(SysConfSlot.pos)

        one_body_func_log_dz = self.one_body_func_log_dz
        two_body_func_log_dz = self.two_body_func_log_dz

        @jit(nopython=True)
        def _delta_ith_drift_kth_move(i_: int, k_: int,
                                      z_k_delta: float,
                                      sys_conf: np.ndarray,
                                      cfc_spec: CFCSpecNT):
            """Computes the change of the i-th component of the drift
            after displacing the k-th particle by a distance ``z_k_delta``.

            :param i_:
            :param k_:
            :param z_k_delta:
            :param sys_conf:
            :param cfc_spec:
            :return:
            """
            model_params = cfc_spec.model_params
            obf_params = cfc_spec.obf_params
            tbf_params = cfc_spec.tbf_params

            z_k = sys_conf[pos_slot, k_]
            z_k_upd = z_k + z_k_delta

            if not i_ == k_:
                #
                delta_ith_drift = 0.

                if model_params.is_ideal:
                    return delta_ith_drift

                # Only a gas with interactions contributes in this case.
                z_i = sys_conf[pos_slot, i_]
                z_ki_upd = real_distance(z_k_upd, z_i, model_params)
                z_ki = real_distance(z_k, z_i, model_params)

                # TODO: Move th sign to the function.
                sgn = sign(z_ki)
                ob_fn_ldz = two_body_func_log_dz(fabs(z_ki),
                                                 tbf_params) * sgn

                sgn = sign(z_ki_upd)
                ob_fn_ldz_upd = two_body_func_log_dz(fabs(z_ki_upd),
                                                     tbf_params) * sgn

                delta_ith_drift += -(ob_fn_ldz_upd - ob_fn_ldz)
                return delta_ith_drift

            delta_ith_drift = 0.
            # NOTE: Is it better to use this conditional from external loops?
            if model_params.is_free and model_params.is_ideal:
                return delta_ith_drift

            if not model_params.is_free:
                #
                ob_fn_ldz = one_body_func_log_dz(z_k, obf_params)
                ob_fn_ldz_upd = one_body_func_log_dz(z_k_upd, obf_params)

                delta_ith_drift += ob_fn_ldz_upd - ob_fn_ldz

            if not model_params.is_ideal:
                # Gas with interactions.
                nop = model_params.boson_number
                for j_ in range(nop):
                    #
                    if j_ == k_:
                        continue

                    z_j = sys_conf[pos_slot, j_]
                    z_kj = real_distance(z_k, z_j, model_params)
                    z_kj_upd = real_distance(z_k_upd, z_j, model_params)

                    sgn = sign(z_kj)
                    tb_fn_ldz = two_body_func_log_dz(fabs(z_kj),
                                                     tbf_params) * sgn

                    sgn = sign(z_kj_upd)
                    tb_fn_ldz_upd = two_body_func_log_dz(fabs(z_kj_upd),
                                                         tbf_params) * sgn

                    delta_ith_drift += (tb_fn_ldz_upd - tb_fn_ldz)

            return delta_ith_drift

        return _delta_ith_drift_kth_move

    @cached_property
    def ith_energy(self):
        """

        :return:
        """
        real_distance = self.real_distance
        pos_slot = int(SysConfSlot.pos)
        # drift_slot = int(SysConfSlot.DRIFT_SLOT)

        potential = self.potential
        one_body_func_log_dz = self.one_body_func_log_dz
        two_body_func_log_dz = self.two_body_func_log_dz
        one_body_func_log_dz2 = self.one_body_func_log_dz2
        two_body_func_log_dz2 = self.two_body_func_log_dz2

        @jit(nopython=True)
        def _ith_energy(i_: int,
                        sys_conf: np.ndarray,
                        cfc_spec: CFCSpecNT):
            """Computes the local energy for a given configuration of the
            position of the bodies. The kinetic energy of the hamiltonian is
            computed through central finite differences.

            :param i_:
            :param sys_conf: The current configuration of the positions of the
                   particles.
            :param cfc_spec:
            :return: The local energy.
            """
            model_params = cfc_spec.model_params
            obf_params = cfc_spec.obf_params
            tbf_params = cfc_spec.tbf_params
            ith_energy = 0.

            if model_params.is_free and model_params.is_ideal:
                return ith_energy

            kin_energy = 0.
            pot_energy = 0
            ith_drift = 0.

            if not model_params.is_free:
                # Case with external potential.
                z_i = sys_conf[pos_slot, i_]
                ob_fn_ldz2 = one_body_func_log_dz2(z_i, obf_params)
                ob_fn_ldz = one_body_func_log_dz(z_i, obf_params)

                kin_energy += (-ob_fn_ldz2 + ob_fn_ldz ** 2)
                pot_energy += potential(z_i, model_params)
                ith_drift += ob_fn_ldz

            if not model_params.is_ideal:
                # Case with interactions.
                z_i = sys_conf[pos_slot, i_]
                nop = model_params.boson_number
                for j_ in range(nop):
                    # Do not account diagonal terms.
                    if j_ == i_:
                        continue

                    z_j = sys_conf[pos_slot, j_]
                    z_ij = real_distance(z_i, z_j, model_params)
                    sgn = sign(z_ij)

                    tb_fn_ldz2 = two_body_func_log_dz2(fabs(z_ij), tbf_params)
                    tb_fn_ldz = two_body_func_log_dz(fabs(z_ij),
                                                     tbf_params) * sgn

                    kin_energy += (-tb_fn_ldz2 + tb_fn_ldz ** 2)
                    ith_drift += tb_fn_ldz

            # Accumulate to the drift velocity squared magnitude.
            ith_drift_mag = ith_drift ** 2
            # Save the drift and avoid extra work.
            # sys_conf[drift_slot, i_] = ith_drift

            ith_energy = kin_energy - ith_drift_mag + pot_energy
            return ith_energy

        return _ith_energy

    @cached_property
    def energy(self):
        """

        :return:
        """
        ith_energy = self.ith_energy

        @jit(nopython=True, nogil=True)
        def _energy(sys_conf: np.ndarray,
                    cfc_spec: CFCSpecNT):
            """

            :param sys_conf:
            :param cfc_spec:
            :return:
            """
            energy = 0.
            nop = cfc_spec.model_params.boson_number
            for i_ in range(nop):
                energy += ith_energy(i_, sys_conf, cfc_spec)
            return energy

        return _energy

    @cached_property
    def ith_energy_and_drift(self):
        """

        :return:
        """
        real_distance = self.real_distance
        pos_slot = int(SysConfSlot.pos)

        potential = self.potential
        one_body_func_log_dz = self.one_body_func_log_dz
        two_body_func_log_dz = self.two_body_func_log_dz
        one_body_func_log_dz2 = self.one_body_func_log_dz2
        two_body_func_log_dz2 = self.two_body_func_log_dz2

        @jit(nopython=True)
        def _ith_energy_and_drift(i_: int,
                                  sys_conf: np.ndarray,
                                  cfc_spec: CFCSpecNT):
            """Computes the local energy for a given configuration of the
            position of the bodies. The kinetic energy of the hamiltonian is
            computed through central finite differences.

            :param i_:
            :param sys_conf: The current configuration of the positions of the
                   particles.
            :param cfc_spec:
            :return: The local energy and the drift.
            """
            model_params = cfc_spec.model_params
            obf_params = cfc_spec.obf_params
            tbf_params = cfc_spec.tbf_params
            ith_energy, ith_drift = 0., 0.

            if model_params.is_free and model_params.is_ideal:
                return ith_energy, ith_drift

            # Unpack the parameters.
            kin_energy = 0.
            pot_energy = 0
            ith_drift = 0.

            if not model_params.is_free:
                # Case with external potential.
                z_i = sys_conf[pos_slot, i_]
                ob_fn_ldz2 = one_body_func_log_dz2(z_i, obf_params)
                ob_fn_ldz = one_body_func_log_dz(z_i, obf_params)

                kin_energy += (-ob_fn_ldz2 + ob_fn_ldz ** 2)
                pot_energy += potential(z_i, model_params)
                ith_drift += ob_fn_ldz

            if not model_params.is_ideal:
                # Case with interactions.
                z_i = sys_conf[pos_slot, i_]
                nop = model_params.boson_number
                for j_ in range(nop):
                    # Do not account diagonal terms.
                    if j_ == i_:
                        continue

                    z_j = sys_conf[pos_slot, j_]
                    z_ij = real_distance(z_i, z_j, model_params)
                    sgn = sign(z_ij)

                    tb_fn_ldz2 = two_body_func_log_dz2(fabs(z_ij), tbf_params)
                    tb_fn_ldz = two_body_func_log_dz(fabs(z_ij),
                                                     tbf_params) * sgn

                    kin_energy += (-tb_fn_ldz2 + tb_fn_ldz ** 2)
                    ith_drift += tb_fn_ldz

            # Evaluate drift velocity squared magnitude.
            ith_drift_mag = ith_drift ** 2
            ith_energy = kin_energy - ith_drift_mag + pot_energy

            return ith_energy, ith_drift

        return _ith_energy_and_drift

    @cached_property
    def ith_one_body_density(self):
        """

        :return:
        """
        real_distance = self.real_distance
        pos_slot = int(SysConfSlot.pos)

        one_body_func = self.one_body_func
        two_body_func = self.two_body_func

        @jit(nopython=True)
        def _ith_one_body_density(i_: int,
                                  sz: float,
                                  sys_conf: np.ndarray,
                                  cfc_spec: CFCSpecNT):
            """Computes the logarithm of the local one-body density matrix
            for a given configuration of the position of the bodies and for a
            specified particle index.

            :param i_:
            :param sz:
            :param sys_conf:
            :param cfc_spec:
            :return:
            """
            model_params = cfc_spec.model_params
            obf_params = cfc_spec.obf_params
            tbf_params = cfc_spec.tbf_params
            ith_obd_log = 0.

            if model_params.is_free and model_params.is_ideal:
                return ith_obd_log

            # The local one-body density matrix is calculated as the
            # quotient of the wave function with the ``i_`` particle
            # shifted a distance ``sz` from its original position
            # divided by the wave function with the particles evaluated
            # in their original positions. To improve statistics, we
            # average over all possible particle displacements.
            if not model_params.is_free:
                #
                z_i = sys_conf[pos_slot, i_]
                z_i_sft = z_i + sz

                ob_fn = one_body_func(z_i, obf_params)
                ob_fn_sft = one_body_func(z_i_sft, obf_params)
                ith_obd_log += (log(ob_fn_sft) - log(ob_fn))

            if not model_params.is_ideal:
                # Interacting gas.
                z_i = sys_conf[pos_slot, i_]
                z_i_sft = z_i + sz
                nop = model_params.boson_number
                for j_ in range(nop):
                    #
                    if i_ == j_:
                        continue

                    z_j = sys_conf[pos_slot, j_]
                    z_ij = real_distance(z_i, z_j, model_params)
                    tb_fn = two_body_func(fabs(z_ij), tbf_params)

                    # Shifted difference.
                    z_ij = real_distance(z_i_sft, z_j, model_params)
                    tb_fn_shift = two_body_func(fabs(z_ij), tbf_params)

                    ith_obd_log += (log(tb_fn_shift) - log(tb_fn))

            return exp(ith_obd_log)

        return _ith_one_body_density

    @cached_property
    def one_body_density(self):
        """

        :return:
        """
        ith_one_body_density = self.ith_one_body_density

        @jit(nopython=True, nogil=True)
        def _one_body_density(sz: float,
                              sys_conf: np.ndarray,
                              cfc_spec: CFCSpecNT):
            """Computes the logarithm of the local one-body density matrix
            for a given configuration of the position of the bodies and for a
            specified particle index.

            :param sz:
            :param sys_conf:
            :param cfc_spec:
            :return:
            """
            obd = 0.
            nop = cfc_spec.model_params.boson_number
            for i_ in range(nop):
                obd += ith_one_body_density(i_, sz, sys_conf, cfc_spec)
            return obd / nop

        return _one_body_density

    @cached_property
    def fourier_density(self):
        """

        :return:
        """
        pos_slot = int(SysConfSlot.pos)

        # noinspection PyUnusedLocal
        @jit(nopython=True)
        def _fourier_density(kz: float,
                             sys_conf: np.ndarray,
                             cfc_spec: CFCSpecNT):
            """Fourier density component with momentum :math:`k`.

            This function corresponds to the Fourier transform of the
            density operator.

            :param kz:
            :param sys_conf:
            :param cfc_spec:
            :return:
            """
            model_params = cfc_spec.model_params
            s_sin, s_cos = 0., 0.

            nop = cfc_spec.model_params.boson_number
            for i_ in range(nop):
                z_i = sys_conf[pos_slot, i_]
                s_cos += cos(kz * z_i)
                s_sin += sin(kz * z_i)

            return s_cos + 1j * s_sin

        return _fourier_density


class PhysicalFuncs(model.PhysicalFuncs):
    """Functions to calculate the main physical properties of a model."""

    #: The model spec these functions correspond to.
    cfc_spec_nt: CFCSpecNT

    @property
    @abstractmethod
    def core_funcs(self) -> CoreFuncs:
        """The core functions of the model."""
        pass

    @cached_property
    def wf_abs_log(self):
        """Logarithm of the absolute value fo the trial wave function."""

        # These functions are compiled for the instance model spec. Therefore,
        # the spec.cfc_spec_nt attribute becomes a compile-time constant and
        # the functions depend only on the configuration of the particles of
        # the system, and possibly on other quantities with physical
        # significance but otherwise independent to the model.
        cfc_spec_nt = self.cfc_spec_nt
        __wf_abs_log = self.core_funcs.wf_abs_log
        signatures = ['(f8[:,:],f8[:])']
        layout = '(ns,nop)->()'

        # noinspection PyTypeChecker
        @guvectorize(signatures, layout, nopython=True, target='parallel')
        def _wf_abs_log(sys_conf: np.ndarray,
                        result: np.ndarray) -> np.ndarray:
            """

            :param sys_conf:
            :param result:
            :return:
            """
            result[0] = __wf_abs_log(sys_conf, cfc_spec_nt)

        return _wf_abs_log

    @cached_property
    def energy(self):
        """Local energy."""

        cfc_spec_nt = self.cfc_spec_nt
        __energy = self.core_funcs.energy
        types = ['(f8[:,:],f8[:])']
        signature = '(ns,nop) -> ()'

        # noinspection PyTypeChecker
        @guvectorize(types, signature, nopython=True, target='parallel')
        def _energy(sys_conf: np.ndarray, result: np.ndarray) -> np.ndarray:
            """

            :param sys_conf:
            :param result:
            :return:
            """
            result[0] = __energy(sys_conf, cfc_spec_nt)

        return _energy

    @property
    def one_body_density(self):
        """One-body density matrix."""

        cfc_spec_nt = self.cfc_spec_nt
        __one_body_density = self.core_funcs.one_body_density
        types = ['(f8,f8[:,:],f8[:])']
        signature = '(),(ns,nop) -> ()'

        # noinspection PyTypeChecker
        @guvectorize(types, signature, nopython=True, target='parallel')
        def _one_body_density(sz: float, sys_conf: np.ndarray,
                              result: np.ndarray) -> np.ndarray:
            """

            :param sz:
            :param sys_conf:
            :param result:
            :return:
            """
            result[0] = __one_body_density(sz, sys_conf, cfc_spec_nt)

        return _one_body_density

    @cached_property
    def fourier_density(self):
        """Fourier density component with momentum :math:`k`.

        This function corresponds to the Fourier transform of the
        density operator.
        """

        cfc_spec_nt = self.cfc_spec_nt
        __fourier_density = self.core_funcs.fourier_density
        types = ['(f8,f8[:,:],c16[:])']
        signature = '(nkz),(ns,nop) -> (nkz)'

        # noinspection PyTypeChecker
        @guvectorize(types, signature, nopython=True, target='parallel')
        def _fourier_density(kz_set: np.ndarray,
                             sys_conf: np.ndarray,
                             result: np.ndarray) -> np.ndarray:
            """

            :param kz_set:
            :param sys_conf:
            :param result:
            :return:
            """
            for i_ in range(kz_set.shape[0]):
                kz = kz_set[i_]
                result[i_] = __fourier_density(kz, sys_conf, cfc_spec_nt)

        return _fourier_density


class CSWFOptimizer(model.WFOptimizer):
    """Class to optimize the trial-wave function.

    It uses the correlated sampling method to minimize the variance of
    the local energy.
    """

    __slots__ = ()

    #: The spec of the model.
    spec: Spec

    #: The system configurations used for the minimization process.
    sys_conf_set: np.ndarray

    #: The initial wave function values. Used to calculate the weights.
    ini_wf_abs_log_set: np.ndarray

    #: The energy of reference to minimize the variance of the local energy.
    ref_energy: t.Optional[float]

    # noinspection PyUnusedLocal
    @staticmethod
    def weighed_variance(weights_log_set: np.ndarray,
                         energy_set: np.ndarray,
                         ref_energy: float = None):
        """Basic calculation of the weighed variance.

        :param energy_set: The set of energies.
        :param weights_log_set: The set of weights.
        :param ref_energy: The energy of reference.
        :return: The variance.
        """
        rel_weights = np.exp(weights_log_set - weights_log_set.max())
        weight_sum = rel_weights.sum()

        ref_energy = (rel_weights * energy_set).sum() / weight_sum
        e_diff = rel_weights * (energy_set - ref_energy) ** 2

        return e_diff.sum() / weight_sum

    @abstractmethod
    def update_spec(self, tbf_contact_cutoff: float):
        """Updates the initial model spec with a new variational parameter.

        :param tbf_contact_cutoff: The variational parameter.
        :return: The updated ``Spec`` instance.
        """
        pass

    @abstractmethod
    def wf_abs_log_and_energy_set(self, cfc_spec: CFCSpecNT) -> \
            t.Tuple[np.ndarray, np.ndarray]:
        """Evaluates the wave function and energy for all the configurations.

        :param cfc_spec: The spec of the core functions.
        :return: Two arrays: one with the values of the wave function, and
            the second with the values of the energy.
        """
        pass

    def principal_function(self, tbf_contact_cutoff: float):
        """Evaluates the variance.

        :param tbf_contact_cutoff:
        :return: The weighed variance.
        """

        # Temporary spec with the variational parameter modified
        upd_model_spec = self.update_spec(tbf_contact_cutoff)
        cfc_spec = upd_model_spec.cfc_spec_nt

        # The function to get the energy and weights.
        core_funcs_data = self.wf_abs_log_and_energy_set(cfc_spec)
        wf_abs_log_set, energies_set = core_funcs_data

        # Move to a internal gufunc
        ini_wf_abs_log_set = self.ini_wf_abs_log_set
        weights_log_set = 2 * (wf_abs_log_set - ini_wf_abs_log_set)

        # Calculate the weighed variance.
        return self.weighed_variance(weights_log_set, energies_set)

    @abstractmethod
    def exec(self):
        """Starts the minimization process."""
        pass
