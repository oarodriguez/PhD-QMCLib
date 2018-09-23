from abc import ABCMeta, abstractmethod
from enum import Enum, IntEnum, unique
from math import (cos, exp, fabs, log, sin)
from typing import Mapping as TMapping

import numpy as np
import numpy.random as random
from numba import jit

from thesis_lib.utils import cached_property
from ..abc import QMCFuncsBase, QMCModelBase
from ..utils import min_distance, sign

__all__ = [
    'QMCFuncs',
    'ModelBase',
    'ParamsSlots',
    'StrictModelBase',
    'BosonConfSlots',
    'SysConfDistType'
]


@unique
class BosonConfSlots(IntEnum):
    """"""
    POS_SLOT = 0
    DRIFT_SLOT = 1


class SysConfDistType(Enum):
    """"""
    RANDOM = 'random'
    REGULAR = 'regular'


@unique
class ParamsSlots(IntEnum):
    """"""

    BOSON_NUMBER = 0  # Value does not matter here
    SUPERCELL_SIZE = 1  # Value does not matter here


class ModelFN(str, Enum):
    """"""

    OBF = 'one_body_func'
    TBF = 'two_body_func'

    OBF_LD = 'one_body_func_log_dz'
    TBF_LD = 'two_body_func_log_dz'

    OBF_LD2 = 'one_body_func_log_dz2'
    TBF_LD2 = 'two_body_func_log_dz2'


DIST_RAND = SysConfDistType.RANDOM
DIST_REGULAR = SysConfDistType.REGULAR

# The dimensions of a system configuration.
BOSON_CONF_SLOT_DIM = 0
BOSON_INDEX_DIM = 1


class ModelBase(QMCModelBase):
    """Abstract Base Class that represents a Quantum Monte Carlo model
    with a trial-wave function of the Bijl-Jastrow type.
    """

    FN = ModelFN

    SysConfDistType = SysConfDistType

    BosonConfSlots = BosonConfSlots

    ParamsSlots = ParamsSlots

    def __init__(self, params: TMapping[str, float],
                 var_params: TMapping[str, float] = None):
        """

        :param params:
        :param var_params:
        """
        super().__init__()

        self._params = self._get_check_params(params)
        self._var_params = dict(var_params or {})

    @property
    def boson_number(self):
        """"""
        params = self.params
        if not params:
            raise ValueError
        return params[self.ParamsSlots.BOSON_NUMBER]

    @property
    def supercell_size(self):
        """"""
        params = self.params
        if not params:
            raise ValueError
        return params[self.ParamsSlots.SUPERCELL_SIZE]

    @property
    def boundaries(self):
        """"""
        sc_size = self.supercell_size
        return 0., 1. * sc_size

    @property
    @abstractmethod
    def is_free(self):
        """"""
        pass

    @property
    @abstractmethod
    def is_ideal(self):
        """"""
        pass

    @property
    def params(self):
        """Returns a tuple with the model parameters in the same order
         as :attr:`ModelBase.ParamsSlots` attribute.

        :return:
        """
        params_slots = self.ParamsSlots
        self_params = self._params

        # We care about order here, so we iterate over __members__
        slots_items = params_slots.__members__.items()
        ord_params = [self_params[name.lower()] for name, _ in slots_items]
        return tuple(ord_params)

    def update_params(self, params: TMapping):
        """"""
        checked_params = self._get_check_params(params, check_all=False)
        self._params.update(checked_params)

    def _get_check_params(self, params: TMapping,
                          check_all: bool = True):
        """

        :param params:
        :param check_all:
        :return:
        """
        # We want a shallow copy of the input parameters.
        self_params = {}
        params_slots = self.ParamsSlots
        params_names = params.keys()
        slots_names = set([slot.name.lower() for slot in params_slots])
        distinct = params_names ^ slots_names
        if not distinct:
            self_params.update(params)
            return self_params
        else:
            missing = distinct - params_names
            extra = distinct - slots_names
            if missing:
                if check_all:
                    msg = "{} required parameter not specified"
                    raise KeyError(msg.format(missing))
                else:
                    present = params_names - distinct
                    for name in present:
                        self_params[name] = params[name]
                    return self_params
            if extra:
                msg = "'unrecognized {} parameter"
                raise KeyError(msg.format(extra))

    @property
    def var_params(self):
        """"""
        return self._var_params

    def update_var_params(self, params: TMapping = None):
        """"""
        self._var_params.update(params)

    @property
    def num_boson_conf_slots(self):
        return len([_ for _ in self.BosonConfSlots])

    @property
    def sys_conf_shape(self):
        """The shape of the array/buffer that stores the configuration
        of the particles (positions, velocities, etc.)
        """
        # NOTE: Should we allocate space for the DRIFT_SLOT?
        # TODO: Fix NUM_SLOTS if DRIFT_SLOT becomes really unnecessary.
        nop = self.boson_number
        ns = self.num_boson_conf_slots
        return ns, nop

    def get_sys_conf_buffer(self):
        """Creates an empty array/buffer to store the configuration
        of the particles of the system.

        :return: A ``np.ndarray`` with zero-filled entries.
        """
        sc_shape = self.sys_conf_shape
        return np.zeros(sc_shape, dtype=np.float64)

    def init_get_sys_conf(self, dist_type=DIST_RAND,
                          offset=None):
        """Creates and initializes a system configuration with the
        positions of the particles arranged in the order specified
        by ``dist_type`` argument.

        :param dist_type:
        :param offset:
        :return:
        """
        nop = self.boson_number
        z_min, z_max = self.boundaries
        pos_slot = self.BosonConfSlots.POS_SLOT
        sys_conf = self.get_sys_conf_buffer()
        sc_size = z_max - z_min
        offset = offset or 0.

        if dist_type is DIST_RAND:
            spread = sc_size * random.random_sample(nop)
        elif dist_type is DIST_REGULAR:
            spread = np.linspace(0, sc_size, nop, endpoint=False)
        else:
            raise ValueError("unrecognized '{}' dist_type".format(dist_type))

        sys_conf[pos_slot, :] = z_min + (offset + spread) % sc_size
        return sys_conf

    @property
    @abstractmethod
    def obf_params(self):
        pass

    @property
    @abstractmethod
    def tbf_params(self):
        pass

    @property
    @abstractmethod
    def energy_params(self):
        pass


class StrictModelBase(ModelBase, metaclass=ABCMeta):
    """"""

    def __init__(self, params: TMapping[str, float],
                 var_params: TMapping[str, float]):
        """

        :param params:
        :param var_params:
        """
        if var_params is None:
            raise ValueError("var_params must be a mapping-like object")

        super().__init__(params, var_params)

    @property
    def params(self):
        """"""
        return super().params

    def update_params(self, value):
        """"""
        raise AttributeError("cant't change params")

    @property
    def var_params(self):
        return super().var_params

    def update_var_params(self, params=None):
        """"""
        raise AttributeError("can't change var_params")


class QMCFuncs(QMCFuncsBase):
    """Abstract Base Class that groups core, JIT-compiled, performance-critical
    functions to realize a Quantum Monte Carlo calculation for a QMC model
    with a trial-wave function of the Bijl-Jastrow type.
    """

    ParamsSlots = ParamsSlots

    BosonConfSlots = BosonConfSlots

    @cached_property
    def boson_number(self):
        """"""
        boson_number_slot = int(self.ParamsSlots.BOSON_NUMBER)

        @jit(nopython=True, cache=True)
        def _boson_number(model_params):
            """"""
            return model_params[boson_number_slot]

        return _boson_number

    @cached_property
    def supercell_size(self):
        """"""
        sc_size_slot = int(self.ParamsSlots.SUPERCELL_SIZE)

        @jit(nopython=True, cache=True)
        def _supercell_size(model_params):
            """"""
            return model_params[sc_size_slot]

        return _supercell_size

    @cached_property
    def boundaries(self):
        """"""
        supercell_size = self.supercell_size

        @jit(nopython=True, cache=True)
        def _boundaries(model_params):
            """"""
            sc_size = supercell_size(model_params)
            return 0., 1. * sc_size

        return _boundaries

    @cached_property
    @abstractmethod
    def is_free(self):
        """"""

        # noinspection PyUnusedLocal
        @jit(nopython=True, cache=True)
        def _is_free(model_params):
            """"""
            return True

        return _is_free

    @cached_property
    @abstractmethod
    def is_ideal(self):
        """"""

        # noinspection PyUnusedLocal
        @jit(nopython=True, cache=True)
        def _is_ideal(model_params):
            """"""
            return False

        return _is_ideal

    @property
    @abstractmethod
    def one_body_func(self):
        pass

    @property
    @abstractmethod
    def two_body_func(self):
        pass

    @property
    @abstractmethod
    def one_body_func_log_dz(self):
        pass

    @property
    @abstractmethod
    def two_body_func_log_dz(self):
        pass

    @property
    @abstractmethod
    def one_body_func_log_dz2(self):
        pass

    @property
    @abstractmethod
    def two_body_func_log_dz2(self):
        pass

    @property
    @abstractmethod
    def potential(self):
        pass

    @cached_property
    def ith_wf_abs_log(self):
        """

        :return:
        """
        is_free = self.is_free
        is_ideal = self.is_ideal
        supercell_size = self.supercell_size
        pos_slot = int(self.SysConfSlots.POS_SLOT)

        one_body_func = self.one_body_func
        two_body_func = self.two_body_func

        # TODO: Add an underscore as a prefix.
        @jit(nopython=True, cache=True)
        def _ith_wf_abs_log(i_, sys_conf,
                            model_params,
                            obf_params,
                            tbf_params):
            """Computes the variational wave function of a system of bosons in
            a specific configuration.

            :param i_:
            :param sys_conf:
            :param model_params:
            :param obf_params:
            :param tbf_params:
            :return:
            """
            ith_wf_abs_log = 0.
            nop = sys_conf.shape[BOSON_INDEX_DIM]
            sc_size = supercell_size(model_params)

            if not is_free(model_params):
                # Gas subject to external potential.
                z_i = sys_conf[pos_slot, i_]
                obv = one_body_func(z_i, *obf_params)
                ith_wf_abs_log += log(fabs(obv))

            if not is_ideal(model_params):
                # Gas with interactions.
                z_i = sys_conf[pos_slot, i_]
                for j_ in range(i_ + 1, nop):
                    z_j = sys_conf[pos_slot, j_]
                    z_ij = min_distance(z_i, z_j, sc_size)
                    tbv = two_body_func(fabs(z_ij), *tbf_params)

                    ith_wf_abs_log += log(fabs(tbv))

            return ith_wf_abs_log

        return _ith_wf_abs_log

    @cached_property
    def wf_abs_log(self):
        """

        :return:
        """
        is_free = self.is_free
        is_ideal = self.is_ideal
        ith_wf_abs_log = self.ith_wf_abs_log

        @jit(nopython=True, cache=True)
        def _wf_abs_log(sys_conf,
                        model_params,
                        obf_params,
                        tbf_params):
            """Computes the variational wave function of a system of bosons in
            a specific configuration.

            :param sys_conf:
            :param model_params:
            :param obf_params:
            :param tbf_params:
            :return:
            """
            wf_abs_log = 0.
            nop = sys_conf.shape[BOSON_INDEX_DIM]

            if is_free(model_params) and is_ideal(model_params):
                return wf_abs_log

            for i_ in range(nop):
                wf_abs_log += ith_wf_abs_log(i_, sys_conf, model_params,
                                             obf_params, tbf_params)
            return wf_abs_log

        return _wf_abs_log

    @cached_property
    def wf_abs(self):
        """

        :return:
        """
        wf_abs_log = self.wf_abs_log

        @jit(nopython=True, cache=True)
        def _wf_abs(sys_conf,
                    model_params,
                    obf_params,
                    tbf_params):
            """Computes the variational wave function of a system of
            bosons in a specific configuration.

            :param sys_conf:
            :param model_params:
            :param obf_params:
            :param tbf_params:
            :return:
            """
            wf_abs_log_ = wf_abs_log(sys_conf, model_params, obf_params,
                                     tbf_params)
            return exp(wf_abs_log_)

        return _wf_abs

    @cached_property
    def delta_wf_abs_log_kth_move(self):
        """

        :return:
        """
        is_free = self.is_free
        is_ideal = self.is_ideal
        supercell_size = self.supercell_size
        pos_slot = int(self.SysConfSlots.POS_SLOT)

        one_body_func = self.one_body_func
        two_body_func = self.two_body_func

        @jit(nopython=True, cache=True)
        def _delta_wf_abs_log_kth_move(k_, sys_conf,
                                       func_params,
                                       model_params,
                                       obf_params,
                                       tbf_params):
            """Computes the change of the logarithm of the wave function
            after displacing the `k-th` particle by a distance ``z_k_delta``.

            :param k_:
            :param sys_conf:
            :param func_params:
            :param model_params:
            :param obf_params:
            :param tbf_params:
            :return:
            """
            delta_wf_abs_log = 0.
            nop = sys_conf.shape[BOSON_INDEX_DIM]
            sc_size = supercell_size(model_params)

            if is_free(model_params) and is_ideal(model_params):
                return delta_wf_abs_log

            # Unpack the parameters of the function.
            z_k_delta, = func_params
            z_k = sys_conf[pos_slot, k_]
            z_k_upd = z_k + z_k_delta

            if not is_free(model_params):
                # Gas subject to external potential.
                obv = one_body_func(z_k, *obf_params)
                obv_upd = one_body_func(z_k_upd, *obf_params)
                delta_wf_abs_log += log(fabs(obv_upd / obv))

            if not is_ideal(model_params):
                # Gas with interactions.
                for i_ in range(nop):
                    if i_ == k_:
                        continue

                    z_i = sys_conf[pos_slot, i_]
                    r_ki = fabs(min_distance(z_k, z_i, sc_size))
                    r_ki_upd = fabs(min_distance(z_k_upd, z_i, sc_size))

                    tbv = two_body_func(r_ki, *tbf_params)
                    tbv_upd = two_body_func(r_ki_upd, *tbf_params)
                    delta_wf_abs_log += log(fabs(tbv_upd / tbv))

            return delta_wf_abs_log

        return _delta_wf_abs_log_kth_move

    @cached_property
    def ith_drift(self):
        """

        :return:
        """
        is_free = self.is_free
        is_ideal = self.is_ideal
        supercell_size = self.supercell_size
        pos_slot = int(self.SysConfSlots.POS_SLOT)

        one_body_func_log_dz = self.one_body_func_log_dz
        two_body_func_log_dz = self.two_body_func_log_dz

        @jit(nopython=True, cache=True)
        def _ith_drift(i_, sys_conf,
                       model_params,
                       obf_params,
                       tbf_params):
            """Computes the local energy for a given configuration of the
            position of the bodies. The kinetic energy of the hamiltonian is
            computed through central finite differences.

            :param i_:
            :param sys_conf: The current configuration of the positions of the
                   particles.
            :param model_params:
            :param obf_params:
            :param tbf_params:
            :return: The local energy.
            """

            if is_free(model_params) and is_ideal(model_params):
                return 0.

            # Unpack the parameters.
            ith_drift = 0.
            nop = sys_conf.shape[BOSON_INDEX_DIM]
            sc_size = supercell_size(model_params)

            if not is_free(model_params):
                # Case with external potential.
                z_i = sys_conf[pos_slot, i_]
                ob_fn_ldz = one_body_func_log_dz(z_i, *obf_params)

                ith_drift += ob_fn_ldz

            if not is_ideal(model_params):
                # Case with interactions.
                z_i = sys_conf[pos_slot, i_]

                for j_ in range(nop):
                    # Do not account diagonal terms.
                    if j_ == i_:
                        continue

                    z_j = sys_conf[pos_slot, j_]
                    z_ij = min_distance(z_i, z_j, sc_size)
                    sgn = sign(z_ij)

                    tb_fn_ldz = two_body_func_log_dz(fabs(z_ij),
                                                     *tbf_params) * sgn

                    ith_drift += tb_fn_ldz

            # Accumulate to the drift velocity squared magnitude.
            return ith_drift

        return _ith_drift

    @cached_property
    def ith_drift_to_buffer(self):
        """

        :return:
        """
        pos_slot = int(self.SysConfSlots.POS_SLOT)
        drift_slot = int(self.SysConfSlots.DRIFT_SLOT)
        ith_drift = self.ith_drift

        @jit(nopython=True, cache=True)
        def _ith_drift_to_buffer(i_, sys_conf,
                                 model_params,
                                 obf_params,
                                 tbf_params,
                                 result):
            """

            :param i_:
            :param sys_conf:
            :param model_params:
            :param obf_params:
            :param tbf_params:
            :param result:
            :return:
            """
            result[pos_slot, i_] = sys_conf[pos_slot, i_]
            result[drift_slot, i_] = ith_drift(i_, sys_conf, model_params,
                                               obf_params, tbf_params)

        return _ith_drift_to_buffer

    @cached_property
    def delta_ith_drift_kth_move(self):
        """

        :return:
        """
        is_free = self.is_free
        is_ideal = self.is_ideal
        supercell_size = self.supercell_size
        pos_slot = int(self.SysConfSlots.POS_SLOT)

        one_body_func_log_dz = self.one_body_func_log_dz
        two_body_func_log_dz = self.two_body_func_log_dz

        @jit(nopython=True, cache=True)
        def _delta_ith_drift_kth_move(i_, k_,
                                      sys_conf,
                                      func_params,
                                      model_params,
                                      obf_params,
                                      tbf_params):
            """Computes the change of the i-th component of the drift
            after displacing the k-th particle by a distance ``z_k_delta``.

            :param i_:
            :param k_:
            :param sys_conf:
            :param func_params:
            :param model_params:
            :param obf_params:
            :param tbf_params:
            :return:
            """
            delta_ith_drift = 0.
            nop = sys_conf.shape[BOSON_INDEX_DIM]
            sc_size = supercell_size(model_params)

            if is_free(model_params) and is_ideal(model_params):
                return delta_ith_drift

            z_k_delta, = func_params
            z_k = sys_conf[pos_slot, k_]
            z_k_upd = z_k + z_k_delta

            if i_ != k_:
                #
                if is_ideal(model_params):
                    return delta_ith_drift

                z_i = sys_conf[pos_slot, i_]
                z_ki_upd = min_distance(z_k_upd, z_i, sc_size)
                z_ki = min_distance(z_k, z_i, sc_size)

                # TODO: Move th sign to the function.
                sgn = sign(z_ki)
                ob_fn_ldz = two_body_func_log_dz(fabs(z_ki),
                                                 *tbf_params) * sgn

                sgn = sign(z_ki_upd)
                ob_fn_ldz_upd = two_body_func_log_dz(fabs(z_ki_upd),
                                                     *tbf_params) * sgn

                delta_ith_drift += -(ob_fn_ldz_upd - ob_fn_ldz)
                return delta_ith_drift

            if not is_free(model_params):
                #
                ob_fn_ldz = one_body_func_log_dz(z_k, *obf_params)
                ob_fn_ldz_upd = one_body_func_log_dz(z_k_upd, *obf_params)

                delta_ith_drift += ob_fn_ldz_upd - ob_fn_ldz

            if not is_ideal(model_params):
                # Gas with interactions.
                for j_ in range(nop):
                    #
                    if j_ == k_:
                        continue

                    z_j = sys_conf[pos_slot, j_]
                    z_kj = min_distance(z_k, z_j, sc_size)
                    z_kj_upd = min_distance(z_k_upd, z_j, sc_size)

                    sgn = sign(z_kj)
                    tb_fn_ldz = two_body_func_log_dz(fabs(z_kj),
                                                     *tbf_params) * sgn

                    sgn = sign(z_kj_upd)
                    tb_fn_ldz_upd = two_body_func_log_dz(fabs(z_kj_upd),
                                                         *tbf_params) * sgn

                    delta_ith_drift += (tb_fn_ldz_upd - tb_fn_ldz)

            return delta_ith_drift

        return _delta_ith_drift_kth_move

    @cached_property
    def ith_energy(self):
        """

        :return:
        """
        is_free = self.is_free
        is_ideal = self.is_ideal
        supercell_size = self.supercell_size
        pos_slot = int(self.SysConfSlots.POS_SLOT)
        drift_slot = int(self.SysConfSlots.DRIFT_SLOT)

        potential = self.potential
        one_body_func_log_dz = self.one_body_func_log_dz
        two_body_func_log_dz = self.two_body_func_log_dz
        one_body_func_log_dz2 = self.one_body_func_log_dz2
        two_body_func_log_dz2 = self.two_body_func_log_dz2

        # noinspection PyUnusedLocal
        @jit(nopython=True, cache=True)
        def _ith_energy(i_, sys_conf,
                        func_params,
                        model_params,
                        obf_params,
                        tbf_params):
            """Computes the local energy for a given configuration of the
            position of the bodies. The kinetic energy of the hamiltonian is
            computed through central finite differences.

            :param i_:
            :param sys_conf: The current configuration of the positions of the
                   particles.
            :param func_params:
            :param model_params:
            :param obf_params:
            :param tbf_params:
            :return: The local energy.
            """

            if is_free(model_params) and is_ideal(model_params):
                return 0.

            nop = sys_conf.shape[BOSON_INDEX_DIM]
            sc_size = supercell_size(model_params)

            # Unpack the parameters.
            kin_energy = 0.
            pot_energy = 0
            ith_drift = 0.

            if not is_free(model_params):
                # Case with external potential.
                z_i = sys_conf[pos_slot, i_]
                ob_fn_ldz2 = one_body_func_log_dz2(z_i, *obf_params)
                ob_fn_ldz = one_body_func_log_dz(z_i, *obf_params)

                kin_energy += (-ob_fn_ldz2 + ob_fn_ldz ** 2)
                pot_energy += potential(z_i, *func_params)
                ith_drift += ob_fn_ldz

            if not is_ideal(model_params):
                # Case with interactions.
                z_i = sys_conf[pos_slot, i_]

                for j_ in range(nop):
                    # Do not account diagonal terms.
                    if j_ == i_:
                        continue

                    z_j = sys_conf[pos_slot, j_]
                    z_ij = min_distance(z_i, z_j, sc_size)
                    sgn = sign(z_ij)

                    tb_fn_ldz2 = two_body_func_log_dz2(fabs(z_ij),
                                                       *tbf_params)
                    tb_fn_ldz = two_body_func_log_dz(fabs(z_ij),
                                                     *tbf_params) * sgn

                    kin_energy += (-tb_fn_ldz2 + tb_fn_ldz ** 2)
                    ith_drift += tb_fn_ldz

            # Accumulate to the drift velocity squared magnitude.
            ith_drift_mag = ith_drift ** 2
            kin_energy -= ith_drift_mag

            # Save the drift and avoid extra work.
            # TODO: We need a better way to store and handle this data.
            sys_conf[drift_slot, i_] = ith_drift

            return kin_energy + pot_energy

        return _ith_energy

    @cached_property
    def energy(self):
        """

        :return:
        """
        ith_energy = self.ith_energy

        @jit(nopython=True, cache=True)
        def _energy(sys_conf,
                    func_params,
                    model_params,
                    obf_params,
                    tbf_params):
            """

            :param sys_conf:
            :param func_params:
            :param model_params:
            :param obf_params:
            :param tbf_params:
            :return:
            """
            energy = 0.
            nop = sys_conf.shape[BOSON_INDEX_DIM]

            for i_ in range(nop):
                energy += ith_energy(i_, sys_conf, func_params, model_params,
                                     obf_params, tbf_params)
            return energy

        return _energy

    @cached_property
    def energy_to_buffer(self):
        """

        :return:
        """
        energy = self.energy

        @jit(nopython=True, cache=True)
        def _energy_to_buffer(sys_conf,
                              func_params,
                              model_params,
                              obf_params,
                              tbf_params,
                              result):
            """Computes the local energy for a given configuration of the
            position of the bodies.

            :param sys_conf:
            :param func_params:
            :param model_params:
            :param obf_params:
            :param tbf_params:
            :param result:
            """
            result[0] = energy(sys_conf, func_params, model_params, obf_params,
                               tbf_params)

        return _energy_to_buffer

    @cached_property
    def ith_one_body_density(self):
        """

        :return:
        """
        is_free = self.is_free
        is_ideal = self.is_ideal
        supercell_size = self.supercell_size
        pos_slot = int(self.SysConfSlots.POS_SLOT)

        one_body_func = self.one_body_func
        two_body_func = self.two_body_func

        @jit(nopython=True, cache=True)
        def _ith_one_body_density(i_, sys_conf,
                                  func_params,
                                  model_params,
                                  obf_params,
                                  tbf_params):
            """Computes the logarithm of the local one-body density matrix
            for a given configuration of the position of the bodies and for a
            specified particle index.

            :param i_:
            :param sys_conf:
            :param func_params:
            :param model_params:
            :param obf_params:
            :param tbf_params:
            :return:
            """
            if is_free(model_params) and is_ideal(model_params):
                return 1.

            # The local one-body density matrix is calculated as the
            # quotient of the wave function with the ``i_`` particle
            # shifted a distance ``sz` from its original position
            # divided by the wave function with the particles evaluated
            # in their original positions. To improve statistics, we
            # average over all possible particle displacements.
            ith_obd = 0.
            nop = sys_conf.shape[BOSON_INDEX_DIM]
            sc_size = supercell_size(model_params)
            sz, = func_params

            if not is_free(model_params):

                z_i = sys_conf[pos_slot, i_]
                z_i_sft = z_i + sz

                ob_fn = one_body_func(z_i, *obf_params)
                ob_fn_sft = one_body_func(z_i_sft, *obf_params)
                ith_obd += (log(ob_fn_sft) - log(ob_fn))

            if not is_ideal(model_params):
                # Interacting gas.
                z_i = sys_conf[pos_slot, i_]
                z_i_sft = z_i + sz

                for j_ in range(nop):
                    #
                    if i_ == j_:
                        continue

                    z_j = sys_conf[pos_slot, j_]
                    z_ij = min_distance(z_i, z_j, sc_size)
                    tb_fn = two_body_func(fabs(z_ij), *tbf_params)

                    # Shifted difference.
                    z_ij = min_distance(z_i_sft, z_j, sc_size)
                    tb_fn_shift = two_body_func(fabs(z_ij), *tbf_params)

                    ith_obd += (log(tb_fn_shift) - log(tb_fn))

            return exp(ith_obd)

        return _ith_one_body_density

    @cached_property
    def one_body_density(self):
        """

        :return:
        """
        ith_one_body_density = self.ith_one_body_density

        @jit(nopython=True, cache=True)
        def _one_body_density(sys_conf,
                              func_params,
                              model_params,
                              obf_params,
                              tbf_params):
            """Computes the logarithm of the local one-body density matrix
            for a given configuration of the position of the bodies and for a
            specified particle index.

            :param sys_conf:
            :param func_params:
            :param model_params:
            :param obf_params:
            :param tbf_params:
            :return:
            """
            obd = 0.
            nop = sys_conf.shape[BOSON_INDEX_DIM]

            for i_ in range(nop):
                obd += ith_one_body_density(i_, sys_conf, func_params,
                                            model_params, obf_params,
                                            tbf_params)
            return obd / nop

        return _one_body_density

    @cached_property
    def one_body_density_to_buffer(self):
        """

        :return:
        """
        one_body_density = self.one_body_density

        @jit(nopython=True, cache=True)
        def _one_body_density_to_buffer(sys_conf,
                                        func_params,
                                        model_params,
                                        obf_params,
                                        tbf_params,
                                        result):
            """Computes the logarithm of the local one-body density matrix
            for a given configuration of the position of the bodies and for a
            specified particle index.

            :param sys_conf:
            :param func_params:
            :param model_params:
            :param obf_params:
            :param tbf_params:
            :param result:
            :return:
            """
            result[0] = one_body_density(sys_conf, func_params,
                                         model_params, obf_params,
                                         tbf_params)

        return _one_body_density_to_buffer

    @cached_property
    def structure_factor(self):
        """

        :return:
        """
        pos_slot = int(self.SysConfSlots.POS_SLOT)

        # noinspection PyUnusedLocal
        @jit(nopython=True, cache=True)
        def _structure_factor(sys_conf,
                              func_params,
                              model_params,
                              obf_params,
                              tbf_params):
            """Computes the local two-body correlation function for a given
            configuration of the position of the bodies.

            :param sys_conf:
            :param func_params:
            :param model_params:
            :param obf_params:
            :param tbf_params:
            :return:
            """
            nop = sys_conf.shape[BOSON_INDEX_DIM]
            kz, = func_params
            s_sin, s_cos = 0., 0.

            for i_ in range(nop):
                z_i = sys_conf[pos_slot, i_]
                s_cos += cos(kz * z_i)
                s_sin += sin(kz * z_i)

            return s_cos ** 2 + s_sin ** 2

        return _structure_factor
