from abc import ABCMeta
from enum import IntEnum, unique

from numba import jit

from thesis_lib.qmc_lib import jastrow
from thesis_lib.utils import cached_property

__all__ = [
    'Model',
    'QMCFuncs',
    'potential_func'
]


@unique
class ParamsSlots(IntEnum):
    """"""
    LATTICE_DEPTH = 0
    LATTICE_RATIO = 1
    INTERACTION_STRENGTH = 2
    BOSON_NUMBER = 3
    SUPERCELL_SIZE = 4


class Model(jastrow.Model, metaclass=ABCMeta):
    """"""

    ParamsSlots = ParamsSlots

    @property
    def lattice_depth(self):
        """"""
        params = self.params
        if not params:
            raise ValueError
        return params[self.ParamsSlots.LATTICE_DEPTH]

    @property
    def lattice_ratio(self):
        """"""
        params = self.params
        if not params:
            raise ValueError
        return params[self.ParamsSlots.LATTICE_RATIO]

    @property
    def interaction_strength(self):
        """"""
        params = self.params
        if not params:
            raise ValueError
        return params[self.ParamsSlots.INTERACTION_STRENGTH]

    @property
    def well_width(self):
        """"""
        r = self.lattice_ratio
        return 1 / (1 + r)

    @property
    def barrier_width(self):
        """"""
        r = self.lattice_ratio
        return r / (1 + r)

    @property
    def is_free(self):
        """"""
        v0 = self.lattice_depth
        r = self.lattice_ratio
        if v0 <= 1e-10:
            return True
        elif r <= 1e-10:
            return True
        else:
            return False

    @property
    def is_ideal(self):
        """"""
        gn = self.interaction_strength
        if gn <= 1e-10:
            return True
        else:
            return False


class QMCFuncs(jastrow.QMCFuncs, metaclass=ABCMeta):
    """"""

    ParamsSlots = ParamsSlots

    @cached_property
    def lattice_depth(self):
        """"""

        lattice_depth_slot = int(self.ParamsSlots.LATTICE_DEPTH)

        @jit(nopython=True, cache=True)
        def _lattice_depth(model_params):
            """"""
            return model_params[lattice_depth_slot]

        return _lattice_depth

    @cached_property
    def lattice_ratio(self):
        """"""
        lattice_ratio_slot = int(self.ParamsSlots.LATTICE_RATIO)

        @jit(nopython=True, cache=True)
        def _lattice_ratio(model_params: tuple) -> float:
            """"""
            return model_params[lattice_ratio_slot]

        return _lattice_ratio

    @cached_property
    def interaction_strength(self):
        """"""

        int_strength_slot = int(self.ParamsSlots.INTERACTION_STRENGTH)

        @jit(nopython=True, cache=True)
        def _interaction_strength(model_params):
            """"""
            return model_params[int_strength_slot]

        return _interaction_strength

    @cached_property
    def well_width(self):
        """"""
        lattice_ratio = self.lattice_ratio

        @jit(nopython=True, cache=True)
        def _well_width(model_params):
            """"""
            r = lattice_ratio(model_params)
            return 1 / (1 + r)

        return _well_width

    @cached_property
    def barrier_width(self):
        """"""
        lattice_ratio = self.lattice_ratio

        @jit(nopython=True, cache=True)
        def _barrier_width(model_params):
            """"""
            r = lattice_ratio(model_params)
            return r / (1 + r)

        return _barrier_width

    @cached_property
    def is_free(self):
        """"""
        lattice_depth = self.lattice_depth
        lattice_ratio = self.lattice_ratio

        @jit(nopython=True, cache=True)
        def _is_free(model_params):
            """"""
            v0 = lattice_depth(model_params)
            r = lattice_ratio(model_params)
            if v0 <= 1e-10:
                return True
            elif r <= 1e-10:
                return True
            else:
                return False

        return _is_free

    @cached_property
    def is_ideal(self):
        """"""
        interaction_strength = self.interaction_strength

        @jit(nopython=True, cache=True)
        def _is_ideal(model_params):
            """"""
            gn = interaction_strength(model_params)
            if gn <= 1e-10:
                return True
            else:
                return False

        return _is_ideal

    @cached_property
    def potential(self):
        """"""
        return potential_func


# noinspection PyUnusedLocal
@jit(nopython=True, cache=True)
def potential_func(z, v0, r, gn):
    """Calculates the potential energy of the Bose gas due to the
     external potential.

     :param z: The current configuration of the positions of the
               particles.
     :param v0:
     :param r:
     :param gn:
     :return:
    """
    z_a, z_b = 1 / (1 + r), r / (1 + r)
    z_cell = z % 1.
    return v0 if z_a < z_cell else 0.
