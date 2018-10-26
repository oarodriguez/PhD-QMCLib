from abc import ABCMeta
from enum import unique

from numba import jit

from phdthesis_lib.qmc_lib import core, jastrow
from phdthesis_lib.utils import cached_property

__all__ = [
    'Model',
    'ModelCoreFuncs',
    'ModelParams',
    'potential_func'
]


@unique
class ParamName(core.ParamNameEnum):
    """Enumerates the parameters of the model (Bijl-Jastrow type) of a
    quantum system in a Multi-Rods periodic structure.
    """
    LATTICE_DEPTH = 'lattice_depth'
    LATTICE_RATIO = 'lattice_ratio'
    INTERACTION_STRENGTH = 'interaction_strength'
    BOSON_NUMBER = 'boson_number'
    SUPERCELL_SIZE = 'supercell_size'


class ModelParams(core.ParamsSet):
    """Represents the parameters of the model."""
    names = ParamName


class Model(jastrow.Model, metaclass=ABCMeta):
    """"""
    #
    params_cls = ModelParams

    @property
    def lattice_depth(self) -> float:
        """"""
        return self.params[self.params_cls.names.LATTICE_DEPTH]

    @property
    def lattice_ratio(self) -> float:
        """"""
        return self.params[self.params_cls.names.LATTICE_RATIO]

    @property
    def interaction_strength(self) -> float:
        """"""
        return self.params[self.params_cls.names.INTERACTION_STRENGTH]

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


class ModelCoreFuncs(jastrow.ModelCoreFuncs, metaclass=ABCMeta):
    """"""

    params_cls = ModelParams

    @cached_property
    def lattice_depth(self):
        """"""
        param_loc = int(self.params_cls.names.LATTICE_DEPTH.loc)

        @jit(nopython=True, cache=True)
        def _lattice_depth(model_params):
            """"""
            return model_params[param_loc]

        return _lattice_depth

    @cached_property
    def lattice_ratio(self):
        """"""
        param_loc = int(self.params_cls.names.LATTICE_RATIO.loc)

        @jit(nopython=True, cache=True)
        def _lattice_ratio(model_params: tuple) -> float:
            """"""
            return model_params[param_loc]

        return _lattice_ratio

    @cached_property
    def interaction_strength(self):
        """"""
        param_loc = int(self.params_cls.names.INTERACTION_STRENGTH.loc)

        @jit(nopython=True, cache=True)
        def _interaction_strength(model_params):
            """"""
            return model_params[param_loc]

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
