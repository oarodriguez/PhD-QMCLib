from abc import ABCMeta
from pathlib import Path

import h5py

from phd_qmclib.qmc_base import dmc as dmc_base
from .. import io as io_base
from ..data.dmc import SamplingData


class HDF5FileHandler(io_base.HDF5FileHandler, metaclass=ABCMeta):
    """A handler for properly structured HDF5 files."""

    #: Path to the file.
    location: str

    #: The HDF5 group in the file to read and/or write data.
    group: str

    #: Replace any existing data in the file.
    dump_replace: bool

    #: A tag to identify this handler.
    type: str

    @property
    def location_path(self):
        """Return the file location as a ``pathlib.Path`` object."""
        return Path(self.location).absolute()

    @property
    def sampling_type(self):
        return 'dmc'

    def save_state(self, state: dmc_base.State,
                   group: h5py.Group):
        """

        :param state:
        :param group:
        :return:
        """
        group.create_dataset('branching_spec', data=state.branching_spec)
        group.create_dataset('confs', data=state.confs)
        props_group = group.require_group('props')
        props_group.create_dataset('energy', data=state.props.energy)
        props_group.create_dataset('weight', data=state.props.weight)
        props_group.create_dataset('mask', data=state.props.mask)

        group.attrs.update({
            'energy': state.energy,
            'weight': state.weight,
            'num_walkers': state.num_walkers,
            'ref_energy': state.ref_energy,
            'accum_energy': state.accum_energy,
            'max_num_walkers': state.max_num_walkers
        })

    def load_state(self, group: h5py.Group):
        """

        :return:
        """
        branching_spec = group.get('branching_spec')[()]
        state_confs = group.get('confs')[()]
        try:
            # Load data according to the new behavior.
            state_props = group.get('props')
            energy = state_props.get('energy')[()]
            weight = state_props.get('weight')[()]
            mask = state_props.get('mask')[()]
            state_props = dmc_base.StateProps(energy, weight, mask)
        except AttributeError:
            # Load data following the previous (flawed) behavior, so legacy
            # data does not become necessarily useless.
            state_props = group.get('props')[()]
        return dmc_base.State(confs=state_confs,
                              props=state_props,
                              branching_spec=branching_spec,
                              **group.attrs)

    def load_sampling_data(self, group: h5py.Group):
        """

        :param group:
        :return:
        """
        return SamplingData.from_hdf5_data(group)

    def save_sampling_data(self, sampling_data: SamplingData,
                           group: h5py.Group):
        """

        :param sampling_data:
        :param group:
        :return:
        """
        sampling_data.hdf5_export(group)
