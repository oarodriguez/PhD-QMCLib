import typing as t
from abc import ABCMeta, abstractmethod
from itertools import islice
from pathlib import Path

import attr
import h5py
import numpy as np
import tqdm

from my_research_libs.qmc_base import (
    dmc as dmc_base, model as model_base
)
from my_research_libs.qmc_base.dmc import SSFPartSlot
from my_research_libs.qmc_data.dmc import (
    EnergyBlocks, NumWalkersBlocks, PropBlocks, PropsDataBlocks,
    PropsDataSeries, SSFBlocks, SSFPartBlocks, SamplingData, WeightBlocks
)
from my_research_libs.util.attr import str_validator
from .logging import exec_logger

__all__ = [
    'Proc',
    'ProcInput',
    'ProcInputError',
    'ProcResult',
    'SSFEstSpec'
]

DMC_TASK_LOG_NAME = f'DMC Sampling'
VMC_SAMPLING_LOG_NAME = 'VMC Sampling'


class SSFEstSpec(metaclass=ABCMeta):
    """Structure factor estimator basic config."""
    num_modes: int
    as_pure_est: bool
    pfw_num_time_steps: int


class ProcInputError(ValueError):
    """Flags an invalid input for a DMC calculation procedure."""
    pass


@attr.s(auto_attribs=True)
class ProcInput(metaclass=ABCMeta):
    """Represents the input for the DMC calculation procedure."""
    # The state of the DMC procedure input.
    # NOTE: Is this class necessary? 🤔
    state: dmc_base.State


class IOHandler(metaclass=ABCMeta):
    """"""

    #: A tag to identify this handler.
    type: str

    @classmethod
    @abstractmethod
    def from_config(cls, config: t.Mapping):
        """

        :param config:
        :return:
        """
        pass

    @abstractmethod
    def load(self, base_path: Path = None):
        pass

    @abstractmethod
    def save(self, data: 'ProcResult',
             base_path: Path = None):
        pass


class ModelSysConfHandler(IOHandler, metaclass=ABCMeta):
    """"""

    dist_type: str


@attr.s(auto_attribs=True, frozen=True)
class NpyFileHandler(IOHandler, metaclass=ABCMeta):
    """"""
    # NOTE: It could be useful in the future...

    location: str = attr.ib(validator=str_validator)

    #: A tag to identify this handler.
    type: str


@attr.s(auto_attribs=True, frozen=True)
class RawHDF5FileHandler(IOHandler, metaclass=ABCMeta):
    """A handler for HDF5 files without a specific structure."""

    location: str = attr.ib(validator=str_validator)

    group: str = attr.ib(validator=str_validator)

    dataset: str = attr.ib(validator=str_validator)

    #: A tag to identify this handler.
    type: str


class HDF5FileHandlerGroupError(ValueError):
    """Flags an error occurring when saving data to an HDF5 file."""
    pass


class HDF5FileHandler(IOHandler, metaclass=ABCMeta):
    """A handler for properly structured HDF5 files."""

    #: Path to the file.
    location: Path

    #: The HDF5 group in the file to read and/or write data.
    group: str

    #: A tag to identify this handler.
    type: str

    def init_main_groups(self, h5_file: h5py.File):
        """Initialize sub-groups to store the data.

        :param h5_file:
        :return:
        """
        base_group = h5_file.require_group(self.group)

        if 'dmc' in base_group:
            raise HDF5FileHandlerGroupError("Unable to create 'dmc' group "
                                            "(name already exists)")

        dmc_group = base_group.require_group('dmc')
        dmc_group.require_group('proc_spec')
        dmc_group.require_group('state')

        data_group = dmc_group.require_group('data')
        blocks_group = data_group.require_group('blocks')
        blocks_group.require_group('energy')
        blocks_group.require_group('weight')
        blocks_group.require_group('num_walkers')

    def save(self, proc_result: 'ProcResult',
             base_path: Path = None):
        """Save a DMC procedure result to file.

        :param proc_result:
        :param base_path:
        :return:
        """
        location = self.location
        if location.is_absolute():
            file_path = location
        else:
            file_path = base_path / location

        h5_file = h5py.File(file_path)
        with h5_file:
            #
            self.init_main_groups(h5_file)

            self.save_proc(proc_result.proc, h5_file)

            self.save_state(proc_result.state, h5_file)

            self.save_data_blocks(proc_result.data, h5_file)

            h5_file.flush()

    @abstractmethod
    def get_proc_config(self, proc: 'Proc'):
        """

        :param proc:
        :return:
        """
        pass

    def save_state(self, state: dmc_base.State,
                   h5_file: h5py.File):
        """

        :param state:
        :param h5_file:
        :return:
        """
        group_name = self.group
        base_group = h5_file.get(group_name)
        state_group = base_group.require_group('dmc/state')

        state_group.create_dataset('branching_spec', data=state.branching_spec)
        state_group.create_dataset('confs', data=state.confs)
        state_group.create_dataset('props', data=state.props)

        state_group.attrs.update({
            'energy': state.energy,
            'weight': state.weight,
            'num_walkers': state.num_walkers,
            'ref_energy': state.ref_energy,
            'accum_energy': state.accum_energy,
            'max_num_walkers': state.max_num_walkers
        })

    def save_proc(self, proc: 'Proc', h5_file: h5py.File):
        """

        :param proc:
        :param h5_file:
        :return:
        """
        group_name = self.group
        base_group = h5_file.require_group(group_name)
        proc_group = base_group.require_group('dmc/proc_spec')

        proc_config = self.get_proc_config(proc)
        model_spec = proc_config.pop('model_spec')

        model_spec_group = proc_group.require_group('model_spec')
        model_spec_group.attrs.update(**model_spec)

        ssf_spec_config = proc_config.pop('ssf_spec', None)
        if ssf_spec_config is not None:
            ssf_spec_group = proc_group.require_group('ssf_spec')
            ssf_spec_group.attrs.update(**ssf_spec_config)

        proc_group.attrs.update(proc_config)

    def save_data_blocks(self, data: SamplingData,
                         h5_file: h5py.File):
        """

        :param data:
        :param h5_file:
        :return:
        """
        group_name = self.group
        base_group = h5_file.get(group_name)
        blocks_group = base_group.require_group('dmc/data/blocks')

        data_blocks = data.blocks
        energy_blocks = data_blocks.energy
        energy_group = blocks_group.require_group('energy')
        self.save_prop_blocks(energy_blocks, energy_group)

        weight_blocks = data_blocks.weight
        weight_group = blocks_group.require_group('weight')
        self.save_prop_blocks(weight_blocks, weight_group,
                              has_weight_totals=False)

        num_walkers_blocks = data_blocks.num_walkers
        num_walkers_group = blocks_group.require_group('num_walkers')
        self.save_prop_blocks(num_walkers_blocks, num_walkers_group,
                              has_weight_totals=False)

        ssf_blocks = data_blocks.ss_factor
        if ssf_blocks is not None:
            # Save each part of S(k).
            fdk_sqr_abs_group = \
                blocks_group.require_group('ss_factor/fdk_sqr_abs')

            self.save_prop_blocks(ssf_blocks.fdk_sqr_abs_part,
                                  fdk_sqr_abs_group)

            fdk_real_group = \
                blocks_group.require_group('ss_factor/fdk_real')

            self.save_prop_blocks(ssf_blocks.fdk_real_part, fdk_real_group)

            fdk_imag_group = \
                blocks_group.require_group('ss_factor/fdk_imag')

            self.save_prop_blocks(ssf_blocks.fdk_imag_part, fdk_imag_group)

    @staticmethod
    def save_prop_blocks(blocks: PropBlocks,
                         group: h5py.Group,
                         has_weight_totals: bool = True):
        """

        :param blocks:
        :param group:
        :param has_weight_totals:
        :return:
        """
        group.create_dataset('totals', data=blocks.totals)

        if has_weight_totals:
            group.create_dataset('weight_totals', data=blocks.weight_totals)

        group.attrs.update({
            'num_blocks': blocks.num_blocks,
            'num_time_steps_block': blocks.num_time_steps_block
        })

    @abstractmethod
    def load_proc(self, h5_file):
        """"""
        pass

    def load_state(self, h5_file: h5py.File):
        """

        :param h5_file:
        :return:
        """
        group_name = self.group
        base_group = h5_file.get(group_name, None)
        if base_group is None:
            raise HDF5FileHandlerGroupError(
                    f"unable to read '{group_name}' group (name "
                    f"does not exists)"
            )

        state_group = base_group.require_group('dmc/state')

        branching_spec = state_group.get('branching_spec').value
        state_confs = state_group.get('confs').value
        state_props = state_group.get('props').value

        return dmc_base.State(confs=state_confs,
                              props=state_props,
                              branching_spec=branching_spec,
                              **state_group.attrs)

    def load_data_blocks(self, h5_file: h5py.File):
        """

        :param h5_file:
        :return:
        """
        group_name = self.group
        base_group = h5_file.require_group(group_name)
        blocks_group = base_group.require_group('dmc/data/blocks')

        energy_group = blocks_group.require_group('energy')
        blocks_data = self.load_prop_blocks_data(energy_group)
        energy_blocks = EnergyBlocks(**blocks_data)

        weight_group = blocks_group.require_group('weight')
        blocks_data = self.load_prop_blocks_data(weight_group,
                                                 has_weight_totals=False)
        weight_blocks = WeightBlocks(**blocks_data)

        num_walkers_group = blocks_group.require_group('num_walkers')
        blocks_data = self.load_prop_blocks_data(num_walkers_group,
                                                 has_weight_totals=False)
        num_walkers_blocks = NumWalkersBlocks(**blocks_data)

        ss_factor_group = blocks_group.require_group('ss_factor')
        if ss_factor_group is not None:
            #
            fdk_sqr_abs_group = \
                blocks_group.require_group('ss_factor/fdk_sqr_abs')
            blocks_data = self.load_prop_blocks_data(fdk_sqr_abs_group)
            fdk_sqr_abs_blocks = SSFPartBlocks(**blocks_data)

            fdk_real_group = \
                blocks_group.require_group('ss_factor/fdk_real')
            blocks_data = self.load_prop_blocks_data(fdk_real_group)
            fdk_real_blocks = SSFPartBlocks(**blocks_data)

            fdk_imag_group = \
                blocks_group.require_group('ss_factor/fdk_imag')
            blocks_data = self.load_prop_blocks_data(fdk_imag_group)
            fdk_imag_blocks = SSFPartBlocks(**blocks_data)

            ssf_blocks = SSFBlocks(fdk_sqr_abs_blocks,
                                   fdk_real_blocks,
                                   fdk_imag_blocks)

        else:
            ssf_blocks = None

        return PropsDataBlocks(energy_blocks,
                               weight_blocks,
                               num_walkers_blocks,
                               ssf_blocks)

    @staticmethod
    def load_prop_blocks_data(group: h5py.Group,
                              has_weight_totals: bool = True):
        """

        :param group:
        :param has_weight_totals:
        :return:
        """
        totals = group.get('totals').value
        prop_blocks_data = {'totals': totals}

        prop_blocks_data.update(group.attrs.items())

        if has_weight_totals:
            weight_totals = group.get('weight_totals').value
            prop_blocks_data.update(weight_totals=weight_totals)

        return prop_blocks_data


class IOHandlerSpec(metaclass=ABCMeta):
    """"""

    type: str

    spec: IOHandler

    @classmethod
    @abstractmethod
    def from_config(cls, config: t.Mapping):
        pass

    def load(self):
        """"""
        return self.spec.load()

    def save(self, data: 'ProcResult'):
        """"""
        return self.spec.save(data)


@attr.s(auto_attribs=True)
class ProcIO:
    """"""
    #:
    input: IOHandlerSpec

    #:
    output: t.Optional[IOHandlerSpec] = None

    @classmethod
    @abstractmethod
    def from_config(cls, config: t.Mapping):
        """

        :param config:
        :return:
        """
        # Extract the input spec.
        pass


class ProcResult:
    """Result of the DMC estimator sampling."""

    #: The last state of the sampling.
    state: dmc_base.State

    #: The sampling object used to generate the results.
    proc: 'Proc'

    #: The data generated during the sampling.
    data: SamplingData


class Proc(metaclass=ABCMeta):
    """DMC sampling procedure spec."""

    #: The model spec.
    model_spec: model_base.Spec

    #: The "time-step" (squared, average move spread) of the sampling.
    time_step: float

    #: The maximum wight of the population of walkers.
    max_num_walkers: int

    #: The average total weight of the population of walkers.
    target_num_walkers: int

    #: Multiplier for the population control during the branching stage.
    num_walkers_control_factor: t.Optional[float]

    #: The seed of the pseudo-RNG used to realize the sampling.
    rng_seed: t.Optional[int]

    #: The number of batches of the sampling.
    num_batches: int

    #: Number of time steps per batch.
    num_time_steps_batch: int

    #: The number of batches to discard.
    burn_in_batches: t.Optional[int]

    #: Keep the estimator values for all the time steps.
    keep_iter_data: bool

    #: Remaining batches
    remaining_batches: t.Optional[int]

    # *** Estimators configuration ***
    ssf_spec: t.Optional[SSFEstSpec]

    def __attrs_post_init__(self):
        """Post-initialization stage."""
        if self.burn_in_batches is None:
            burn_in_batches = max(1, self.num_batches // 8)
            object.__setattr__(self, 'burn_in_batches', burn_in_batches)

    @classmethod
    @abstractmethod
    def from_config(cls, config: t.Mapping):
        pass

    @classmethod
    @abstractmethod
    def evolve(cls, config: t.Mapping):
        pass

    @property
    def should_eval_ssf(self):
        """"""
        return False if self.ssf_spec is None else True

    @property
    @abstractmethod
    def sampling(self) -> dmc_base.Sampling:
        pass

    @abstractmethod
    def build_input_from_model(self, proc_io_input: IOHandlerSpec):
        """

        :param proc_io_input:
        :return:
        """
        pass

    @abstractmethod
    def build_input_from_result(self, proc_result: ProcResult):
        pass

    @abstractmethod
    def build_result(self, state: dmc_base.State,
                     sampling: dmc_base.Sampling,
                     data: SamplingData):
        """

        :param state: The last state of the sampling.
        :param data: The data generated during the sampling.
        :param sampling: The sampling object used to generate the results.
        :return:
        """
        pass

    def checkpoint(self):
        """"""
        pass

    def exec(self, proc_input: ProcInput):
        """

        :param proc_input:
        :return:
        """
        energy_field = dmc_base.IterProp.ENERGY
        weight_field = dmc_base.IterProp.WEIGHT
        num_walkers_field = dmc_base.IterProp.NUM_WALKERS
        ref_energy_field = dmc_base.IterProp.REF_ENERGY
        accum_energy_field = dmc_base.IterProp.ACCUM_ENERGY

        num_batches = self.num_batches
        num_time_steps_batch = self.num_time_steps_batch
        target_num_walkers = self.target_num_walkers
        burn_in_batches = self.burn_in_batches
        keep_iter_data = self.keep_iter_data

        # Alias 😐.
        nts_batch = num_time_steps_batch

        # Structure factor configuration.
        ssf_spec = self.ssf_spec
        should_eval_ssf = self.should_eval_ssf

        exec_logger.info('Starting DMC sampling...')
        exec_logger.info(f'Using an average of {target_num_walkers} walkers.')
        exec_logger.info(f'Sampling {num_batches} batches of time.')
        exec_logger.info(f'Sampling {num_time_steps_batch} time-steps '
                         f'per batch.')

        # We will burn-in the first ten percent of the sampling chain.
        if burn_in_batches is None:
            burn_in_batches = num_batches // 8
        else:
            burn_in_batches = burn_in_batches

        sampling = self.sampling
        if not isinstance(proc_input, ProcInput):
            raise ProcInputError('the input data for the DMC procedure is '
                                 'not valid')

        # The estimator sampling iterator.
        ini_state = proc_input.state
        batches_iter = sampling.batches(ini_state, num_time_steps_batch)

        # Current batch data.
        batch_data = None

        if burn_in_batches:

            exec_logger.info('Computing DMC burn-in stage...')

            exec_logger.info(f'A total of {burn_in_batches} batches will be '
                             f'discarded.')

            # Burn-in stage.
            pgs_bar = tqdm.tqdm(total=burn_in_batches, dynamic_ncols=True)
            with pgs_bar:
                for batch_data in islice(batches_iter, burn_in_batches):
                    # Burn, burn, burn...
                    pgs_bar.update(1)

            exec_logger.info('Burn-in stage completed.')

        else:
            exec_logger.info(f'No burn-in batches requested')

        # Main containers of data.
        if keep_iter_data:
            iter_props_shape = num_batches, nts_batch
        else:
            iter_props_shape = num_batches,

        props_blocks_data = \
            np.empty(iter_props_shape, dtype=dmc_base.iter_props_dtype)

        props_energy = props_blocks_data[energy_field]
        props_weight = props_blocks_data[weight_field]
        props_num_walkers = props_blocks_data[num_walkers_field]
        props_ref_energy = props_blocks_data[ref_energy_field]
        props_accum_energy = props_blocks_data[accum_energy_field]

        if should_eval_ssf:
            # The shape of the structure factor array.
            ssf_num_modes = ssf_spec.num_modes
            # noinspection PyTypeChecker
            ssf_num_parts = len(SSFPartSlot)

            if keep_iter_data:
                ssf_shape = \
                    num_batches, nts_batch, ssf_num_modes, ssf_num_parts
            else:
                ssf_shape = num_batches, ssf_num_modes, ssf_num_parts

            # S(k) batch.
            ssf_blocks_data = np.zeros(ssf_shape, dtype=np.float64)

        else:
            ssf_blocks_data = None

        # Reduction factor for pure estimators.
        pure_est_reduce_factor = np.ones(num_batches, dtype=np.float64)

        # The effective batches to calculate the estimators.
        eff_batches: dmc_base.T_SBatchesIter = \
            islice(batches_iter, num_batches)

        # Enumerated effective batches.
        enum_eff_batches: dmc_base.T_E_SBatchesIter \
            = enumerate(eff_batches)

        exec_logger.info('Starting the evaluation of estimators...')

        if should_eval_ssf:
            exec_logger.info(f'Static structure factor is going to be '
                             f'calculated.')
            exec_logger.info(f'A total of {ssf_spec.num_modes} k-modes will '
                             f'be used as input for S(k).')

        with tqdm.tqdm(total=num_batches, dynamic_ncols=True) as pgs_bar:

            for batch_idx, batch_data in enum_eff_batches:

                batch_props = batch_data.iter_props
                energy = batch_props[energy_field]
                weight = batch_props[weight_field]
                num_walkers = batch_props[num_walkers_field]
                ref_energy = batch_props[ref_energy_field]
                accum_energy = batch_props[accum_energy_field]

                # NOTE: Should we return the iter_props by default? 🤔🤔🤔
                #  If we keep the whole sampling data, i.e.,
                #  ``keep_iter_data`` is True, then it has not much sense to
                #  realize the reblocking for each batch of data.
                #  Let's think about it...

                if keep_iter_data:

                    # Store all the information of the DMC sampling.
                    props_blocks_data[batch_idx] = batch_props[:]

                else:

                    weight_sum = weight.sum()
                    props_energy[batch_idx] = energy.sum()
                    props_weight[batch_idx] = weight_sum
                    props_num_walkers[batch_idx] = num_walkers.sum()
                    props_ref_energy[batch_idx] = ref_energy[-1]
                    props_accum_energy[batch_idx] = accum_energy[-1]

                    reduce_fac = num_walkers[-1] / weight_sum
                    pure_est_reduce_factor[batch_idx] = reduce_fac

                # Handling the structure factor results.
                if should_eval_ssf:

                    iter_ssf_array = batch_data.iter_ssf

                    if keep_iter_data:
                        # Keep a copy of the generated data of the sampling.
                        ssf_blocks_data[batch_idx] = iter_ssf_array[:]

                    else:

                        if ssf_spec.as_pure_est:
                            # Get only the last element of the batch.
                            ssf_blocks_data[batch_idx] = \
                                iter_ssf_array[nts_batch - 1]

                        else:
                            # Make the reduction.
                            ssf_blocks_data[batch_idx] = \
                                iter_ssf_array.sum(axis=0)

                rem_batches = num_batches - batch_idx - 1
                object.__setattr__(self, 'remaining_batches', rem_batches)

                # logger.info(f'Batch #{batch_idx:d} completed')
                pgs_bar.update()

        # Pick the last state
        if batch_data is None:
            last_state = None
        else:
            last_state = batch_data.last_state

        exec_logger.info('Evaluation of estimators completed.')
        exec_logger.info('DMC sampling completed.')

        # Create block objects with the totals of each block data.
        reduce_data = True if keep_iter_data else False

        energy_blocks = \
            EnergyBlocks.from_data(num_batches, nts_batch,
                                   props_blocks_data, reduce_data)

        weight_blocks = \
            WeightBlocks.from_data(num_batches, nts_batch,
                                   props_blocks_data, reduce_data)

        num_walkers_blocks = \
            NumWalkersBlocks.from_data(num_batches, nts_batch,
                                       props_blocks_data, reduce_data)

        if should_eval_ssf:

            ssf_blocks = \
                SSFBlocks.from_data(num_batches, nts_batch,
                                    ssf_blocks_data,
                                    props_blocks_data, reduce_data,
                                    ssf_spec.as_pure_est,
                                    pure_est_reduce_factor)

        else:
            ssf_blocks = None

        data_blocks = PropsDataBlocks(energy_blocks,
                                      weight_blocks,
                                      num_walkers_blocks,
                                      ssf_blocks)

        if keep_iter_data:
            data_series = PropsDataSeries(props_blocks_data,
                                          ssf_blocks_data)
        else:
            data_series = None

        sampling_data = SamplingData(data_blocks, data_series)

        # NOTE: Should we return a new instance?
        # sampling = attr.evolve(sampling)

        return self.build_result(last_state,
                                 sampling=sampling,
                                 data=sampling_data)
