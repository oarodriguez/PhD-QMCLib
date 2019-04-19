import typing as t
from abc import ABCMeta, abstractmethod
from itertools import islice

import attr
import numpy as np
import tqdm

from my_research_libs.qmc_base import (
    dmc as dmc_base, model as model_base
)
from my_research_libs.qmc_base.dmc import SSFPartSlot
from my_research_libs.qmc_base.jastrow.model import DensityPartSlot
from .data import (
    DensityBlocks, EnergyBlocks, NumWalkersBlocks, PropsDataBlocks,
    PropsDataSeries, SSFBlocks, SamplingData, WeightBlocks
)
from ..logging import exec_logger

DMC_TASK_LOG_NAME = f'DMC Sampling'
VMC_SAMPLING_LOG_NAME = 'VMC Sampling'


class ModelSysConfSpec(metaclass=ABCMeta):
    """Handler to build inputs from system configurations."""

    #:
    dist_type: str

    #:
    num_sys_conf: t.Optional[int]


class DensityEstSpec(metaclass=ABCMeta):
    """Structure factor estimator basic config."""
    num_bins: int
    as_pure_est: bool


class SSFEstSpec(metaclass=ABCMeta):
    """Structure factor estimator basic config."""
    num_modes: int
    as_pure_est: bool


@attr.s(auto_attribs=True)
class ProcInput(metaclass=ABCMeta):
    """Represents the input for the DMC calculation procedure."""
    # The state of the DMC procedure input.
    # NOTE: Is this class necessary? 🤔
    state: dmc_base.State

    @classmethod
    @abstractmethod
    def from_model_sys_conf_spec(cls, sys_conf_spec: ModelSysConfSpec,
                                 proc: 'Proc'):
        """

        :param sys_conf_spec:
        :param proc:
        :return:
        """
        pass

    @classmethod
    @abstractmethod
    def from_result(cls, proc_result: 'ProcResult',
                    proc: 'Proc'):
        """

        :param proc_result:
        :param proc:
        :return:
        """
        pass


class ProcInputError(ValueError):
    """Flags an invalid input for a DMC calculation procedure."""
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
    #:
    density_spec: t.Optional[DensityEstSpec]

    #:
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

    @abstractmethod
    def as_config(self) -> t.Dict:
        """Converts the procedure to a dictionary / mapping object."""
        pass

    @classmethod
    @abstractmethod
    def evolve(cls, config: t.Mapping):
        pass

    @property
    def should_eval_density(self):
        """"""
        return False if self.density_spec is None else True

    @property
    def should_eval_ssf(self):
        """"""
        return False if self.ssf_spec is None else True

    @property
    @abstractmethod
    def sampling(self) -> dmc_base.Sampling:
        pass

    @abstractmethod
    def build_result(self, state: dmc_base.State,
                     sampling: dmc_base.Sampling,
                     data: SamplingData) -> ProcResult:
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

        # Density configuration.
        density_spec = self.density_spec
        should_eval_density = self.should_eval_density

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

        if should_eval_density:
            # The shape of the structure factor array.
            density_num_bins = density_spec.num_bins
            # noinspection PyTypeChecker
            density_num_parts = len(DensityPartSlot)

            if keep_iter_data:
                density_shape = \
                    num_batches, nts_batch, density_num_bins, density_num_parts
            else:
                density_shape = \
                    num_batches, density_num_bins, density_num_parts

            # Density batch.
            density_blocks_data = np.zeros(density_shape, dtype=np.float64)

        else:
            density_blocks_data = None

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

                    # Handling the density results.
                    if should_eval_density:
                        iter_density_array = batch_data.iter_density
                        density_blocks_data[batch_idx] = iter_density_array[:]

                    # Handling the structure factor results.
                    if should_eval_ssf:
                        iter_ssf_array = batch_data.iter_ssf
                        ssf_blocks_data[batch_idx] = iter_ssf_array[:]

                else:

                    weight_sum = weight.sum()
                    props_energy[batch_idx] = energy.sum()
                    props_weight[batch_idx] = weight_sum
                    props_num_walkers[batch_idx] = num_walkers.sum()
                    props_ref_energy[batch_idx] = ref_energy[-1]
                    props_accum_energy[batch_idx] = accum_energy[-1]

                    reduce_fac = num_walkers[nts_batch - 1] / weight_sum
                    pure_est_reduce_factor[batch_idx] = reduce_fac

                    # Handling the density factor results.
                    if should_eval_density:

                        iter_density_array = batch_data.iter_density

                        if density_spec.as_pure_est:
                            # Get only the last element of the batch.
                            density_blocks_data[batch_idx] = \
                                iter_density_array[nts_batch - 1]

                        else:
                            # Make the reduction.
                            density_blocks_data[batch_idx] = \
                                iter_density_array.sum(axis=0)

                    # Handling the structure factor results.
                    if should_eval_ssf:

                        iter_ssf_array = batch_data.iter_ssf

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

        if should_eval_density:
            main_slot = DensityPartSlot.MAIN
            density_data = density_blocks_data[..., main_slot]
            density_blocks = \
                DensityBlocks.from_data(num_batches, nts_batch,
                                        density_data,
                                        props_blocks_data, reduce_data,
                                        density_spec.as_pure_est,
                                        pure_est_reduce_factor)

        else:
            density_blocks = None

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
                                      density_blocks,
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