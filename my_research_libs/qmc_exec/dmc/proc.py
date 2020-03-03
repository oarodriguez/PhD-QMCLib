import typing as t
from abc import ABCMeta, abstractmethod
from itertools import islice

import numpy as np
import tqdm

from my_research_libs.qmc_base import dmc as dmc_base, model as model_base
from my_research_libs.qmc_base.jastrow.model import DensityPartSlot
from .. import proc as proc_base
from ..data.dmc import (
    DensityBlocks, EnergyBlocks, NumWalkersBlocks, PropsDataBlocks,
    PropsDataSeries, SSFBlocks, SamplingData, WeightBlocks
)
from ..logging import exec_logger

DMC_TASK_LOG_NAME = f'DMC Sampling'
VMC_SAMPLING_LOG_NAME = 'VMC Sampling'


class ModelSysConfSpec(proc_base.ModelSysConfSpec):
    """Handler to build inputs from system configurations."""

    #:
    dist_type: str

    #:
    num_sys_conf: t.Optional[int]


class DensityEstSpec(proc_base.DensityEstSpec):
    """Structure factor estimator basic config."""
    num_bins: int
    as_pure_est: bool


class SSFEstSpec(proc_base.SSFEstSpec):
    """Structure factor estimator basic config."""
    num_modes: int
    as_pure_est: bool


class ProcInput(proc_base.ProcInput, metaclass=ABCMeta):
    """Represents the input for the DMC calculation procedure."""
    # The state of the DMC procedure input.
    state: dmc_base.State


class ProcInputError(ValueError):
    """Flags an invalid input for a DMC calculation procedure."""
    pass


class Proc(proc_base.Proc):
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

    #: The number of blocks of the sampling.
    num_blocks: int

    #: Number of time steps per block.
    num_time_steps_block: int

    #: The number of blocks to discard.
    burn_in_blocks: t.Optional[int]

    #: Keep the estimator values for all the time steps.
    keep_iter_data: bool

    #: Remaining blocks
    remaining_blocks: t.Optional[int]

    # *** Estimators configuration ***
    #:
    density_spec: t.Optional[DensityEstSpec]

    #:
    ssf_spec: t.Optional[SSFEstSpec]

    def __attrs_post_init__(self):
        """Post-initialization stage."""
        if self.burn_in_blocks is None:
            burn_in_blocks = max(1, self.num_blocks // 8)
            object.__setattr__(self, 'burn_in_blocks', burn_in_blocks)

    @property
    @abstractmethod
    def sampling(self) -> dmc_base.Sampling:
        pass

    def describe_sampling(self):
        """Describe the DMC sampling."""
        time_step = self.time_step
        target_num_walkers = self.target_num_walkers
        max_num_walkers = self.max_num_walkers
        nwc_factor = self.num_walkers_control_factor
        rng_seed = self.rng_seed
        num_blocks = self.num_blocks
        num_time_steps_block = self.num_time_steps_block
        burn_in_blocks = self.burn_in_blocks

        # TODO: add unit to message.
        exec_logger.info(f'Using an imaginary time step of {time_step}...')
        exec_logger.info(f'Sampling {num_blocks} blocks of steps...')
        exec_logger.info(f'Sampling {num_time_steps_block} steps per block...')
        exec_logger.info(f'The first {burn_in_blocks} blocks of the sampling '
                         f'will be discarded for statistics...')
        exec_logger.info(f'Targeting an average of {target_num_walkers} '
                         f'random walkers, with a maximum number of '
                         f'{max_num_walkers} walkers...')
        if rng_seed is None:
            exec_logger.info(f'No random seed was specified...')
        else:
            exec_logger.info(f'The random seed of the sampling '
                             f'is {rng_seed}...')
        exec_logger.info(f'using a population control factor '
                         f'of {nwc_factor}...')

    def exec(self, proc_input: ProcInput):
        """

        :param proc_input:
        :return:
        """
        num_blocks = self.num_blocks
        num_time_steps_block = self.num_time_steps_block
        target_num_walkers = self.target_num_walkers
        burn_in_blocks = self.burn_in_blocks
        keep_iter_data = self.keep_iter_data

        # Alias 😐.
        nts_block = num_time_steps_block

        # Density configuration.
        density_spec = self.density_spec
        should_eval_density = self.should_eval_density

        # Structure factor configuration.
        ssf_spec = self.ssf_spec
        should_eval_ssf = self.should_eval_ssf

        exec_logger.info('Starting DMC sampling...')

        self.describe_model_spec()
        self.describe_sampling()

        # We will burn-in the first ten percent of the sampling chain.
        if burn_in_blocks is None:
            burn_in_blocks = num_blocks // 8
        else:
            burn_in_blocks = burn_in_blocks

        sampling = self.sampling
        if not isinstance(proc_input, ProcInput):
            raise ProcInputError('the input data for the DMC procedure is '
                                 'not valid')

        # The estimator sampling iterator.
        ini_state = proc_input.state
        blocks_iter = sampling.blocks(ini_state, num_time_steps_block,
                                      burn_in_blocks)

        # Current block data.
        block_data = None

        if burn_in_blocks:

            exec_logger.info('Computing DMC burn-in stage...')

            exec_logger.info(f'A total of {burn_in_blocks} blocks will be '
                             f'discarded.')

            # Burn-in stage.
            pgs_bar = tqdm.tqdm(total=burn_in_blocks, dynamic_ncols=True)
            with pgs_bar:
                for block_data in islice(blocks_iter, burn_in_blocks):
                    # Burn, burn, burn...
                    pgs_bar.update(1)

            exec_logger.info('Burn-in stage completed.')

        else:
            exec_logger.info(f'No burn-in blocks requested')

        # Main containers of data.
        if keep_iter_data:
            iter_props_shape = num_blocks, nts_block
        else:
            iter_props_shape = num_blocks,

        props_blocks_data = \
            self.sampling.core_funcs.init_props_data_block(iter_props_shape)

        props_energy = props_blocks_data.energy
        props_weight = props_blocks_data.weight
        props_num_walkers = props_blocks_data.num_walkers
        props_ref_energy = props_blocks_data.ref_energy
        props_accum_energy = props_blocks_data.accum_energy

        if should_eval_density:
            # The shape of the structure factor array.
            density_num_bins = density_spec.num_bins
            # noinspection PyTypeChecker
            density_num_parts = len(DensityPartSlot)

            if keep_iter_data:
                density_shape = \
                    num_blocks, nts_block, density_num_bins, density_num_parts
            else:
                density_shape = \
                    num_blocks, density_num_bins, density_num_parts

            # Density block.
            density_blocks_data = np.zeros(density_shape, dtype=np.float64)

        else:
            density_blocks_data = None

        if should_eval_ssf:
            # The shape of the structure factor array.
            ssf_num_modes = ssf_spec.num_modes
            # noinspection PyTypeChecker
            ssf_num_parts = len(dmc_base.SSFPartSlot)

            if keep_iter_data:
                ssf_shape = \
                    num_blocks, nts_block, ssf_num_modes, ssf_num_parts
            else:
                ssf_shape = num_blocks, ssf_num_modes, ssf_num_parts

            # S(k) block.
            ssf_blocks_data = np.zeros(ssf_shape, dtype=np.float64)

        else:
            ssf_blocks_data = None

        # Reduction factor for pure estimators.
        pure_est_reduce_factor = np.ones(num_blocks, dtype=np.float64)

        # The effective blocks to calculate the estimators.
        eff_blocks: dmc_base.T_SBlocksIter = \
            islice(blocks_iter, num_blocks)

        # Enumerated effective blocks.
        enum_eff_blocks: dmc_base.T_E_SBlocksIter \
            = enumerate(eff_blocks)

        exec_logger.info('Starting the evaluation of estimators...')

        if should_eval_ssf:
            exec_logger.info(f'Static structure factor is going to be '
                             f'calculated.')
            exec_logger.info(f'A total of {ssf_spec.num_modes} k-modes will '
                             f'be used as input for S(k).')

        with tqdm.tqdm(total=num_blocks, dynamic_ncols=True) as pgs_bar:

            for block_idx, block_data in enum_eff_blocks:

                block_props = block_data.iter_props

                # NOTE: Should we return the iter_props by default? 🤔🤔🤔
                #  If we keep the whole sampling data, i.e.,
                #  ``keep_iter_data`` is True, then it has not much sense to
                #  realize the reblocking for each block of data.
                #  Let's think about it...

                if keep_iter_data:

                    # Store all the information of the DMC sampling.
                    props_energy[block_idx] = block_props.energy[:]
                    props_weight[block_idx] = block_props.weight[:]
                    props_num_walkers[block_idx] = block_props.num_walkers[:]
                    props_ref_energy[block_idx] = block_props.ref_energy[:]
                    props_accum_energy[block_idx] = block_props.accum_energy[:]

                    # Handling the density results.
                    if should_eval_density:
                        iter_density_array = block_data.iter_density
                        density_blocks_data[block_idx] = iter_density_array[:]

                    # Handling the structure factor results.
                    if should_eval_ssf:
                        iter_ssf_array = block_data.iter_ssf
                        ssf_blocks_data[block_idx] = iter_ssf_array[:]

                else:

                    energy = block_props.energy
                    weight = block_props.weight
                    num_walkers = block_props.num_walkers
                    ref_energy = block_props.ref_energy
                    accum_energy = block_props.accum_energy

                    weight_sum = weight.sum()
                    props_energy[block_idx] = energy.sum()
                    props_weight[block_idx] = weight_sum
                    props_num_walkers[block_idx] = num_walkers.sum()
                    props_ref_energy[block_idx] = ref_energy[-1]
                    props_accum_energy[block_idx] = accum_energy[-1]

                    reduce_fac = num_walkers[nts_block - 1] / weight_sum
                    pure_est_reduce_factor[block_idx] = reduce_fac

                    # Handling the density factor results.
                    if should_eval_density:

                        iter_density_array = block_data.iter_density

                        if density_spec.as_pure_est:
                            # Get only the last element of the block.
                            density_blocks_data[block_idx] = \
                                iter_density_array[nts_block - 1]

                        else:
                            # Make the reduction.
                            density_blocks_data[block_idx] = \
                                iter_density_array.sum(axis=0)

                    # Handling the structure factor results.
                    if should_eval_ssf:

                        iter_ssf_array = block_data.iter_ssf

                        if ssf_spec.as_pure_est:
                            # Get only the last element of the block.
                            ssf_blocks_data[block_idx] = \
                                iter_ssf_array[nts_block - 1]

                        else:
                            # Make the reduction.
                            ssf_blocks_data[block_idx] = \
                                iter_ssf_array.sum(axis=0)

                # rem_blocks = num_blocks - block_idx - 1
                # object.__setattr__(self, 'remaining_blocks', rem_blocks)

                # logger.info(f'Block #{block_idx:d} completed')
                pgs_bar.update()

        # Pick the last state
        if block_data is None:
            last_state = None
        else:
            last_state = block_data.last_state

        exec_logger.info('Evaluation of estimators completed.')
        exec_logger.info('DMC sampling completed.')

        # Create block objects with the totals of each block data.
        reduce_data = True if keep_iter_data else False

        energy_blocks = \
            EnergyBlocks.from_data(props_blocks_data, reduce_data)

        weight_blocks = \
            WeightBlocks.from_data(props_blocks_data, reduce_data)

        num_walkers_blocks = \
            NumWalkersBlocks.from_data(props_blocks_data, reduce_data)

        if should_eval_density:
            main_slot = DensityPartSlot.MAIN
            density_data = density_blocks_data[..., main_slot]
            density_blocks = \
                DensityBlocks.from_data(nts_block, density_data,
                                        props_blocks_data, reduce_data,
                                        density_spec.as_pure_est,
                                        pure_est_reduce_factor)

        else:
            density_blocks = None

        if should_eval_ssf:

            ssf_blocks = \
                SSFBlocks.from_data(nts_block, ssf_blocks_data,
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
        return self.build_result(last_state, sampling_data)
