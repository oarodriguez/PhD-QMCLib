import typing as t
from abc import ABCMeta, abstractmethod
from itertools import islice

import attr
import numpy as np
import tqdm

from my_research_libs.qmc_base import model as model_base, vmc as vmc_base
from ..data.vmc import (
    EnergyBlocks, PropsDataBlocks, PropsDataSeries, SSFBlocks, SamplingData
)
from ..logging import exec_logger


class ModelSysConfSpec(metaclass=ABCMeta):
    """Handler to build inputs from system configurations."""
    #:
    dist_type: str


class SSFEstSpec(metaclass=ABCMeta):
    """Structure factor estimator basic config."""
    #: The number of momenta modes.
    num_modes: int


@attr.s(auto_attribs=True)
class ProcInput(metaclass=ABCMeta):
    """Represents the input for the VMC calculation procedure."""
    # The state of the VMC procedure input.
    state: vmc_base.State


class ProcInputError(ValueError):
    """Flags an invalid input for a VMC calculation procedure."""
    pass


class ProcResult:
    """Result of the VMC estimator sampling."""

    #: The last state of the sampling.
    state: vmc_base.State

    #: The procedure object used to generate the results.
    proc: 'Proc'

    #: The data generated during the sampling.
    data: SamplingData


class Proc(metaclass=ABCMeta):
    """VMC Sampling procedure spec."""

    #: The model spec.
    model_spec: model_base.Spec

    #: The spread magnitude of the random moves for the sampling.
    move_spread: float

    #: The seed of the pseudo-RNG used to explore the configuration space.
    rng_seed: t.Optional[int]

    #: The number of blocks of the sampling.
    num_blocks: int

    #: Number of steps per block.
    num_steps_block: int

    #: The number of blocks to discard.
    burn_in_blocks: t.Optional[int]

    #: Keep the estimator values for all the time steps.
    keep_iter_data: bool

    # *** Estimators configuration ***
    ssf_spec: t.Optional[SSFEstSpec]

    @abstractmethod
    def as_config(self) -> t.Dict:
        """Converts the procedure to a dictionary / mapping object."""
        pass

    @property
    def should_eval_ssf(self):
        """Whether or not to evaluate the static structure factor."""
        return False if self.ssf_spec is None else True

    @property
    @abstractmethod
    def sampling(self) -> vmc_base.Sampling:
        """VMC sampling object."""
        pass

    @abstractmethod
    def describe_model_spec(self):
        """Describe the spec of the model."""
        pass

    def describe_sampling(self):
        """Describe the VMC sampling."""
        move_spread = self.move_spread
        rng_seed = self.rng_seed
        num_blocks = self.num_blocks
        num_steps_block = self.num_steps_block

        exec_logger.info(f'Sampling {num_blocks} blocks of steps...')
        exec_logger.info(f'Sampling {num_steps_block} steps per block...')
        exec_logger.info(f'Using uniform random moves of maximum spread '
                         f'{move_spread} LKP...')
        if rng_seed is None:
            exec_logger.info(f'No random seed was specified...')
        else:
            exec_logger.info(f'The random seed of the sampling '
                             f'is {rng_seed}...')

    @abstractmethod
    def build_result(self, state: vmc_base.State,
                     sampling: vmc_base.Sampling,
                     data: SamplingData) -> ProcResult:
        """Build the procedure result object.

        :param state: The last state of the sampling.
        :param data: The data generated during the sampling.
        :param sampling: The sampling object used to generate the results.
        :return:
        """
        pass

    def exec(self, proc_input: ProcInput):
        """Trigger the execution of the procedure.

        :param proc_input:
        :return:
        """
        wf_abs_log_field = vmc_base.IterProp.WF_ABS_LOG
        energy_field = vmc_base.IterProp.ENERGY

        num_blocks = self.num_blocks
        num_steps_block = self.num_steps_block
        burn_in_blocks = self.burn_in_blocks
        keep_iter_data = self.keep_iter_data

        # Alias 😐.
        ns_block = num_steps_block

        # Static structure factor configuration.
        ssf_spec = self.ssf_spec
        should_eval_ssf = self.should_eval_ssf

        exec_logger.info('Starting VMC sampling...')

        self.describe_model_spec()
        self.describe_sampling()

        # New sampling instance
        sampling = self.sampling
        if not isinstance(proc_input, ProcInput):
            raise ProcInputError('the input data for the VMC procedure is '
                                 'not valid')
        ini_state = proc_input.state
        blocks_iter = sampling.blocks(num_steps_block, ini_state)

        # By default burn-in all but the last block.
        # We will burn-in the first ten percent of the sampling chain.
        if burn_in_blocks is None:
            burn_in_blocks = num_blocks // 8
        else:
            burn_in_blocks = burn_in_blocks

        # Current block data.
        block_data = None

        if burn_in_blocks:
            # Burn-in stage.
            exec_logger.info('Executing VMC burn-in stage...')
            exec_logger.info(f'A total of {burn_in_blocks} blocks will be '
                             f'discarded.')

            pgs_bar = tqdm.tqdm(total=burn_in_blocks, dynamic_ncols=True)
            with pgs_bar:
                for block_data in islice(blocks_iter, burn_in_blocks):
                    # Burn blocks...
                    pgs_bar.update()

            exec_logger.info('Burn-in stage completed.')
        else:
            exec_logger.info(f'No burn-in blocks requested.')

        # Main containers of data.
        if keep_iter_data:
            iter_props_shape = num_blocks, ns_block
        else:
            iter_props_shape = num_blocks,

        props_blocks_data = \
            np.empty(iter_props_shape, dtype=vmc_base.iter_props_dtype)

        props_wf_abs_log = props_blocks_data[wf_abs_log_field]
        props_energy = props_blocks_data[energy_field]

        if should_eval_ssf:
            # The shape of the structure factor array.
            ssf_num_modes = ssf_spec.num_modes
            # noinspection PyTypeChecker
            ssf_num_parts = len(vmc_base.SSFPartSlot)

            if keep_iter_data:
                ssf_shape = \
                    num_blocks, ns_block, ssf_num_modes, ssf_num_parts
            else:
                ssf_shape = num_blocks, ssf_num_modes, ssf_num_parts

            # S(k) block.
            ssf_blocks_data = np.zeros(ssf_shape, dtype=np.float64)

        else:
            ssf_blocks_data = None

        # The effective blocks to calculate the estimators.
        eff_blocks: vmc_base.T_SBlocksIter = \
            islice(blocks_iter, num_blocks)

        # Enumerated effective blocks.
        enum_eff_blocks: vmc_base.T_E_SBlocksIter \
            = enumerate(eff_blocks)

        # Get the last block.
        pgs_bar = tqdm.tqdm(total=num_blocks, dynamic_ncols=True)
        with pgs_bar:
            for block_idx, block_data in enum_eff_blocks:
                #
                block_props = block_data.iter_props
                wf_abs_log: np.ndarray = block_props[wf_abs_log_field]
                energy: np.ndarray = block_props[energy_field]

                if keep_iter_data:
                    # Store all the information of the DMC sampling.
                    props_blocks_data[block_idx] = block_props[:]
                else:
                    props_energy[block_idx] = energy.mean()
                    props_wf_abs_log[block_idx] = wf_abs_log.mean()

                if should_eval_ssf:
                    # Get the static structure factor data.
                    iter_ssf_array = block_data.iter_ssf
                    if keep_iter_data:
                        # Keep a copy of the generated data of the sampling.
                        ssf_blocks_data[block_idx] = iter_ssf_array[:]
                    else:
                        # Make the reduction.
                        ssf_blocks_data[block_idx] = \
                            iter_ssf_array.mean(axis=0)

                pgs_bar.update()

        exec_logger.info('Evaluation of estimators completed.')
        exec_logger.info('VMC Sampling completed.')

        # Pick the last state
        if block_data is None:
            last_state = None
        else:
            last_state = block_data.last_state

        # Create block objects with the totals of each block data.
        reduce_data = True if keep_iter_data else False

        energy_blocks = \
            EnergyBlocks.from_data(num_blocks, ns_block,
                                   props_blocks_data, reduce_data)

        if should_eval_ssf:
            ssf_blocks = \
                SSFBlocks.from_data(num_blocks, ns_block,
                                    ssf_blocks_data, reduce_data)
        else:
            ssf_blocks = None

        data_blocks = PropsDataBlocks(energy_blocks, ss_factor=ssf_blocks)

        if keep_iter_data:
            data_series = PropsDataSeries(props_blocks_data,
                                          ssf_blocks_data)
        else:
            data_series = None

        sampling_data = SamplingData(data_blocks, data_series)

        return self.build_result(last_state, self.sampling, sampling_data)
