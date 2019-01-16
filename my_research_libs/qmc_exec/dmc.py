import typing as t
from abc import ABCMeta, abstractmethod
from itertools import islice

import attr
import numpy as np
import tqdm

from my_research_libs.qmc_base import (
    dmc as dmc_base, model as model_base, vmc as vmc_base
)
from my_research_libs.qmc_data.dmc import (
    DMCESData, DMCESDataBlocks, DMCESDataSeries, EnergyBlocks,
    NumWalkersBlocks, SSFBlocks, WeightBlocks
)
from .logging import exec_logger

DMC_TASK_LOG_NAME = f'DMC Sampling'
VMC_SAMPLING_LOG_NAME = 'VMC Sampling'


class VMCSamplingSpec(metaclass=ABCMeta):
    """VMC Sampling."""

    #: The spread magnitude of the random moves for the sampling.
    move_spread: float

    #: The seed of the pseudo-RNG used to explore the configuration space.
    rng_seed: t.Optional[int]

    #: The initial configuration of the sampling.
    ini_sys_conf: t.Optional[np.ndarray]

    #: The number of batches of the sampling.
    num_batches: int

    #: Number of steps per batch.
    num_steps_batch: int

    @abstractmethod
    def build_sampling(self, model_spec: model_base.Spec) -> \
            vmc_base.Sampling:
        """"""
        pass


class WFOptimizationSpec(metaclass=ABCMeta):
    """Wave function optimization."""
    pass


@attr.s(auto_attribs=True, frozen=True)
class DMCESResult:
    """Result of the DMC estimator sampling."""

    #: The last state of the sampling.
    state: dmc_base.State

    #: The data generated during the sampling.
    data: t.Optional[DMCESData] = None

    #: The sampling object used to generate the results.
    sampling: t.Optional[dmc_base.EstSampling] = None


class SSFEstSpec(metaclass=ABCMeta):
    """Structure factor estimator basic config."""
    num_modes: int
    as_pure_est: bool
    pfw_num_time_steps: t.Optional[int]


class DMCIniSysConfSetError(ValueError):
    """Indicates an invalid ``ini_sys_conf_set`` in a ``DMCSamplingSpec``."""
    pass


class DMCSamplingSpec(metaclass=ABCMeta):
    """DMC sampling."""

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

    #: The initial configuration set of the sampling.
    ini_sys_conf_set: t.Optional[np.ndarray]

    #: The initial energy of reference.
    ini_ref_energy: t.Optional[float]

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

    @property
    def should_eval_ssf(self):
        """"""
        return False if self.ssf_spec is None else True

    @abstractmethod
    def build_sampling(self, model_spec: model_base.Spec) -> \
            dmc_base.EstSampling:
        """"""
        pass

    def checkpoint(self):
        """"""
        pass


class DMC(metaclass=ABCMeta):
    """Class to realize a whole DMC calculation."""

    #:
    model_spec: model_base.Spec

    #:
    dmc_spec: DMCSamplingSpec

    #:
    vmc_spec: t.Optional[VMCSamplingSpec]

    #:
    wf_opt_spec: t.Optional[WFOptimizationSpec]

    #:
    output_file: t.Optional[str]

    #:
    skip_optimize: bool

    #:
    verbose: bool

    @property
    def should_exec_vmc(self):
        """"""
        return False if self.vmc_spec is None else True

    @property
    def should_optimize(self):
        """"""
        if self.skip_optimize:
            return False
        return False if self.wf_opt_spec is None else True

    @property
    @abstractmethod
    def dmc_sampling(self) -> dmc_base.EstSampling:
        pass

    @property
    @abstractmethod
    def vmc_sampling(self) -> vmc_base.Sampling:
        pass

    def exec_vmc(self):
        """

        :return:
        """
        vmc_spec = self.vmc_spec
        num_batches = vmc_spec.num_batches
        num_steps_batch = vmc_spec.num_steps_batch

        exec_logger.info('Starting VMC sampling...')
        exec_logger.info(f'Sampling {num_batches} batches of steps...')
        exec_logger.info(f'Sampling {num_steps_batch} steps per batch...')

        # New sampling instance
        sampling = self.vmc_sampling
        batches = sampling.batches(num_steps_batch, vmc_spec.ini_sys_conf)

        # By default burn-in all but the last batch.
        burn_in_batches = num_batches - 1
        if burn_in_batches:
            exec_logger.info('Executing burn-in stage...')

            exec_logger.info(f'A total of {burn_in_batches} batches will be '
                             f'discarded.')

            # Burn-in stage.
            pgs_bar = tqdm.tqdm(total=burn_in_batches, dynamic_ncols=True)
            with pgs_bar:
                for _ in islice(batches, burn_in_batches):
                    # Burn batches...
                    pgs_bar.update(1)

            exec_logger.info('Burn-in stage completed.')

        else:

            exec_logger.info(f'No burn-in batches requested.')

        # *** *** ***

        exec_logger.info('Sampling the last batch...')

        # Get the last batch.
        last_batch: vmc_base.SamplingBatch = next(batches)

        exec_logger.info('VMC Sampling completed.')

        # TODO: Should we return the sampling object?
        return last_batch, sampling

    def exec_dmc(self):
        """

        :return:
        """
        energy_field = dmc_base.IterProp.ENERGY
        weight_field = dmc_base.IterProp.WEIGHT
        num_walkers_field = dmc_base.IterProp.NUM_WALKERS
        ref_energy_field = dmc_base.IterProp.REF_ENERGY
        accum_energy_field = dmc_base.IterProp.ACCUM_ENERGY

        dmc_spec = self.dmc_spec
        num_batches = dmc_spec.num_batches
        num_time_steps_batch = dmc_spec.num_time_steps_batch
        target_num_walkers = dmc_spec.target_num_walkers
        burn_in_batches = dmc_spec.burn_in_batches
        keep_iter_data = dmc_spec.keep_iter_data

        # Alias 😐.
        nts_batch = num_time_steps_batch

        # Structure factor configuration.
        ssf_spec = dmc_spec.ssf_spec
        should_eval_ssf = dmc_spec.should_eval_ssf

        if dmc_spec.ini_sys_conf_set is None:
            raise DMCIniSysConfSetError('the initial system configuration '
                                        'is undefined')

        #
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

        sampling = self.dmc_sampling

        # The estimator sampling iterator.
        ini_state = \
            sampling.build_state(dmc_spec.ini_sys_conf_set,
                                 dmc_spec.ini_ref_energy)

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
            num_modes = ssf_spec.num_modes

            if keep_iter_data:
                ssf_shape = num_batches, nts_batch, num_modes
            else:
                ssf_shape = num_batches, num_modes

            # S(k) batch.
            ssf_blocks_data = np.zeros(ssf_shape, dtype=np.float64)

        else:
            ssf_blocks_data = None

        # Reduction factor for pure estimators.
        pure_est_reduce_factor = np.ones(num_batches, dtype=np.float64)

        # The effective batches to calculate the estimators.
        eff_batches: dmc_base.T_ESBatchesIter = \
            islice(batches_iter, num_batches)

        # Enumerated effective batches.
        enum_eff_batches: dmc_base.T_E_ESBatchesIter \
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

        data_blocks = DMCESDataBlocks(energy_blocks,
                                      weight_blocks,
                                      num_walkers_blocks,
                                      ssf_blocks)

        if keep_iter_data:
            data_series = DMCESDataSeries(props_blocks_data,
                                          ssf_blocks_data)
        else:
            data_series = None

        data = DMCESData(data_blocks, data_series)

        # NOTE: Should we return a new instance?
        # sampling = attr.evolve(sampling)

        return DMCESResult(last_state,
                           data=data,
                           sampling=sampling)

    @abstractmethod
    def run(self):
        """"""
        pass
