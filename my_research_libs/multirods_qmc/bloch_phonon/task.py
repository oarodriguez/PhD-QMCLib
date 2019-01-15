import logging
import typing as t
from itertools import islice

import attr
import numpy as np
import tqdm

import my_research_libs.qmc_base.dmc as dmc_base
from my_research_libs.qmc_base import vmc as vmc_base
from my_research_libs.qmc_data.dmc import (
    DMCESData, DMCESDataBlocks, DMCESDataSeries, EnergyBlocks,
    NumWalkersBlocks, SSFBlocks, WeightBlocks
)
from . import dmc, model, vmc

__all__ = [
    'DMC',
    'DMCSamplingSpec',
    'DMCESResult',
    'VMCSamplingSpec',
    'WFOptimizationSpec'
]

QMC_DMC_TASK_LOG_NAME = 'QMC-DMC Task'
DMC_TASK_LOG_NAME = f'DMC Sampling'
VMC_SAMPLING_LOG_NAME = 'VMC Sampling'
WF_OPTIMIZATION_LOG_NAME = 'WF Optimize'

BASIC_FORMAT = "%(asctime)-15s | %(name)-12s %(levelname)-5s: %(message)s"
logging.basicConfig(format=BASIC_FORMAT, level=logging.INFO)


@attr.s(auto_attribs=True, frozen=True)
class VMCSamplingSpec:
    """VMC Sampling."""

    #: The spread magnitude of the random moves for the sampling.
    move_spread: float

    #: The seed of the pseudo-RNG used to explore the configuration space.
    rng_seed: t.Optional[int] = None

    #: The initial configuration of the sampling.
    ini_sys_conf: t.Optional[np.ndarray] = None

    #: The number of batches of the sampling.
    num_batches: int = 64

    #: Number of steps per batch.
    num_steps_batch: int = 4096

    def run(self, model_spec: model.Spec):
        """

        :param model_spec:
        :return:
        """
        num_batches = self.num_batches
        num_steps_batch = self.num_steps_batch

        logger = logging.getLogger(VMC_SAMPLING_LOG_NAME)
        logger.setLevel(logging.INFO)

        logger.info('Starting VMC sampling...')
        logger.info(f'Sampling {num_batches} batches of steps...')
        logger.info(f'Sampling {num_steps_batch} steps per batch...')

        # New sampling instance
        sampling = vmc.Sampling(model_spec, self.move_spread, self.rng_seed)
        batches = sampling.batches(num_steps_batch, self.ini_sys_conf)

        # By default burn-in all but the last batch.
        burn_in_batches = num_batches - 1
        if burn_in_batches:
            logger.info('Executing burn-in stage...')

            logger.info(f'A total of {burn_in_batches} batches will be '
                        f'discarded.')

            # Burn-in stage.
            pgs_bar = tqdm.tqdm(total=burn_in_batches, dynamic_ncols=True)
            with pgs_bar:
                for _ in islice(batches, burn_in_batches):
                    # Burn batches...
                    pgs_bar.update(1)

            logger.info('Burn-in stage completed.')

        else:

            logger.info(f'No burn-in batches requested.')

        # *** *** ***

        logger.info('Sampling the last batch...')

        # Get the last batch.
        last_batch: vmc_base.SamplingBatch = next(batches)

        logger.info('VMC Sampling completed.')

        # TODO: Should we return the sampling object?
        return last_batch, sampling


@attr.s(auto_attribs=True, frozen=True)
class WFOptimizationSpec:
    """Wave function optimization."""

    #: The number of configurations used in the process.
    num_sys_confs: int = 1024

    #: The energy of reference to minimize the variance of the local energy.
    ref_energy: t.Optional[float] = None

    #: Use threads or multiple process.
    use_threads: bool = True

    #: Number of threads or process to use.
    num_workers: t.Optional[int] = None

    #: Display log messages or not.
    verbose: bool = False

    def run(self, model_spec: model.Spec,
            sys_conf_set: np.ndarray,
            ini_wf_abs_log_set: np.ndarray):
        """

        :param model_spec: The spec of the model.
        :param sys_conf_set: he system configurations used for the
            minimization process.
        :param ini_wf_abs_log_set: The initial wave function values. Used
            to calculate the weights.
        :return:
        """
        num_sys_confs = self.num_sys_confs

        logger = logging.getLogger(WF_OPTIMIZATION_LOG_NAME)

        logger.info('Starting wave function optimization...')
        logger.info(f'Using {num_sys_confs} configurations to '
                    f'minimize the variance...')

        sys_conf_set = sys_conf_set[-num_sys_confs:]
        ini_wf_abs_log_set = ini_wf_abs_log_set[-num_sys_confs:]

        optimizer = model.CSWFOptimizer(model_spec,
                                        sys_conf_set,
                                        ini_wf_abs_log_set,
                                        self.ref_energy,
                                        self.use_threads,
                                        self.num_workers,
                                        self.verbose)
        opt_result = optimizer.exec()

        logger.info('Wave function optimization completed.')

        return opt_result


T_OptSeqNDArray = t.Optional[t.Sequence[np.ndarray]]


@attr.s(auto_attribs=True, frozen=True)
class DMCESResult:
    """Result of the DMC estimator sampling."""

    #: The last state of the sampling.
    state: dmc_base.State

    #: The data generated during the sampling.
    data: t.Optional[DMCESData] = None

    #: The sampling object used to generate the results.
    sampling: t.Optional[dmc.EstSampling] = None


@attr.s(auto_attribs=True, frozen=True)
class SSFEstSpec:
    """Structure factor estimator basic config."""
    num_modes: int
    as_pure_est: bool = True
    pfw_num_time_steps: t.Optional[int] = None


class DMCIniSysConfSetError(ValueError):
    """Indicates an invalid ``ini_sys_conf_set`` in a ``DMCSamplingSpec``."""
    pass


@attr.s(auto_attribs=True, frozen=True)
class DMCSamplingSpec:
    """DMC sampling."""

    time_step: float
    max_num_walkers: int = 512
    target_num_walkers: int = 480
    num_walkers_control_factor: t.Optional[float] = 0.5
    rng_seed: t.Optional[int] = None

    #: The initial configuration set of the sampling.
    ini_sys_conf_set: t.Optional[np.ndarray] = None

    #: The initial energy of reference.
    ini_ref_energy: t.Optional[float] = None

    #: The number of batches of the sampling.
    num_batches: int = 512  # 2^9

    #: Number of time steps per batch.
    num_time_steps_batch: int = 512  # 2^9

    #: The number of batches to discard.
    burn_in_batches: t.Optional[int] = None

    #: Keep the estimator values for all the time steps.
    keep_iter_data: bool = False

    #: Remaining batches
    remaining_batches: t.Optional[int] = attr.ib(default=None, init=False)

    # *** Estimators configuration ***
    ssf_spec: t.Optional[SSFEstSpec] = None

    def __attrs_post_init__(self):
        """Post-initialization stage."""
        if self.burn_in_batches is None:
            burn_in_batches = max(1, self.num_batches // 8)
            object.__setattr__(self, 'burn_in_batches', burn_in_batches)

    def checkpoint(self):
        """"""
        pass

    def run(self, model_spec: model.Spec):
        """

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
        ssf_spec_task = self.ssf_spec

        # Alias 😐.
        nts_batch = num_time_steps_batch

        # Structure factor configuration.
        should_eval_ssf = ssf_spec_task is not None

        if self.ini_sys_conf_set is None:
            raise DMCIniSysConfSetError('the initial system configuration '
                                        'is undefined')

        logger = logging.getLogger(DMC_TASK_LOG_NAME)

        #
        logger.info('Starting DMC sampling...')
        logger.info(f'Using an average of {target_num_walkers} walkers.')
        logger.info(f'Sampling {num_batches} batches of time.')
        logger.info(f'Sampling {num_time_steps_batch} time-steps per batch.')

        # We will burn-in the first ten percent of the sampling chain.
        if burn_in_batches is None:
            burn_in_batches = num_batches // 8
        else:
            burn_in_batches = burn_in_batches

        if should_eval_ssf:

            ssf_spec = dmc.SSFEstSpec(model_spec,
                                      ssf_spec_task.num_modes,
                                      ssf_spec_task.as_pure_est,
                                      ssf_spec_task.pfw_num_time_steps)

        else:

            ssf_spec = None

        sampling = dmc.EstSampling(model_spec,
                                   self.time_step,
                                   self.max_num_walkers,
                                   self.target_num_walkers,
                                   self.num_walkers_control_factor,
                                   self.rng_seed,
                                   ssf_spec=ssf_spec)

        # The estimator sampling iterator.
        ini_state = \
            sampling.build_state(self.ini_sys_conf_set, self.ini_ref_energy)

        batches_iter = sampling.batches(ini_state, num_time_steps_batch)

        # Current batch data.
        batch_data = None

        if burn_in_batches:

            logger.info('Computing DMC burn-in stage...')

            logger.info(f'A total of {burn_in_batches} batches will be '
                        f'discarded.')

            # Burn-in stage.
            pgs_bar = tqdm.tqdm(total=burn_in_batches, dynamic_ncols=True)
            with pgs_bar:
                for batch_data in islice(batches_iter, burn_in_batches):
                    # Burn, burn, burn...
                    pgs_bar.update(1)

            logger.info('Burn-in stage completed.')

        else:
            logger.info(f'No burn-in batches requested')

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

        logger.info('Starting the evaluation of estimators...')

        if should_eval_ssf:
            logger.info(f'Static structure factor is going to be calculated.')
            logger.info(f'A total of {ssf_spec.num_modes} k-modes will '
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

        logger.info('Evaluation of estimators completed.')
        logger.info('DMC sampling completed.')

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


@attr.s(auto_attribs=True, frozen=True)
class DMC:
    """Class to realize a whole DMC calculation."""

    #:
    model_spec: model.Spec

    #:
    dmc_spec: DMCSamplingSpec

    #:
    vmc_spec: t.Optional[VMCSamplingSpec] = None

    #:
    wf_opt_spec: t.Optional[WFOptimizationSpec] = None

    #:
    output_file: t.Optional[str] = None

    #:
    skip_optimize: bool = False

    #:
    verbose: bool = False

    def __attrs_post_init__(self):
        """"""
        pass

    @property
    def should_optimize(self):
        """"""
        if self.skip_optimize:
            return False

        return False if self.wf_opt_spec is None else True

    def run(self):
        """

        :return:
        """
        wf_abs_log_field = vmc_base.StateProp.WF_ABS_LOG

        vmc_spec = self.vmc_spec
        wf_opt_spec = self.wf_opt_spec
        dmc_spec = self.dmc_spec
        should_optimize = self.should_optimize

        logger = logging.getLogger(QMC_DMC_TASK_LOG_NAME)
        logger.setLevel(logging.INFO)

        logger.info('Starting QMC-DMC calculation...')

        # The base DMC model spec.
        dmc_model_spec = self.model_spec

        # This reference to the current task will be modified, so the same
        # task can be executed again without realizing the VMC sampling and
        # the wave function optimization.
        self_evolve = self

        if vmc_spec is None:

            logger.info('VMC sampling task is not configured.')
            logger.info('Continue to next task...')

            if wf_opt_spec is not None:

                logger.warning("can't do WF optimization without a previous "
                               "VMC sampling task specification.")
                logging.warning("Skipping wave function optimization "
                                "task...")
                # raise TypeError("can't do WF optimization without a "
                #                 "previous VMC sampling task specification")
            else:
                logging.info("Wave function optimization task is not "
                             "configured.")
                logger.info('Continue to next task...')

        else:

            vmc_result, _ = vmc_spec.run(self.model_spec)
            sys_conf_set = vmc_result.confs
            wf_abs_log_set = vmc_result.props[wf_abs_log_field]

            # A future execution of this task won't need the realization
            # of the VMC sampling again.
            self_evolve = attr.evolve(self_evolve, vmc_spec=None)

            if should_optimize:

                logger.info('Starting VMC sampling with optimized '
                            'trial wave function...')

                # Run optimization task.
                dmc_model_spec = \
                    wf_opt_spec.run(dmc_model_spec,
                                    sys_conf_set=sys_conf_set,
                                    ini_wf_abs_log_set=wf_abs_log_set)

                vmc_result, _ = vmc_spec.run(dmc_model_spec)
                sys_conf_set = vmc_result.confs

                # In a posterior execution the same task again, we
                # may skip the optimization stage.
                self_evolve = attr.evolve(self_evolve,
                                          model_spec=dmc_model_spec,
                                          wf_opt_spec=wf_opt_spec)

            # Update the model spec of the VMC sampling, as well
            # as the initial configuration set.
            dmc_spec = \
                attr.evolve(dmc_spec, ini_sys_conf_set=sys_conf_set)

            # We have to update the DMC sampling.
            self_evolve = \
                attr.evolve(self_evolve, dmc_spec=dmc_spec)

        try:
            dmc_result = dmc_spec.run(dmc_model_spec)

        except DMCIniSysConfSetError:
            dmc_result = None
            logger.exception('The following exception occurred during the '
                             'execution of the task:')

        logger.info("All tasks finished.")

        return dmc_result, self_evolve
