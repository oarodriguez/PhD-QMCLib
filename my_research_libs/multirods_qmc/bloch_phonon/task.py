import logging
import typing as t
from itertools import islice

import attr
import numpy as np
import tqdm

import my_research_libs.qmc_base.dmc as dmc_base
from my_research_libs import utils
from my_research_libs.qmc_base import vmc as vmc_base
from my_research_libs.stats import reblock
from . import dmc, model, vmc

__all__ = [
    'DMC',
    'DMCEstSampling',
    'VMCSampling',
    'DMCESResult',
    'WFOptimization'
]

QMC_DMC_TASK_LOG_NAME = 'QMC-DMC Task'
DMC_TASK_LOG_NAME = f'DMC Sampling'
VMC_SAMPLING_LOG_NAME = 'VMC Sampling'
WF_OPTIMIZATION_LOG_NAME = 'WF Optimize'

BASIC_FORMAT = "%(asctime)-15s | %(name)-12s %(levelname)-5s: %(message)s"
logging.basicConfig(format=BASIC_FORMAT, level=logging.INFO)


@attr.s(auto_attribs=True, frozen=True)
class VMCSampling(vmc.Sampling):
    """VMC Sampling."""

    model_spec: model.Spec
    move_spread: float
    rng_seed: t.Optional[int] = None

    #: The initial configuration of the sampling.
    ini_sys_conf: t.Optional[np.ndarray] = None

    #: The number of batches of the sampling.
    num_batches: int = 64

    #: Number of steps per batch.
    num_steps_batch: int = 4096

    def __attrs_post_init__(self):
        """"""
        if self.rng_seed is None:
            rng_seed = int(utils.get_random_rng_seed())
            object.__setattr__(self, 'rng_seed', rng_seed)

    def run(self):
        """

        :return:
        """
        num_batches = self.num_batches
        num_steps_batch = self.num_steps_batch

        logger = logging.getLogger(VMC_SAMPLING_LOG_NAME)
        logger.setLevel(logging.INFO)

        logger.info('Starting VMC sampling...')
        logger.info(f'Sampling {num_batches} batches of steps...')
        logger.info(f'Sampling {num_steps_batch} steps per batch...')

        batches = self.batches(num_steps_batch, self.ini_sys_conf)

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

        return last_batch


@attr.s(auto_attribs=True, frozen=True)
class WFOptimization:
    """Wave function optimization."""

    #: The spec of the model.
    model_spec: model.Spec

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

    def run(self, sys_conf_set: np.ndarray,
            ini_wf_abs_log_set: np.ndarray):
        """

        :param sys_conf_set: he system configurations used for the
            minimization process.
        :param ini_wf_abs_log_set: The initial wave function values. Used
            to calculate the weights.
        :return:
        """
        num_sys_confs = self.num_sys_confs

        logger = logging.getLogger(WF_OPTIMIZATION_LOG_NAME)

        logger.info('Starting wave function optimization...')
        logger.info(f'Using { num_sys_confs } configurations to '
                    f'minimize the variance...')

        sys_conf_set = sys_conf_set[-num_sys_confs:]
        ini_wf_abs_log_set = ini_wf_abs_log_set[-num_sys_confs:]

        optimizer = model.CSWFOptimizer(self.model_spec,
                                        sys_conf_set,
                                        ini_wf_abs_log_set,
                                        self.ref_energy,
                                        self.use_threads,
                                        self.num_workers,
                                        self.verbose)
        opt_result = optimizer.exec()

        logger.info('Wave function optimization completed.')

        return opt_result


class DMCEstDataSeries(t.NamedTuple):
    """"""
    props: t.List[np.ndarray]
    structure_factor: t.List[np.ndarray]


T_OptSeqNDArray = t.Optional[t.Sequence[np.ndarray]]


@attr.s(auto_attribs=True, frozen=True)
class DMCESResult:
    """Result of the DMC sampling."""

    #:
    state: dmc_base.State

    #:
    data_series: DMCEstDataSeries

    @property
    def props(self):
        """"""
        return np.hstack(self.data_series.props)

    @property
    def energy(self):
        """"""
        return self.props[dmc_base.IterProp.ENERGY]

    @property
    def weight(self):
        """"""
        return self.props[dmc_base.IterProp.WEIGHT]

    @property
    def num_walkers(self):
        """"""
        return self.props[dmc_base.IterProp.NUM_WALKERS]

    @property
    def ref_energy(self):
        """"""
        return self.props[dmc_base.IterProp.REF_ENERGY]

    @property
    def accum_energy(self):
        """"""
        return self.props[dmc_base.IterProp.ACCUM_ENERGY]

    @property
    def structure_factor(self):
        """"""
        return np.vstack(self.data_series.structure_factor)


@attr.s(auto_attribs=True, frozen=True)
class DMCEstSampling(dmc.EstSampling):
    """DMC sampling."""

    #: Estimator sampling object
    model_spec: model.Spec
    time_step: float
    max_num_walkers: int = 512
    target_num_walkers: int = 480
    num_walkers_control_factor: t.Optional[float] = 0.5
    rng_seed: t.Optional[int] = None

    # *** Estimators configuration ***
    structure_factor: t.Optional[dmc.StructureFactorEst] = None

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

    def __attrs_post_init__(self):
        """"""
        super().__attrs_post_init__()
        # Only take as much sys_conf items as target_num_walkers.
        ini_sys_conf_set = self.ini_sys_conf_set
        if ini_sys_conf_set is not None:
            ini_sys_conf_set = ini_sys_conf_set[-self.target_num_walkers:]
            object.__setattr__(self, 'ini_sys_conf_set', ini_sys_conf_set)

        if self.burn_in_batches is None:
            burn_in_batches = max(1, self.num_batches // 8)
            object.__setattr__(self, 'burn_in_batches', burn_in_batches)

    def checkpoint(self):
        """"""
        pass

    def run(self):
        """

        :return:
        """
        energy_field = dmc_base.IterProp.ENERGY
        weight_field = dmc_base.IterProp.WEIGHT
        num_walkers_field = dmc_base.IterProp.NUM_WALKERS

        num_batches = self.num_batches
        num_time_steps_batch = self.num_time_steps_batch
        target_num_walkers = self.target_num_walkers
        burn_in_batches = self.burn_in_batches
        keep_iter_data = self.keep_iter_data

        # Structure factor configuration.
        sf_config = self.structure_factor
        should_eval_sf = sf_config is not None

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

        # The estimator sampling iterator.
        batches_iter = self.batches(num_time_steps_batch,
                                    self.ini_sys_conf_set,
                                    self.ini_ref_energy)

        if burn_in_batches:
            logger.info('Computing DMC burn-in stage...')

            logger.info(f'A total of {burn_in_batches} batches will be '
                        f'discarded.')

            # Burn-in stage.
            pgs_bar = tqdm.tqdm(total=burn_in_batches, dynamic_ncols=True)
            with pgs_bar:
                for _ in islice(batches_iter, burn_in_batches):
                    # Burn, burn, burn...
                    pgs_bar.update(1)

            logger.info('Burn-in stage completed.')

        else:
            logger.info(f'No burn-in batches requested')

        # Main containers of data.
        props_batches_data = []
        sf_batches_data = []
        energy_reblock_set = []
        weight_reblock_set = []
        ew_reblock_set = []
        num_walkers_reblock_set = []
        sf_reblock_set = []

        # The effective batches to calculate the estimators.
        eff_batches: dmc_base.T_ESBatchesIter = \
            islice(batches_iter, num_batches)

        # Enumerated effective batches.
        enum_eff_batches: dmc_base.T_E_ESBatchesIter \
            = enumerate(eff_batches)

        # Reference to the last DMC state.
        last_state = None

        logger.info('Starting the evaluation of estimators...')

        if should_eval_sf:
            logger.info(f'Structure factor is going to be calculated.')
            logger.info(f'A total of {sf_config.num_modes} k-modes will '
                        f'be used as input for S(k).')

        with tqdm.tqdm(total=num_batches, dynamic_ncols=True) as pgs_bar:

            for batch_idx, batch_data in enum_eff_batches:

                iter_props = batch_data.iter_props
                energies = iter_props[energy_field]
                weights = iter_props[weight_field]
                num_walkers = iter_props[num_walkers_field]

                # NOTE: Should we return the iter_props by default? 🤔🤔🤔
                #  If we keep the whole sampling data, i.e.,
                #  ``keep_iter_data`` is True, then it has not much sense to
                #  realize the reblocking for each batch of data.
                #  Let's think about it...

                if keep_iter_data:
                    # Store the information of the DMC sampling.
                    iter_props = iter_props.copy()
                    props_batches_data.append(iter_props)

                else:
                    # Just get reblocking objects.
                    energy_reblock = reblock.on_the_fly_obj_create(energies)
                    weight_reblock = reblock.on_the_fly_obj_create(weights)
                    ew_reblock = \
                        reblock.on_the_fly_obj_create(energies * weights)
                    num_walkers_reblock = \
                        reblock.on_the_fly_obj_create(num_walkers)

                    energy_reblock_set.append(energy_reblock)
                    weight_reblock_set.append(weight_reblock)
                    ew_reblock_set.append(ew_reblock)
                    num_walkers_reblock_set.append(num_walkers_reblock)

                # Handling the structure factor results.
                iter_sf_array = batch_data.iter_structure_factor

                if should_eval_sf:

                    if keep_iter_data:

                        # Keep a copy of the generated data of the sampling.

                        if sf_config.as_pure_est:
                            iter_sf_array = iter_sf_array.copy()

                            # Normalize the pure estimator.
                            sf_pure_est = \
                                iter_sf_array / num_walkers[:, np.newaxis]

                            sf_batches_data.append(sf_pure_est)

                        else:
                            iter_sf_array = iter_sf_array.copy()
                            sf_batches_data.append(iter_sf_array)

                    else:

                        # Do not keep a copy of the generated data of the
                        # sampling. Instead, return a reblocking object
                        # to calculate the average values and errors.

                        if sf_config.as_pure_est:
                            iter_sf_array_last = iter_sf_array[-1].copy()

                            # Normalize the pure estimator.
                            sf_pure_est = \
                                iter_sf_array_last / num_walkers[-1]

                            sf_reblock_set.append(sf_pure_est)

                        else:
                            sf_reblock = \
                                reblock.on_the_fly_obj_create(iter_sf_array)

                            sf_reblock_set.append(sf_reblock)

                rem_batches = num_batches - batch_idx - 1
                object.__setattr__(self, 'remaining_batches', rem_batches)

                # Pick the last state
                last_state = batch_data.last_state

                # logger.info(f'Batch #{batch_idx:d} completed')
                pgs_bar.update()

        logger.info('Evaluation of estimators completed.')

        logger.info('DMC sampling completed.')

        data_series = DMCEstDataSeries(props_batches_data,
                                       sf_batches_data)
        return DMCESResult(last_state, data_series)




@attr.s(auto_attribs=True, frozen=True)
class DMC:
    """Class to realize a whole DMC calculation."""

    #:
    dmc_est_sampling: DMCEstSampling

    #:
    vmc_sampling: t.Optional[VMCSampling] = None

    #:
    wf_optimize: t.Optional[WFOptimization] = None

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

        return False if self.wf_optimize is None else True

    def run(self):
        """

        :return:
        """
        wf_abs_log_field = vmc_base.StateProp.WF_ABS_LOG

        vmc_sampling = self.vmc_sampling
        wf_optimize = self.wf_optimize
        dmc_est_sampling = self.dmc_est_sampling
        should_optimize = self.should_optimize

        logger = logging.getLogger(QMC_DMC_TASK_LOG_NAME)
        logger.setLevel(logging.INFO)

        logger.info('Starting QMC-DMC calculation...')

        if vmc_sampling is None:

            logger.info('VMC sampling task is not configured.')
            logger.info('Continue to next task...')

            if wf_optimize is not None:

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

            vmc_result = vmc_sampling.run()
            sys_conf_set = vmc_result.confs
            wf_abs_log_set = vmc_result.props[wf_abs_log_field]

            if should_optimize:

                logger.info('Starting VMC sampling with optimized '
                            'trial wave function...')

                # Run optimization task.
                dmc_model_spec = \
                    wf_optimize.run(sys_conf_set=sys_conf_set,
                                    ini_wf_abs_log_set=wf_abs_log_set)

                vmc_sampling: VMCSampling = \
                    attr.evolve(vmc_sampling, model_spec=dmc_model_spec)

                vmc_result = vmc_sampling.run()
                sys_conf_set = vmc_result.confs

                # Update the model spec of the VMC sampling, as well
                # as the initial configuration set.
                dmc_est_sampling = attr.evolve(dmc_est_sampling,
                                               model_spec=dmc_model_spec,
                                               ini_sys_conf_set=sys_conf_set)

            else:

                dmc_est_sampling = attr.evolve(dmc_est_sampling,
                                               ini_sys_conf_set=sys_conf_set)

        dmc_result = dmc_est_sampling.run()

        logger.info("All tasks finished.")

        return dmc_result
