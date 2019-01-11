import logging
import typing as t
from abc import ABCMeta, abstractmethod
from itertools import islice

import attr
import numpy as np
import tqdm
from cached_property import cached_property

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


class DataBlocks(metaclass=ABCMeta):
    """Abstract class to represent data in blocks."""

    num_blocks: int
    num_time_steps_block: int
    totals: np.ndarray
    weight_totals: t.Optional[np.ndarray] = None

    @classmethod
    @abstractmethod
    def from_data(cls, *args, **kwargs):
        pass

    @property
    def mean(self):
        """Mean value of the blocks."""
        energy_rbc = self.reblock
        weight_rbc = self.weight_reblock

        if weight_rbc is None:
            return self.reblock.mean
        return energy_rbc.mean / weight_rbc.mean

    @property
    def mean_error(self):
        """Error of the mean value of the blocks."""
        ow_rbc = self.reblock

        ow_mean = ow_rbc.mean
        ow_var = ow_rbc.var
        ow_eff_size = ow_rbc.eff_size
        mean = self.mean

        if self.weight_reblock is None:
            #
            w_mean = 1.
            w_var = 0.
            oww_mean = ow_mean
            w_eff_size = 0.5
            oww_eff_size = 0.5

        else:
            #
            w_rbc = self.weight_reblock
            oww_rbc = self.cross_weight_reblock

            w_mean = w_rbc.mean
            w_var = w_rbc.var
            oww_mean = oww_rbc.mean
            w_eff_size = w_rbc.eff_size
            oww_eff_size = oww_rbc.eff_size

        err_ow = ow_var / ow_mean ** 2
        err_w = w_var / w_mean ** 2
        err_oww = (oww_mean - ow_mean * w_mean) / (ow_mean * w_mean)

        return mean * np.sqrt(err_ow / ow_eff_size +
                              err_w / w_eff_size -
                              2 * err_oww / oww_eff_size)

    @property
    def reblock(self):
        """Reblocking of the totals of every block."""
        return reblock.OTFObject.from_non_obj_data(self.totals)

    @property
    def weight_reblock(self):
        """Reblocking of the totals of the weights of every block."""
        if self.weight_totals is None:
            return None
        return reblock.OTFObject.from_non_obj_data(self.weight_totals)

    @property
    def cross_weight_reblock(self):
        """Reblocking of the total * weight_total of every block."""
        totals = self.totals
        weight_totals = self.weight_totals
        if weight_totals is None:
            return None
        cross_totals = totals * weight_totals
        return reblock.OTFObject.from_non_obj_data(cross_totals)


@attr.s(auto_attribs=True, frozen=True)
class NumWalkersBlocks(DataBlocks):
    """Number of walkers data in blocks."""

    num_blocks: int
    num_time_steps_block: int
    totals: np.ndarray

    @classmethod
    def from_data(cls, num_blocks: int,
                  num_time_steps_block: int,
                  data: np.ndarray,
                  reduce_data: bool = True):
        """

        :param num_blocks:
        :param num_time_steps_block:
        :param data:
        :param reduce_data:
        :return:
        """
        weight_data = data[dmc_base.IterProp.NUM_WALKERS]
        if reduce_data:
            weight_totals = weight_data.sum(axis=1)
        else:
            weight_totals = weight_data

        return cls(num_blocks,
                   num_time_steps_block,
                   weight_totals)


@attr.s(auto_attribs=True, frozen=True)
class WeightBlocks(DataBlocks):
    """Weight data in blocks."""

    num_blocks: int
    num_time_steps_block: int
    totals: np.ndarray

    @classmethod
    def from_data(cls, num_blocks: int,
                  num_time_steps_block: int,
                  data: np.ndarray,
                  reduce_data: bool = True):
        """

        :param num_blocks:
        :param num_time_steps_block:
        :param data:
        :param reduce_data:
        :return:
        """
        weight_data = data[dmc_base.IterProp.WEIGHT]
        if reduce_data:
            weight_totals = weight_data.sum(axis=1)
        else:
            weight_totals = weight_data

        return cls(num_blocks,
                   num_time_steps_block,
                   weight_totals)


@attr.s(auto_attribs=True, frozen=True)
class EnergyBlocks(DataBlocks):
    """Energy data in blocks."""

    num_blocks: int
    num_time_steps_block: int
    totals: np.ndarray
    weight_totals: np.ndarray

    @classmethod
    def from_data(cls, num_blocks: int,
                  num_time_steps_block: int,
                  data: np.ndarray,
                  reduce_data: bool = True):
        """

        :param num_blocks:
        :param num_time_steps_block:
        :param data:
        :param reduce_data:
        :return:
        """
        energy_data = data[dmc_base.IterProp.ENERGY]
        weight_data = data[dmc_base.IterProp.WEIGHT]
        if reduce_data:
            totals = energy_data.sum(axis=1)
            weight_totals = weight_data.sum(axis=1)
        else:
            totals = energy_data
            weight_totals = weight_data

        return cls(num_blocks,
                   num_time_steps_block,
                   totals,
                   weight_totals)


@attr.s(auto_attribs=True, frozen=True)
class SFBlocks(DataBlocks):
    """Structure Factor data in blocks."""

    num_blocks: int
    num_time_steps_block: int
    totals: np.ndarray
    weight_totals: np.ndarray
    as_pure_est: bool = True  # NOTE: Maybe we do not need this.

    @classmethod
    def from_data(cls, num_blocks: int,
                  num_time_steps_block: int,
                  sf_data: np.ndarray,
                  props_data: np.ndarray,
                  reduce_data: bool = True,
                  as_pure_est: bool = True,
                  pure_est_reduce_factor: np.ndarray = None):
        """

        :param reduce_data:
        :param num_blocks:
        :param num_time_steps_block:
        :param sf_data:
        :param props_data:
        :param as_pure_est:
        :param pure_est_reduce_factor: 
        :return:
        """
        nts_block = num_time_steps_block
        weight_data = props_data[dmc_base.IterProp.WEIGHT]

        if not as_pure_est:

            if reduce_data:
                totals = sf_data.sum(axis=1)
                weight_totals = weight_data.sum(axis=1)

            else:
                totals = sf_data
                weight_totals = weight_data

        else:
            # Normalize the pure estimator.
            if reduce_data:

                # Reductions are not used in pure estimators.
                # We just take the last element.
                totals = sf_data[:, nts_block - 1, :]
                weight_totals = weight_data[:, nts_block - 1]

            else:
                totals = sf_data
                weight_totals = weight_data * pure_est_reduce_factor

        # Add an extra dimension.
        weight_totals = weight_totals[:, np.newaxis]

        return cls(num_blocks, num_time_steps_block,
                   totals, weight_totals, as_pure_est)

    @property
    def reblock(self):
        """Reblocking of the totals of every block."""
        return reblock.OTFSet.from_non_obj_data(self.totals)

    @property
    def weight_reblock(self):
        """Reblocking of the totals of the weights of every block."""
        if self.weight_totals is None:
            return None
        return reblock.OTFSet.from_non_obj_data(self.weight_totals)

    @property
    def cross_weight_reblock(self):
        """Reblocking of the total * weight_total of every block."""
        totals = self.totals
        weight_totals = self.weight_totals
        if weight_totals is None:
            return None
        cross_totals = totals * weight_totals
        return reblock.OTFSet.from_non_obj_data(cross_totals)


@attr.s(auto_attribs=True, frozen=True)
class DMCESData:
    """The data from a DMC sampling."""

    #: The blocks / batches of data.
    props_blocks: np.ndarray

    #:
    struct_factor_blocks: t.Optional[np.ndarray] = None

    @cached_property
    def props(self):
        """"""
        source_data = self.props_blocks
        return np.hstack(source_data)

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

    @cached_property
    def struct_factor(self):
        """"""
        if self.struct_factor_blocks is None:
            return None
        return np.vstack(self.struct_factor_blocks)


@attr.s(auto_attribs=True, frozen=True)
class DMCESBlocks:
    """Results of a DMC sampling grouped in block totals."""

    energy: EnergyBlocks
    weight: DataBlocks
    num_walkers: DataBlocks
    struct_factor: t.Optional[SFBlocks] = None


T_OptSeqNDArray = t.Optional[t.Sequence[np.ndarray]]


@attr.s(auto_attribs=True, frozen=True)
class DMCESResult:
    """Result of the DMC estimator sampling."""

    #: The last state of the sampling.
    state: dmc_base.State

    #: The blocked data of the sampling.
    data_blocks: DMCESBlocks

    #: The data generated during the sampling.
    data: t.Optional[DMCESData] = None

    #: The sampling object used to generate the results.
    sampling: t.Optional['DMCEstSampling'] = None


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
        """Post-initialization stage."""

        if self.rng_seed is None:
            rng_seed = int(utils.get_random_rng_seed())
            object.__setattr__(self, 'rng_seed', rng_seed)

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
        ref_energy_field = dmc_base.IterProp.REF_ENERGY
        accum_energy_field = dmc_base.IterProp.ACCUM_ENERGY

        num_batches = self.num_batches
        num_time_steps_batch = self.num_time_steps_batch
        target_num_walkers = self.target_num_walkers
        burn_in_batches = self.burn_in_batches
        keep_iter_data = self.keep_iter_data
        sf_config = self.structure_factor

        nts_batch = num_time_steps_batch

        # Structure factor configuration.
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

        if should_eval_sf:
            # The shape of the structure factor array.
            num_modes = sf_config.num_modes

            if keep_iter_data:
                sf_shape = num_batches, nts_batch, num_modes
            else:
                sf_shape = num_batches, num_modes

            # S(k) batch.
            sf_blocks_data = np.zeros(sf_shape, dtype=np.float64)

        else:
            sf_blocks_data = None

        # Reduction factor for pure estimators.
        pure_est_reduce_factor = np.ones(num_batches, dtype=np.float64)

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
                if should_eval_sf:

                    iter_sf_array = batch_data.iter_structure_factor

                    if keep_iter_data:
                        # Keep a copy of the generated data of the sampling.
                        sf_blocks_data[batch_idx] = iter_sf_array[:]

                    else:

                        if sf_config.as_pure_est:
                            # Get only the last element of the batch.
                            sf_blocks_data[batch_idx] = \
                                iter_sf_array[nts_batch - 1]

                        else:
                            # Make the reduction.
                            sf_blocks_data[batch_idx] = \
                                iter_sf_array.sum(axis=0)

                rem_batches = num_batches - batch_idx - 1
                object.__setattr__(self, 'remaining_batches', rem_batches)

                # Pick the last state
                last_state = batch_data.last_state

                # logger.info(f'Batch #{batch_idx:d} completed')
                pgs_bar.update()

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
            WeightBlocks.from_data(num_batches, nts_batch,
                                   props_blocks_data, reduce_data)

        if should_eval_sf:

            struct_factor_blocks = \
                SFBlocks.from_data(num_batches, nts_batch,
                                   sf_blocks_data,
                                   props_blocks_data, reduce_data,
                                   sf_config.as_pure_est,
                                   pure_est_reduce_factor)

        else:
            struct_factor_blocks = None

        data_blocks = DMCESBlocks(energy_blocks,
                                  weight_blocks,
                                  num_walkers_blocks,
                                  struct_factor_blocks)

        if keep_iter_data:
            data_series = DMCESData(props_blocks_data,
                                    sf_blocks_data)
        else:
            data_series = None

        return DMCESResult(last_state,
                           data_blocks,
                           data_series,
                           sampling=self)


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
