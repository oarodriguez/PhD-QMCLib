import logging
import typing as t

import attr
import numpy as np

from my_research_libs.qmc_base import vmc as vmc_base
from my_research_libs.qmc_exec import dmc as dmc_exec, exec_logger
from . import dmc, model, vmc

__all__ = [
    'DMC',
    'WFOptimizationSpec'
]


@attr.s(auto_attribs=True, frozen=True)
class VMCSamplingSpec(dmc_exec.VMCSamplingSpec):
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

    def build_sampling(self, model_spec: model.Spec) -> vmc.Sampling:
        """

        :param model_spec:
        :return:
        """
        return vmc.Sampling(model_spec,
                            self.move_spread,
                            self.rng_seed)


@attr.s(auto_attribs=True, frozen=True)
class WFOptimizationSpec(dmc_exec.WFOptimizationSpec):
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

        exec_logger.info('Starting wave function optimization...')
        exec_logger.info(f'Using {num_sys_confs} configurations to '
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

        exec_logger.info('Wave function optimization completed.')

        return opt_result


T_OptSeqNDArray = t.Optional[t.Sequence[np.ndarray]]


@attr.s(auto_attribs=True, frozen=True)
class SSFEstSpec(dmc_exec.SSFEstSpec):
    """Structure factor estimator basic config."""
    num_modes: int
    as_pure_est: bool = True
    pfw_num_time_steps: t.Optional[int] = None


@attr.s(auto_attribs=True, frozen=True)
class DMCSamplingSpec(dmc_exec.DMCSamplingSpec):
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

    def build_sampling(self, model_spec: model.Spec) -> dmc.EstSampling:
        """

        :param model_spec:
        :return:
        """
        if self.should_eval_ssf:
            ssf_spec = self.ssf_spec
            ssf_est = dmc.SSFEstSpec(model_spec,
                                     ssf_spec.num_modes,
                                     ssf_spec.as_pure_est,
                                     ssf_spec.pfw_num_time_steps)

        else:
            ssf_est = None

        sampling = dmc.EstSampling(model_spec,
                                   self.time_step,
                                   self.max_num_walkers,
                                   self.target_num_walkers,
                                   self.num_walkers_control_factor,
                                   self.rng_seed,
                                   ssf_spec=ssf_est)
        return sampling

    def checkpoint(self):
        """"""
        pass


@attr.s(auto_attribs=True, frozen=True)
class DMC(dmc_exec.DMC):
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

    def run(self):
        """

        :return:
        """
        wf_abs_log_field = vmc_base.StateProp.WF_ABS_LOG

        vmc_spec = self.vmc_spec
        wf_opt_spec = self.wf_opt_spec
        dmc_spec = self.dmc_spec
        should_optimize = self.should_optimize

        exec_logger.info('Starting QMC-DMC calculation...')

        # The base DMC model spec.
        dmc_model_spec = self.model_spec

        # This reference to the current task will be modified, so the same
        # task can be executed again without realizing the VMC sampling and
        # the wave function optimization.
        self_evolve = self

        if vmc_spec is None:

            exec_logger.info('VMC sampling task is not configured.')
            exec_logger.info('Continue to next task...')

            if wf_opt_spec is not None:

                exec_logger.warning("can't do WF optimization without a "
                                    "previous VMC sampling task "
                                    "specification.")
                logging.warning("Skipping wave function optimization "
                                "task...")
                # raise TypeError("can't do WF optimization without a "
                #                 "previous VMC sampling task specification")
            else:
                logging.info("Wave function optimization task is not "
                             "configured.")
                exec_logger.info('Continue to next task...')

        else:

            vmc_result, _ = vmc_spec.run(self.model_spec)
            sys_conf_set = vmc_result.confs
            wf_abs_log_set = vmc_result.props[wf_abs_log_field]

            # A future execution of this task won't need the realization
            # of the VMC sampling again.
            self_evolve = attr.evolve(self_evolve, vmc_spec=None)

            if should_optimize:

                exec_logger.info('Starting VMC sampling with optimized '
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

        except dmc_exec.DMCIniSysConfSetError:
            dmc_result = None
            exec_logger.exception('The following exception occurred '
                                  'during the execution of the task:')

        exec_logger.info("All tasks finished.")

        return dmc_result, self_evolve
