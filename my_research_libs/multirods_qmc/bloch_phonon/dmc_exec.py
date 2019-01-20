import typing as t

import attr
import numpy as np
from cached_property import cached_property

from my_research_libs.qmc_exec import dmc as dmc_exec, exec_logger
from . import dmc, model, vmc

__all__ = [
    'DMCProc',
    'SSFEstSpec',
    'VMCProc',
    'WFOptProc'
]


@attr.s(auto_attribs=True, frozen=True)
class VMCProc(dmc_exec.VMCProc):
    """VMC Sampling."""

    model_spec: model.Spec

    move_spread: float

    rng_seed: t.Optional[int] = None

    num_batches: int = 64

    num_steps_batch: int = 4096

    @cached_property
    def sampling(self) -> vmc.Sampling:
        """

        :return:
        """
        return vmc.Sampling(self.model_spec,
                            self.move_spread,
                            self.rng_seed)


@attr.s(auto_attribs=True, frozen=True)
class WFOptProc(dmc_exec.WFOptProc):
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

    def exec(self, model_spec: model.Spec,
             sys_conf_set: np.ndarray,
             ini_wf_abs_log_set: np.ndarray):
        """

        :param model_spec:
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


@attr.s(auto_attribs=True, frozen=True)
class SSFEstSpec(dmc_exec.SSFEstSpec):
    """Structure factor estimator basic config."""
    num_modes: int
    as_pure_est: bool = True
    pfw_num_time_steps: t.Optional[int] = None


@attr.s(auto_attribs=True, frozen=True)
class DMCProc(dmc_exec.DMCProc):
    """DMC sampling."""

    model_spec: model.Spec

    time_step: float

    max_num_walkers: int = 512

    target_num_walkers: int = 480

    num_walkers_control_factor: t.Optional[float] = 0.5

    rng_seed: t.Optional[int] = None

    num_batches: int = 512  # 2^9

    num_time_steps_batch: int = 512  # 2^9

    burn_in_batches: t.Optional[int] = None

    keep_iter_data: bool = False

    #: Remaining batches
    remaining_batches: t.Optional[int] = attr.ib(default=None, init=False)

    # *** Estimators configuration ***
    ssf_spec: t.Optional[SSFEstSpec] = None

    @cached_property
    def sampling(self) -> dmc.EstSampling:
        """

        :return:
        """
        if self.should_eval_ssf:
            ssf_spec = self.ssf_spec
            ssf_est = dmc.SSFEstSpec(self.model_spec,
                                     ssf_spec.num_modes,
                                     ssf_spec.as_pure_est,
                                     ssf_spec.pfw_num_time_steps)

        else:
            ssf_est = None

        sampling = dmc.EstSampling(self.model_spec,
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
