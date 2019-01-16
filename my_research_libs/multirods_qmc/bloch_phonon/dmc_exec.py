import typing as t

import attr
import numpy as np

from my_research_libs.qmc_base import vmc as vmc_base
from my_research_libs.qmc_exec import dmc as dmc_exec, exec_logger
from my_research_libs.qmc_exec.dmc import (
    DMCProcInput, ProcExecutorResult, VMCProcInput
)
from . import dmc, model, vmc

__all__ = [
    'DMCProcSpec',
    'ProcExecutor',
    'SSFEstSpec',
    'VMCProcSpec',
    'WFOptProcSpec'
]


@attr.s(auto_attribs=True, frozen=True)
class VMCProcSpec(dmc_exec.VMCProcSpec):
    """VMC Sampling."""

    move_spread: float

    rng_seed: t.Optional[int] = None

    num_batches: int = 64

    num_steps_batch: int = 4096

    def build_sampling(self, model_spec: model.Spec) -> vmc.Sampling:
        """

        :param model_spec:
        :return:
        """
        return vmc.Sampling(model_spec, self.move_spread, self.rng_seed)


@attr.s(auto_attribs=True, frozen=True)
class WFOptProcSpec(dmc_exec.WFOptProcSpec):
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


T_OptSeqNDArray = t.Optional[t.Sequence[np.ndarray]]


@attr.s(auto_attribs=True, frozen=True)
class SSFEstSpec(dmc_exec.SSFEstSpec):
    """Structure factor estimator basic config."""
    num_modes: int
    as_pure_est: bool = True
    pfw_num_time_steps: t.Optional[int] = None


@attr.s(auto_attribs=True, frozen=True)
class DMCProcSpec(dmc_exec.DMCProcSpec):
    """DMC sampling."""

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

    def build_sampling(self, model_spec: model.Spec) -> dmc.EstSampling:
        """

        :param model_spec:
        :return:
        """
        if self.should_eval_ssf:
            ssf_spec = self.ssf_spec
            ssf_est = dmc.SSFEstSpec(model_spec, ssf_spec.num_modes,
                                     ssf_spec.as_pure_est,
                                     ssf_spec.pfw_num_time_steps)

        else:
            ssf_est = None

        sampling = dmc.EstSampling(model_spec, self.time_step,
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
class ProcExecutor(dmc_exec.ProcExecutor):
    """Class to realize a whole DMC calculation."""

    model_spec: model.Spec

    dmc_proc_spec: DMCProcSpec

    vmc_proc_spec: t.Optional[VMCProcSpec] = None

    wf_opt_proc_spec: t.Optional[WFOptProcSpec] = None

    output_file: t.Optional[str] = None

    skip_wf_opt_proc: bool = False

    verbose: bool = False

    def __attrs_post_init__(self):
        """"""
        pass

    @property
    def dmc_sampling(self) -> dmc.EstSampling:
        """

        :return:
        """
        return self.dmc_proc_spec.build_sampling(self.model_spec)

    @property
    def vmc_sampling(self) -> t.Union[vmc.Sampling, None]:
        """

        :return:
        """
        vmc_spec = self.vmc_proc_spec
        if vmc_spec is None:
            return None
        return vmc_spec.build_sampling(self.model_spec)

    def exec_wf_opt_proc(self, sys_conf_set: np.ndarray,
                         ini_wf_abs_log_set: np.ndarray):
        """

        :param sys_conf_set: he system configurations used for the
            minimization process.
        :param ini_wf_abs_log_set: The initial wave function values. Used
            to calculate the weights.
        :return:
        """
        wf_opt_proc_spec = self.wf_opt_proc_spec
        num_sys_confs = wf_opt_proc_spec.num_sys_confs

        exec_logger.info('Starting wave function optimization...')
        exec_logger.info(f'Using {num_sys_confs} configurations to '
                         f'minimize the variance...')

        sys_conf_set = sys_conf_set[-num_sys_confs:]
        ini_wf_abs_log_set = ini_wf_abs_log_set[-num_sys_confs:]

        optimizer = model.CSWFOptimizer(self.model_spec,
                                        sys_conf_set,
                                        ini_wf_abs_log_set,
                                        wf_opt_proc_spec.ref_energy,
                                        wf_opt_proc_spec.use_threads,
                                        wf_opt_proc_spec.num_workers,
                                        wf_opt_proc_spec.verbose)
        opt_result = optimizer.exec()

        exec_logger.info('Wave function optimization completed.')

        return opt_result

    def build_proc_input(self, maybe_sys_conf_set: np.ndarray,
                         ref_energy: float = None):
        """

        :param maybe_sys_conf_set:
        :param ref_energy:
        :return:
        """
        maybe_sys_conf_set = np.asarray(maybe_sys_conf_set)
        if self.should_exec_vmc:
            proc_input = \
                VMCProcInput.from_sys_conf(maybe_sys_conf_set, self)

        else:
            proc_input = \
                DMCProcInput.from_sys_conf_set(maybe_sys_conf_set, self,
                                               ref_energy)

        return proc_input

    def exec(self, proc_input: dmc_exec.T_ProcDirectorInput):
        """

        :return:
        """
        wf_abs_log_field = vmc_base.StateProp.WF_ABS_LOG

        should_exec_vmc = self.should_exec_vmc
        should_optimize = self.should_optimize

        exec_logger.info('Starting QMC-DMC calculation...')

        # This reference to the current task will be modified, so the same
        # task can be executed again without realizing the VMC sampling and
        # the wave function optimization.
        self_evolve = self
        dmc_proc_input = proc_input

        if not should_exec_vmc:

            exec_logger.info('VMC sampling task is not configured.')
            exec_logger.info('Continue to next task.')

            if should_optimize:
                raise TypeError("can't do WF optimization without a "
                                "previous VMC sampling task specification")

        else:

            vmc_result, _ = self.exec_vmc_proc(proc_input)
            sys_conf_set = vmc_result.confs
            wf_abs_log_set = vmc_result.props[wf_abs_log_field]

            # A future execution of this task won't need the realization
            # of the VMC sampling again.
            self_evolve = attr.evolve(self_evolve, vmc_proc_spec=None)

            if should_optimize:

                exec_logger.info('Starting VMC sampling with optimized '
                                 'trial wave function...')

                # Run optimization task.
                dmc_model_spec = \
                    self.exec_wf_opt_proc(sys_conf_set=sys_conf_set,
                                          ini_wf_abs_log_set=wf_abs_log_set)

                # Use the same build_proc_input.
                vmc_result, _ = self.exec_vmc_proc(proc_input)
                sys_conf_set = vmc_result.confs

                # In a posterior execution the same task again, we
                # may skip the optimization stage.
                self_evolve = attr.evolve(self_evolve,
                                          model_spec=dmc_model_spec,
                                          wf_opt_proc_spec=None)

            else:

                exec_logger.info("Wave function optimization task is not "
                                 "configured.")
                exec_logger.info('Continue to next task.')

                # In a posterior execution the same task again, we
                # may skip the optimization stage.
                self_evolve = attr.evolve(self_evolve, wf_opt_proc_spec=None)

            # Update the model spec of the VMC sampling, as well
            # as the initial configuration set.
            dmc_proc_input = \
                dmc_exec.DMCProcInput.from_sys_conf_set(sys_conf_set,
                                                        proc_director=self)

        # Execute the main task.
        dmc_proc_result = self_evolve.exec_dmc_proc(dmc_proc_input)

        exec_logger.info("All tasks finished.")

        return ProcExecutorResult(dmc_proc_result, self_evolve)
