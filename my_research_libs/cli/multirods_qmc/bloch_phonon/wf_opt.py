import typing as t

import attr
import numpy as np

from my_research_libs.multirods_qmc.bloch_phonon import model
from my_research_libs.qmc_exec import exec_logger, wf_opt as wf_opt_exec
from my_research_libs.util.attr import (
    bool_validator, int_validator, opt_float_converter, opt_int_validator
)


@attr.s(auto_attribs=True, frozen=True)
class WFOptProc(wf_opt_exec.WFOptProc):
    """Wave function optimization."""

    #: The number of configurations used in the process.
    num_sys_confs: int = \
        attr.ib(default=1024, validator=int_validator)

    #: The energy of reference to minimize the variance of the local energy.
    ref_energy: t.Optional[float] = \
        attr.ib(default=None, converter=opt_float_converter)

    #: Use threads or multiple process.
    use_threads: bool = attr.ib(default=True, validator=bool_validator)

    #: Number of threads or process to use.
    num_workers: t.Optional[int] = \
        attr.ib(default=None, validator=opt_int_validator)

    #: Display log messages or not.
    verbose: bool = attr.ib(default=False, validator=bool_validator)

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


wf_opt_proc_validator = attr.validators.instance_of(WFOptProc)
opt_wf_opt_proc_validator = attr.validators.optional(wf_opt_proc_validator)
