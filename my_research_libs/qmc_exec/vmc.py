import typing as t
from abc import ABCMeta, abstractmethod
from itertools import islice

import attr
import numpy as np
import tqdm

from my_research_libs.qmc_base import model as model_base, vmc as vmc_base
from .logging import exec_logger


@attr.s(auto_attribs=True)
class VMCProcInput(metaclass=ABCMeta):
    """Represents the input for the VMC calculation procedure."""
    # The state of the VMC procedure input.
    state: vmc_base.State


class VMCProc(metaclass=ABCMeta):
    """VMC Sampling procedure spec."""

    #: The model spec.
    model_spec: model_base.Spec

    #: The spread magnitude of the random moves for the sampling.
    move_spread: float

    #: The seed of the pseudo-RNG used to explore the configuration space.
    rng_seed: t.Optional[int]

    #: The number of batches of the sampling.
    num_batches: int

    #: Number of steps per batch.
    num_steps_batch: int

    @property
    @abstractmethod
    def sampling(self) -> vmc_base.Sampling:
        pass

    def build_input(self, sys_conf: np.ndarray):
        """

        :param sys_conf:
        :return:
        """
        vmc_sampling = self.sampling
        state = vmc_sampling.build_state(sys_conf)
        return VMCProcInput(state)

    def exec(self, proc_input: VMCProcInput):
        """

        :param proc_input:
        :return:
        """
        num_batches = self.num_batches
        num_steps_batch = self.num_steps_batch

        exec_logger.info('Starting VMC sampling...')
        exec_logger.info(f'Sampling {num_batches} batches of steps...')
        exec_logger.info(f'Sampling {num_steps_batch} steps per batch...')

        # New sampling instance
        sampling = self.sampling
        if not isinstance(proc_input, VMCProcInput):
            raise VMCProcInputError('the input data for the VMC procedure is '
                                    'not valid')
        ini_sys_conf = proc_input.state.sys_conf
        batches = sampling.batches(num_steps_batch, ini_sys_conf)

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


class VMCProcInputError(ValueError):
    """Flags an invalid input for a VMC calculation procedure."""
    pass
