from abc import ABCMeta, abstractmethod


class WFOptProc(metaclass=ABCMeta):
    """Wave function optimization procedure spec."""

    @abstractmethod
    def exec(self, *args, **kwargs):
        """"""
        pass
