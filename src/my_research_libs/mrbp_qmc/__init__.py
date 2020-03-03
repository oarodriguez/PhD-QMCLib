"""
    my_research_libs.mrbp_qmc
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    Package for QMC routines of the Bloch-Phonon Jastrow model.
"""
#
from . import config, dmc, dmc_exec, vmc, vmc_exec, vmc_ndf, wf_opt

#
from .model import (
    CFCSpec, CSWFOptimizer, CoreFuncs, OBFParams, Params, PhysicalFuncs,
    Spec, TBFParams, core_funcs
)
