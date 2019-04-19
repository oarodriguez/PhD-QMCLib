# CHANGELOG

## Version 0.11.0

Main changes in this version:

- Add a ``Record`` class to ease the construction of numpy structured arrays
  from annotated ``attr`` classes.

For the analysis of the properties of the "multi-rods Bloch-Phonon" model (MR-BP):

- Rename the ``multirods_qmc.bloch_phonon`` package to ``mrbp_qmc``.
- Improving tests.
- Add support to calculate the estimators of the static structure factor
  and energy during a VMC sampling.
- Add support to export and import data from a VMC sampling to HDF5 files.
- Calculate the density distribution of the system during DMC sampling.

## Version 0.10.1

🐛  Bug fixes.

## Version 0.10.0

In this version:

- Adding ``ProcCLI`` class to handle the execution of DMC procedures from the
  command line.
- Implement classes to read and write DMC procedure results to HDF5 files.
- 🔥 Removing (temporarily) numba jit option ``cache=True`` as it does not
  plays well with ``pyyaml``.

Also, we fix up several memory leaks in numba compiled functions. The memory
leaks are the consequence of:

- Initializing a scalar variable in a function compiled with ``parallel=True``,
  and then accumulate other values on it inside a ``numba.prange`` loop.
- Using the enumerate function inside a jit compiled function. This is a known issue: <https://github.com/numba/numba/issues/3473.>
- Accumulating estimators inside a ``numba.prange`` parallel loop, similar
  to the first leak.

## Version 0.9.1

🐛  Bug fixes and warnings.

## Version 0.9.0

Changes in this version:

- Modifications in the reblocking routines of the ``stats`` package,
  in order to do an on-the-fly reblocking analysis over 2d arrays
  (tabular data).
- Improvements in code quality.

## Version 0.8.0

Version highlights:

- Add the ``Reblocking`` class to do a blocking analysis over serially correlated
  data.
- The VMC and DMC samplings iterable interfaces return iterators that generate
  data indefinitely. Any iteration over the sampling must be broken explicitly.
- Support to realize VMC and DMC samplings in batches. A sampling done in this
  way realizes several Markov chain steps before it yields the data to the
  caller scope.

## Version 0.7.0

- Adding the branching spec as an attribute of the DMC state.
- Change DMC sampling to yield single DMC states.

## Version 0.6.1

🐛 Bug fixes:

- Add workaround for numba v0.41.0 bug in VMC sampling. See <https://github.com/numba/numba/issues/3565> for details.

## Version 0.6.0

Changes and new features with this version:

- Remove the energy slot in the system configurations of a Jastrow
  QMC model.
- Converting some attributes to properties in attrs classes, like
  the model spec and the sampling classes, to ease their conversion to
  dictionary objects using the ``asdict`` function.

## Version 0.5.1

🐛 Bug fixes.

## Version 0.5.0

This version adds support to realize a Diffusion Monte Carlo calculation,
including the estimation of the ground state energy, the one-body
density matrix and the static structure factor.

Highlights:

- Implementation of the core functions to realize a DMC sampling using a
  branching scheme to sample the ground state density function.
- The sampling has the capability to use all the CPU cores available in
  the host machine.
- The sampling is done in batches to minimize the time used for
  switching from the jit-compiled code to pure python code.
- The one-body density matrix and structure factor routines are
  implemented as jit-compiled generalized universal functions (numba).
  Just as the DMC sampling, they have with capabilities to use all the
  CPU cores available.

## Version 0.4.2

🐛 Bug fixes.

## Version 0.4.1

🐛 Bug fixes.

## Version 0.4.0

This version brings a lot of big changes:

- Renaming of the main package to my_research_libs.
- Renaming of several sub-packages and modules.
- The use of named-tuples instead of plain tuples to handle the
  parameters of the jit-compiled core functions of a model.
- The use of named-tuples to handle the parameters of the jit-compiled
  core functions of a VMC sampling.
- The use of the attrs library to handle the initialization of the
  classes that represent the spec of a model. The same is done for the
  spec of a VMC sampling.
- The implementation of numpy generalized universal functions with
  parallel execution capabilities, thanks to numba, to evaluate the main
  physical properties of a model.
- Improved tests.
- Using the ``cached-property`` library to cache the most costly property
  attributes used to compile functions with numba.
- And, of course, a lot of unused, legacy code has been removed 🔥.

## Version 0.3.0

✨ 🎨 This release adds the next major features:

- The Quantum Monte Carlo (QMC) library. This includes several abstract classes
  that define the building blocks of the library, as well as several utility
  functions.
- Abstract classes for QMC models with a trial wave function of the Bijl-Jastrow
  type.
- Interfaces to define generalized universal functions (using the numba
  ``guvectorize`` decorator) that evaluate the properties of a Jastrow QMC model.
  The execution is done in parallel by default.
- An interface to do the sampling of the probability density function using the
  Metropolis-Hastings algorithm used to realize a Variational Monte Carlo
  calculation.
- Classes and routines to define a system consisting of a Bose gas in a
  multi-rods structure with repulsive, contact interactions, with a Jastrow
  trial wave function. Also, the sampling of the probability density is
  implemented.
- 🔥 Remove LateX sources of Ph.D. Thesis report.
- 🔥 Remove unrelated jupyter notebooks.

## Version 0.2

- Initialize the library.
- Add LaTeX sources for Ph.D. Thesis report.

## Version 0.1

- ✨✨ Initial commit
