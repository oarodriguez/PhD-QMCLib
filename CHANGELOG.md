# CHANGELOG

## Version 0.3

:sparkles: :art: This release adds the next major features:

- The Quantum Monte Carlo (QMC) library. This includes several abstract classes
  that define the building blocks of the library, as well as several utility
  functions.
- Abstract classes for QMC models with a trial wave function of the Bijl-Jastrow
  type.
- Interfaces to define generalized universal functions (using the numba
  @guvectorize decorator) that evaluate the properties of a Jastrow QMC model.
  The execution is done in parallel by default.
- An interface to do the sampling of the probability density function using the
  Metropolis-Hastings algorithm used to realize a Variational Monte Carlo
  calculation.
- Classes and routines to define a system consisting of a Bose gas in a
  multi-rods structure with repulsive, contact interactions, with a Jastrow
  trial wave function. Also, the sampling of the probability density is
  implemented.
- Remove LateX sources of Ph.D. Thesis report.
- Remove unrelated jupyter notebooks.

## Version 0.2

- Initialize the library
- Add LaTeX sources for Ph.D. Thesis report.

## Version 0.1

- :tada: Initial commit
