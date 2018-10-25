"""
    PhDThesis-Lib
    ~~~~~~~~~~~~~

    A library to estimate and analyse the physical properties of
    an interacting quantum many-body Bose gas within a multi-rods
    structure. It implements the following methods:

    * Mean-field theory, based on the solutions of the
      Gross-Pitaevskii equation.
    * Variational an Diffusion Monte Carlo approaches.

    It's written in pure Python. It uses `Numba <http://numba.pydata.org/>`_
    to accelerate performance-critical routines that execute CPU-intensive
    calculations, as well as `Dask <http://dask.pydata.org/en/latest/>`_
    to distribute and handle the asynchronous execution of several
    tasks in parallel. The library is released under the BSD-3 License.
"""

from setuptools import setup

setup(
        name='Ph.D. Thesis Code Library',
        version='0.3.0',
        url='https://bitbucket.org/oarodriguez/phdthesis-lib',
        packages=[
            'phdthesis_lib',
            'phdthesis_lib.qmc_lib',
            'phdthesis_lib.multirods_qmc'
        ],
        license='BSD-3',
        author='Omar Abel Rodríguez-López',
        author_email='oarodriguez.mx@gmail.com',
        description='A library to estimate and analyse the physical '
                    'properties of an interacting quantum many-body '
                    'Bose gas within a multi-rods structure.',
        long_description=__doc__,
        include_package_data=True,
        zip_safe=False,
        platforms='any',
        python_requires='>=3.6.1',
        install_requires=[
            'numpy>=0.10',
            'scipy>=0.17.0',
            'matplotlib>=2.2',
            'numba>=0.35',
            'h5py>=2.5.0',
            'pytables>=3.4',
            'mpmath>=1.0',
            'progressbar2>=3.6.0',
            'pytz>=2016.4',
            'tzlocal>=1.2',
            'dask>=0.17',
            'distributed>=1.2',
            'PyYAML>=3.10',
            'jinja2>=2.10',
            'decorator'
        ],
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Topic :: Scientific/Engineering :: Physics',
            'Topic :: Software Development :: Libraries :: Python Modules'
        ]
)
