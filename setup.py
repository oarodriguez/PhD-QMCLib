"""
    MyResearch-Libs
    ~~~~~~~~~~~~~~~

    A collection of libraries to estimate the physical properties of an
    interacting, quantum many-body system. These libraries are part of
    my doctoral research.

    The source code it's written in pure Python. It uses
    `Numba <http://numba.pydata.org/>`_ to accelerate performance-critical
    routines that execute CPU-intensive calculations, as well as
    `Dask <http://dask.pydata.org/en/latest/>`_ to distribute and handle
    the asynchronous execution of several tasks in parallel. The library
    is released under the BSD-3 License.
"""

from setuptools import setup

DESCRIPTION = 'A collection of libraries to estimate ' \
              'the physical properties of an interacting, quantum ' \
              'many-body system.'

setup(
        name='Ph.D. Thesis Code Libraries',
        version='0.3.0',
        url='https://bitbucket.org/oarodriguez/myresearch-libs/',
        packages=[
            'my_research_libs',
            'my_research_libs.qmc_base',
            'my_research_libs.multirods_qmc'
        ],
        license='BSD-3',
        author='Omar Abel Rodríguez-López',
        author_email='oarodriguez.mx@gmail.com',
        description=DESCRIPTION,
        long_description=__doc__,
        include_package_data=True,
        zip_safe=False,
        platforms='any',
        python_requires='>=3.6.1',
        install_requires=[
            'numpy>=1.10',
            'scipy>=1.0',
            'matplotlib>=2.2',
            'numba>=0.39',
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
            'decorator>=4.2',
            'pytest>=3.4',
            'gmpy2>=2.0'
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
