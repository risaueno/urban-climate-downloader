#!/usr/bin/env python

# input("Dependencies of this package will be installed on this environment. Press Enter to continue...")

from setuptools import setup, find_packages

setup(
    name="urban-climate-downloader",
    packages=find_packages(exclude=['*test']),
    install_requires=['xarray==0.11.3',
		      'dask',
		      'pandas',
                      'toolz',
                      'numpy',
                      'geopy',
                      'jupyter',
                      'netCDF4',
                      'scipy',
                      'cftime==1.0.1',
                      ],
)
