#!/usr/bin/env python

input("Dependencies: geopy, jupyter, netCDF4, scipy and cftime (1.0.1) will be installed on this environment. Press Enter to continue...")

from setuptools import setup, find_packages

setup(
    name="urban-climate-downloader",
    packages=find_packages(exclude=['*test']),
    install_requires=['pandas',
                      'numpy',
                      'geopy',
                      'jupyter',
                      'netCDF4',
                      'scipy',
                      'cftime==1.0.1',
                      ],
)
