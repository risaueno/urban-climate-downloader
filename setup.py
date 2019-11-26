#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="cityclim",
    packages=find_packages(exclude=['*test']),
    install_requires=['pandas',
                      'xarray',
                      'numpy',
		      'pickle',
                      'geopy',
                      ],
)
