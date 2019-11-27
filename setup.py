#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="urban-climate-downloader",
    packages=find_packages(exclude=['*test']),
    install_requires=['pandas',
                      'xarray',
                      'numpy',
                      'geopy',
                      ],
)
