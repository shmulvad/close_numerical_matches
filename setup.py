#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os

from setuptools import setup, find_packages

NAME = 'close-numerical-matches'
DESCRIPTION = 'Finds close numerical matches across two arrays'
URL = 'https://github.com/shmulvad/close_numerical_matches'
EMAIL = 'shmulvad@gmail.com'
AUTHOR = 'Soeren Mulvad'
REQUIRES_PYTHON = '>=3.6.0'
REQUIRED = ['numpy']
VERSION = '0.1.0'


DIR = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
try:
    with io.open(os.path.join(DIR, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

try:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(DIR, project_slug, 'version.py'), 'w') as f:
        f.write("__version__ = '{version}'\n".format(version=VERSION))
except Exception:
    pass

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=['tests', '*.tests', '*.tests.*', 'tests.*']),
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Scientific/Engineering'
    ]
)
