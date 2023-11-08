from __future__ import annotations

import os
import pathlib
import re

from setuptools import find_packages
from setuptools import setup

BASE_DIR = pathlib.Path(__file__).resolve().parent


def load_version() -> str:
    file_path = BASE_DIR / 'close_numerical_matches' / 'version.py'
    file_contents = file_path.read_text()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    match = re.search(version_regex, file_contents, re.M)
    if not match:
        raise RuntimeError('Unable to find version string')

    return match.group(1)


NAME = 'close-numerical-matches'
DESCRIPTION = 'Finds close numerical matches across two arrays'
URL = 'https://github.com/shmulvad/close_numerical_matches'
EMAIL = 'shmulvad@gmail.com'
AUTHOR = 'Soeren Mulvad'
REQUIRES_PYTHON = '>=3.8.0'
REQUIRED = ['numpy', 'mypy']
VERSION = load_version()

readme_path = BASE_DIR / 'README.md'
long_description = readme_path.read_text()

SKIP_COMPILE = os.getenv('SS_SKIP_COMPILE', '0').strip() == '1'
COMPILE = not SKIP_COMPILE
extra_kwargs = {}
if COMPILE:
    from mypyc.build import mypycify

    extra_kwargs = {
        'ext_modules': mypycify(
            [
                '--disallow-untyped-defs',
                'close_numerical_matches/find_matches.py',
            ]
        ),
    }

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
    **extra_kwargs,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering',
        'Typing :: Typed',
    ],
)
