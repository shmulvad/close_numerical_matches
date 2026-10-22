"""Runs the examples in the docstrings and in the README."""

from __future__ import annotations

import doctest
import importlib
import pathlib
import re

import numpy as np

README = pathlib.Path(__file__).resolve().parent.parent / 'README.md'
FLAGS = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
# The package re-exports a function with the same name as this module.
module = importlib.import_module('close_numerical_matches.find_matches')


def test_docstrings() -> None:
    result = doctest.testmod(module, optionflags=FLAGS, extraglobs={'np': np})
    assert result.attempted > 0
    assert result.failed == 0


def test_readme() -> None:
    # Only the ```python blocks, as one session (later blocks reuse earlier names).
    blocks = re.findall(r'```python\n(.*?)```', README.read_text(), flags=re.S)
    parser = doctest.DocTestParser()
    test = parser.get_doctest('\n'.join(blocks), {}, 'README', str(README), 0)
    runner = doctest.DocTestRunner(optionflags=FLAGS)
    runner.run(test)
    result = runner.summarize(verbose=False)
    assert result.attempted > 0
    assert result.failed == 0
