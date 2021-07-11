# Close Numerical Matches

[![PyPI - Downloads](https://img.shields.io/pypi/dm/close_numerical_matches)][pypi]
[![PyPI - Version](https://img.shields.io/pypi/v/close_numerical_matches)][pypi]
![GitHub Workflow Status (branch)](https://github.com/shmulvad/close_numerical_matches/workflows/CI/badge.svg)
[![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/shmulvad/close_numerical_matches/main)][codefactor]
[![GitHub issues](https://img.shields.io/github/issues/shmulvad/close_numerical_matches)][issues]
[![GitHub license](https://img.shields.io/github/license/shmulvad/close_numerical_matches)][license]
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/close_numerical_matches)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)][makeAPullRequest]

This package finds close numerical matches *fast* across two 2D arrays of shape (n, d) and (m, d) (if it can be assumed there will be relatively few matches and d is relatively low). Returns the indices of the matches.

## Installation

You can install `close-numerical-matches` from [PyPI][pypi]:

```bash
$ pip install close-numerical-matches
```

The package is supported on Python 3.6 and above and requires Numpy.


## How to use

Import `find_matches` from `close_numerical_matches` and supply two arrays of shape (n, d) and (m, d) and a given tolerance level. Optionally provide your desired distance metric and a bucket tolerance multiplier. The arguments in more detail:

* `arr0` : `np.ndarray`  
    First array to find matches against. Should be of size (n, d).
* `arr1` : `np.ndarray`  
    Second array to find matches against. Should be of size (m, d).
* `dist` : `{'norm', 'max'}` or `Callable[[np.ndarray], np.ndarray]`, default='norm'  
    Distance metric to calculate distance. `'norm'` and `'max'` are currently supported. If you want some other distance function, you can supply your own function. It should take an (n, d) array as argument and return an (n,) array.
* `tol` : `float`, default=0.1  
    The tolerance where values are considered the similar enough to count as a match. Should be > 0.
* `bucket_tol_mult` : `int`, default=2  
    The tolerance multiplier to use for assigning buckets. Can in some instances make algorithm faster to tweak this. Should never be less than 1.

### Example

```python
>>> import numpy as np
>>> from close_numerical_matches import find_matches
>>> arr0 = np.array([[25, 24], [50, 50], [25, 26]])
>>> arr1 = np.array([[25, 23], [25, 25], [50.6, 50.6], [60, 60]])
>>> find_matches(arr0, arr1, tol=1.0001)
array([[0, 0], [0, 1], [1, 2], [2, 1]])
>>> find_matches(arr0, arr1, tol=0.9999)
array([[1, 2]])
>>> find_matches(arr0, arr1, tol=0.60001)
array([], dtype=int64)
>>> find_matches(arr0, arr1, tol=0.60001, dist='max')
array([[1, 2]])
>>> manhatten_dist = lambda arr: np.sum(np.abs(arr), axis=1)
>>> matches = find_matches(arr0, arr1, tol=0.11, dist=manhatten_dist)
>>> matches
array([[0, 1], [0, 1], [2, 1]])
>>> indices0, indices1 = matches.T
>>> arr0[indices0]
array([[25, 24], [25, 24], [25, 26]])
```

More examples can be found in the [test cases][testCasesFile].

## How fast is it?

Here is an unscientific example:

```python
from timeit import default_timer as timer
import numpy as np
from close_numerical_matches import naive_find_matches, find_matches

arr1 = np.random.rand(320_000, 2)
arr2 = np.random.rand(44_000, 2)

start = timer()
naive_find_matches(arr1, arr2, tol=0.001)
end = timer()
print(end - start)  # 255.335 s

start = timer()
find_matches(arr1, arr2, tol=0.001)
end = timer()
print(end - start)  # 5.821 s
```


## How it works

Instead of comparing every element in the first array against every element in the second array, resulting in an O(nmd) runtime, all elements are at first assigned to buckets so only elements that are relatively close are compared. In the case of relatively few matches and a low dimensionality d, this cuts the runtime down to almost linear O((n + m)d).

In general, the algorithm runtime of the bucket approach is O((n + m)d + Bd³ + ∑\_{b ∈ B} n\_b m\_b) where B is the number of buckets and n\_b and m\_b are the number of items assigned to bucket b. As can be seen, it scales bad with dimensionality and also does not improve from the naive approach if all elements are assigned to the same bucket. In case the bucket approach is likely to be slower than the naive approach, this library will fall back to the naive approach.

[testCasesFile]: https://github.com/shmulvad/close_numerical_matches/blob/main/tests/test_find_matches.py
[pypi]: https://pypi.org/project/close-numerical-matches/
[license]: https://github.com/shmulvad/close_numerical_matches/blob/master/LICENSE
[issues]: https://github.com/shmulvad/close_numerical_matches/issues
[release]: https://github.com/shmulvad/close_numerical_matches/releases/latest
[codefactor]: https://www.codefactor.io/repository/github/shmulvad/close_numerical_matches
[makeAPullRequest]: https://makeapullrequest.com
