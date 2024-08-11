"""
The functions used to create programs.

The :mod:`gplearn.functions` module contains all of the functions used by
gplearn programs. It also contains helper methods for a user to define their
own custom functions.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
from joblib import wrap_non_picklable_objects
from typing import Callable, List
from numba import njit, vectorize, prange




__all__ = ['make_function']

class _Function:


    def __init__(self, function: Callable, name: str, arity: int):
        self.function = function
        self.name = name
        self.arity = arity

    def __call__(self, *args: np.ndarray) -> np.ndarray:
        return self.function(*args)

def _test_function_closure(function: Callable, arity: int, test_values: List[np.ndarray], name: str) -> None:

    for values in test_values:
        args = [values] * arity
        result = function(*args)
        
        if not np.all(np.isfinite(result)):
            raise ValueError(f'Supplied function {name} produces non-finite values.')
        
        if np.any(np.abs(result) > np.finfo(np.float64).max):
            raise ValueError(f'Supplied function {name} produces values exceeding float64 limits.')





def _validate_function(function: Callable, name: str, arity: int) -> None:

    if not isinstance(arity, int):
        raise ValueError(f'Arity must be an int, got {type(arity)}')
    if not isinstance(name, str):
        raise ValueError(f'Name must be a string, got {type(name)}')

    if not isinstance(function, np.ufunc):
        if function.__code__.co_argcount != arity:
            raise ValueError(
                f'Arity {arity} does not match the required number of '
                f'function arguments ({function.__code__.co_argcount}).'
            )

    test_values = [
        np.ones(10),
        np.zeros(10),
        -1 * np.ones(10),
        np.array([np.finfo(np.float64).max, np.finfo(np.float64).min, np.nan, np.inf, -np.inf] * 2)
    ]

    args = [test_values[0]] * arity
    try:
        result = function(*args)
    except (ValueError, TypeError):
        raise ValueError(f'Supplied function {name} does not support arity of {arity}.')
    
    if not hasattr(result, 'shape') or result.shape != (10,):
        raise ValueError(
            f'Supplied function {name} must return a numpy array of the same shape as input vectors.'
        )

    _test_function_closure(function, arity, test_values, name)

def make_function(
    *,
    function: Callable,
    name: str,
    arity: int,
    wrap: bool = True
) -> _Function:
    """Make a function object for use in genetic programming."""
    _validate_function(function, name, arity)

    if wrap:
        function = wrap_non_picklable_objects(function)

    return _Function(function=function, name=name, arity=arity)





NumbaTest = True

if NumbaTest:
    @vectorize(['float64(float64)'], nopython=True, fastmath=True)
    def _protected_log(x1):
        abs_x1 = abs(x1)
        return np.log(abs_x1 + 1e-10)  # Add a small constant to avoid log(0)

    @vectorize(['float64(float64, float64)'], nopython=True, fastmath=True)
    def _protected_division(x1, x2):
        return 1.0 if abs(x2) <= 0.001 else x1 / x2

    @vectorize(['float64(float64)'], nopython=True, fastmath=True)
    def _protected_sqrt(x1):
        return np.sqrt(abs(x1))

    @vectorize(['float64(float64)'], nopython=True, fastmath=True)
    def _protected_inverse(x1):
        return 0.0 if abs(x1) <= 0.001 else 1.0 / x1

    @njit(fastmath=True)
    def _sigmoid(x1):
        return 1.0 / (1.0 + np.exp(-np.clip(x1, -100, 100)))

    # Wrapper functions to handle NumPy warnings
    def protected_log(x1):
        with np.errstate(divide='ignore', invalid='ignore'):
            return _protected_log(x1)

    def protected_division(x1, x2):
        return _protected_division(x1, x2)

    def protected_sqrt(x1):
        return _protected_sqrt(x1)

    def protected_inverse(x1):
        return _protected_inverse(x1)

    def sigmoid(x1):
        return _sigmoid(x1)

else:
    # Optimized non-Numba functions
    def protected_division(x1, x2):
        """Closure of division (x1/x2) with improved handling of small denominators."""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(x1, x2)
            mask = np.abs(x2) < 1e-10
            result[mask] = np.sign(x1[mask]) * np.sign(x2[mask]) * 1e10
            result[np.isnan(result) | np.isinf(result)] = 0
            return result.astype(np.float64)


    def protected_sqrt(x1: np.ndarray) -> np.ndarray:
        return np.sqrt(np.abs(x1))

    def protected_log(x1: np.ndarray) -> np.ndarray:
        abs_x1 = np.abs(x1)
        mask = abs_x1 > 0.001
        result = np.zeros_like(x1)
        np.log(abs_x1, out=result, where=mask)
        return result

    def protected_inverse(x1: np.ndarray) -> np.ndarray:
        mask = np.abs(x1) > 0.001
        result = np.zeros_like(x1)
        np.divide(1.0, x1, out=result, where=mask)
        return result

    def sigmoid(x1: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(np.clip(-x1, -100, 100)))

# Define function instances
add2 = _Function(function=np.add, name='add', arity=2)
sub2 = _Function(function=np.subtract, name='sub', arity=2)
mul2 = _Function(function=np.multiply, name='mul', arity=2)
max2 = _Function(function=np.maximum, name='max', arity=2)
min2 = _Function(function=np.minimum, name='min', arity=2)
abs1 = _Function(function=np.abs, name='abs', arity=1)
neg1 = _Function(function=np.negative, name='neg', arity=1)
sin1 = _Function(function=np.sin, name='sin', arity=1)
cos1 = _Function(function=np.cos, name='cos', arity=1)
tan1 = _Function(function=np.tan, name='tan', arity=1)

# Custom functions at the bottom
div2 = _Function(function=protected_division, name='div', arity=2)
sqrt1 = _Function(function=protected_sqrt, name='sqrt', arity=1)
log1 = _Function(function=protected_log, name='log', arity=1)
inv1 = _Function(function=protected_inverse, name='inv', arity=1)
sig1 = _Function(function=sigmoid, name='sig', arity=1)

# Map of function names to function instances
_function_map = {
    'add': add2,
    'sub': sub2,
    'mul': mul2,
    'div': div2,
    'sqrt': sqrt1,
    'log': log1,
    'abs': abs1,
    'neg': neg1,
    'inv': inv1,
    'max': max2,
    'min': min2,
    'sin': sin1,
    'cos': cos1,
    'tan': tan1
}