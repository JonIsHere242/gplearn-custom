"""Metrics to evaluate the fitness of a program.

The :mod:`gplearn.fitness` module contains some metric with which to evaluate
the computer programs created by the :mod:`gplearn.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numbers
import numpy as np
from joblib import wrap_non_picklable_objects
from numba import njit

__all__ = ['make_fitness']

class _Fitness(object):
    def __init__(self, function, greater_is_better):
        self.function = function
        self.greater_is_better = greater_is_better
        self.sign = 1 if greater_is_better else -1

    def __call__(self, *args):
        return self.function(*args)

def make_fitness(*, function, greater_is_better, wrap=True):
    if not isinstance(greater_is_better, bool):
        raise ValueError('greater_is_better must be bool, got %s'
                         % type(greater_is_better))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))
    if function.__code__.co_argcount != 3:
        raise ValueError('function requires 3 arguments (y, y_pred, w),'
                         ' got %d.' % function.__code__.co_argcount)
    if not isinstance(function(np.array([1, 1]),
                      np.array([2, 2]),
                      np.array([1, 1])), numbers.Number):
        raise ValueError('function must return a numeric.')

    if wrap:
        return _Fitness(function=wrap_non_picklable_objects(function),
                        greater_is_better=greater_is_better)
    return _Fitness(function=function,
                    greater_is_better=greater_is_better)

@njit
def _exact_rankdata(a):
    """Exact rankdata implementation."""
    arr = np.ravel(a)
    sorter = np.argsort(arr)
    inv = np.empty(sorter.shape[0], dtype=np.intp)
    inv[sorter] = np.arange(sorter.shape[0])
    return (inv + 1).reshape(a.shape)

@njit
def _approximate_rankdata(a, num_buckets=1000):
    """Approximate rankdata using bucketing."""
    min_val, max_val = np.min(a), np.max(a)
    bucket_size = (max_val - min_val) / num_buckets
    return np.floor((a - min_val) / bucket_size).astype(np.float64)

@njit
def _weighted_pearson(y, y_pred, w):
    """Calculate the weighted Pearson correlation coefficient."""
    y_pred_demean = y_pred - np.average(y_pred, weights=w)
    y_demean = y - np.average(y, weights=w)
    corr = ((np.sum(w * y_pred_demean * y_demean) / np.sum(w)) /
            np.sqrt((np.sum(w * y_pred_demean ** 2) *
                     np.sum(w * y_demean ** 2)) /
                    (np.sum(w) ** 2)))
    return np.abs(corr) if np.isfinite(corr) else 0.

@njit
def dynamic_weighted_spearman(y, y_pred, w, threshold=10000):
    """
    Dynamically choose between exact and approximate Spearman correlation.
    """
    if len(y) <= threshold:
        y_ranked = _exact_rankdata(y)
        y_pred_ranked = _exact_rankdata(y_pred)
    else:
        y_ranked = _approximate_rankdata(y)
        y_pred_ranked = _approximate_rankdata(y_pred)
    
    return _weighted_pearson(y_ranked, y_pred_ranked, w)

@njit
def _mean_absolute_error(y, y_pred, w):
    """Calculate the mean absolute error."""
    return np.average(np.abs(y_pred - y), weights=w)

@njit
def _mean_square_error(y, y_pred, w):
    """Calculate the mean square error."""
    return np.average(((y_pred - y) ** 2), weights=w)

@njit
def _root_mean_square_error(y, y_pred, w):
    """Calculate the root mean square error."""
    return np.sqrt(np.average(((y_pred - y) ** 2), weights=w))

@njit
def _log_loss(y, y_pred, w):
    """Calculate the log loss."""
    eps = 1e-15
    inv_y_pred = np.minimum(np.maximum(1 - y_pred, eps), 1 - eps)
    y_pred = np.minimum(np.maximum(y_pred, eps), 1 - eps)
    score = y * np.log(y_pred) + (1 - y) * np.log(inv_y_pred)
    return np.average(-score, weights=w)

weighted_pearson = _Fitness(function=_weighted_pearson,
                            greater_is_better=True)
weighted_spearman = _Fitness(function=dynamic_weighted_spearman,
                             greater_is_better=True)
mean_absolute_error = _Fitness(function=_mean_absolute_error,
                               greater_is_better=False)
mean_square_error = _Fitness(function=_mean_square_error,
                             greater_is_better=False)
root_mean_square_error = _Fitness(function=_root_mean_square_error,
                                  greater_is_better=False)
log_loss = _Fitness(function=_log_loss,
                    greater_is_better=False)

_fitness_map = {'pearson': weighted_pearson,
                'spearman': weighted_spearman,
                'mean absolute error': mean_absolute_error,
                'mse': mean_square_error,
                'rmse': root_mean_square_error,
                'log loss': log_loss}