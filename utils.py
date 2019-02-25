__all__ = [
    'loguniform',
    'read_json_log',
    'log_json',
    'scatterdots',
    'scatterjitter',
    'clean_experiment_logs',
    'check_omitted_parameters',
    'N_FOLDS',
    'METRICS',
    'CONT_PARAMETERS',
    'LOG_PARAMETERS',
    'SET_PARAMETERS',
    'INT_PARAMETERS',
    'drop_boring_columns'
]

import warnings
import traceback
import sys
from datetime import datetime
import json
from timeit import default_timer as timer
from itertools import product

import numpy as np
import pandas as pd

from sklearn.utils import check_random_state
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score, precision_score, recall_score, accuracy_score, f1_score
import lightgbm as lgb

from bokeh.plotting import figure
from bokeh.transform import jitter


N_FOLDS = 5

METRICS = ['accuracy', 'average_precision', 'f1', 'log_loss', 'precision', 'recall', 'roc_auc']

CONT_PARAMETERS = [
    'param_bagging_fraction',
    'param_feature_fraction',
    'mean_best_iteration'
]

LOG_PARAMETERS = [
    'param_learning_rate',
    'param_min_sum_hessian_in_leaf',
    'param_max_delta_step',
    'param_lambda_l1', 
    'param_lambda_l2',
    'param_min_gain_to_split', 
    'param_cat_l2',
    'param_cat_smooth', 
    'param_scale_pos_weight'
]

SET_PARAMETERS = [
    'param_is_unbalance', 
    'param_boost_from_average',
    'param_metric'
]

INT_PARAMETERS = [
    'param_num_leaves',
    'param_max_depth',
    'param_min_data_in_leaf',
    'param_bagging_freq',
    'param_min_data_per_group',
    'param_max_cat_threshold',
    'param_max_cat_to_onehot', 
    'param_max_bin',
    'param_min_data_in_bin',
    'param_bin_construct_sample_cnt',
]


# from https://github.com/scikit-learn/scikit-learn/blob/19bffee9b172cf169fded295e3474d1de96cdc57/sklearn/utils/random.py
class loguniform:
    """A class supporting log-uniform random variables.
    Parameters
    ----------
    low : float
        The log minimum value
    high : float
        The log maximum value
    base : float
        The base for the exponent.
    Methods
    -------
    rvs(self, size=None, random_state=None)
        Generate log-uniform random variables
    Notes
    -----
    This class generates values between ``base**low`` and ``base**high`` or
        base**low <= loguniform(low, high, base=base).rvs() <= base**high
    The logarithmic probability density function (PDF) is uniform. When
    ``x`` is a uniformly distributed random variable between 0 and 1, ``10**x``
    are random variales that are equally likely to be returned.
    """
    def __init__(self, low, high, base=10):
        """
        Create a log-uniform random variable.
        """
        self._low = low
        self._high = high
        self._base = base

    def rvs(self, size=None, random_state=None):
        """
        Generates random variables with ``base**low <= rv <= base**high``
        where ``rv`` is the return value of this function.
        Parameters
        ----------
        size : int or tuple, optional
            The size of the random variable.
        random_state : int, RandomState, optional
            A seed (int) or random number generator (RandomState).
        Returns
        -------
        rv : float or np.ndarray
            Either a single log-uniform random variable or an array of them
        """
        _rng = check_random_state(random_state)
        unif = _rng.uniform(self._low, self._high, size=size)
        rv = np.power(self._base, unif)
        return rv


def read_json_log(f):
    return pd.read_json(f, typ='frame', orient='records', lines=True)


def log_json(file, data):
    with open(file, 'at') as output:
        data['timestamp'] = datetime.now().isoformat()
        print(json.dumps(data), file=output, flush=True)


def scatterdots(cds, x, y, alpha=0.1, x_axis_type='auto', y_axis_type='auto'):
    p = figure(x_axis_type=x_axis_type, y_axis_type=y_axis_type)
    p.square(x=x, y=y, source=cds, size=1, alpha=alpha)
    p.xaxis[0].axis_label = x
    return p


def scatterjitter(cds, x, y, alpha=0.1, jitter_width=1/200):
    p = figure()
    p.square(x=jitter(x, distribution='normal',
                      width=np.ptp(cds.data[x]) * jitter_width),
             y=y, source=cds, size=1, alpha=alpha)
    p.xaxis[0].axis_label = x
    return p


def evaluate_parameters(parameters, folds, X_val, y_val, experiment_name,
                        log_file, log_lock):
    if parameters['is_unbalance']:
        parameters['scale_pos_weight'] = None

    log_data = {'param_' + k: v for k, v in parameters.items()}
    log_data['name'] = experiment_name

    try:
        metrics = {}
        for fold in range(len(folds)):
            train, test = folds[fold]

            train_start = timer()
            model = lgb.train(parameters,
                              train, valid_sets=test,
                              num_boost_round=5000,
                              early_stopping_rounds=50,
                              verbose_eval=False)
            train_time = timer() - train_start

            train_pred = model.predict(train.get_data())

            pred_start = timer()
            test_pred = model.predict(test.get_data())
            pred_time = timer() - pred_start

            val_pred = model.predict(X_val)

            metrics.update({
                f'split{fold}_train_time': train_time,
                f'split{fold}_pred_time': pred_time,
                f'split{fold}_best_iteration': model.best_iteration,
            })

            for data_name, data_true, data_pred in [
                    ['train', train.get_label(), train_pred],
                    ['test', test.get_label(), test_pred],
                    ['val', y_val, val_pred]]:
                for scorer in [log_loss, roc_auc_score, average_precision_score]:
                    metrics[f'split{fold}_{data_name}_{scorer.__name__}'] \
                        = scorer(data_true, data_pred)
                for label_scorer in [accuracy_score, precision_score, recall_score, f1_score]:
                    metrics[f'split{fold}_{data_name}_{label_scorer.__name__}'] \
                        = label_scorer(data_true, data_pred > 0.5)

        metrics['success'] = True
        log_data.update(metrics)

    except Exception as e:
        warnings.warn(f'got Exception "{e}" for parameters {parameters}')
        traceback.print_exc(file=sys.stderr)  # TODO use logger instead
    finally:
        with log_lock:
            log_json(log_file, log_data)


def clean_experiment_logs(df):
    df = df[df.success == True]\
        .rename(lambda x: x.replace('_score', ''), axis='columns')

    folds = ['split' + str(i) for i in range(0, 5)]

    for t, m in product(['train', 'test', 'val'], METRICS):
        b = '_' + t + '_' + m
        c = [f + b for f in folds]
        df['mean' + b] = df[c].mean(axis=1)
        df['range' + b] = df[c].max(axis=1) - df[c].min(axis=1)
        df.drop(c, axis=1, inplace=True)

    for m in METRICS:
        df['test_overfit_' + m] = df['mean_train_' + m] - df['mean_test_' + m]
        df['val_test_diff_' + m] = df['mean_test_' + m] - df['mean_val_' + m]

    train_time_cols = [f + '_train_time' for f in folds]
    df['mean_train_time'] = df[train_time_cols].mean(axis=1)
    pred_time_cols = [f + '_pred_time' for f in folds]
    df['mean_pred_time'] = df[pred_time_cols].mean(axis=1)
    df.drop(train_time_cols + pred_time_cols, axis=1, inplace=True)

    iteration_cols = [f + '_best_iteration' for f in folds]
    df['mean_best_iteration'] = df[iteration_cols].mean(axis=1)
    df['range_best_iteration'] = \
        df[iteration_cols].max(axis=1) - df[iteration_cols].min(axis=1)

    df.drop(iteration_cols, axis=1, inplace=True)

    return df


def drop_boring_columns(df):
    return df.dropna(how='all', axis='columns')\
        .pipe(lambda x: x.loc[:, x.nunique() != 1])


def check_omitted_parameters(df):
    all_parameters = df.filter(like='param_', axis='columns')\
        .drop('param_seed', axis='columns')\
        .pipe(drop_boring_columns)\
        .columns.values

    return set(all_parameters) - (set(CONT_PARAMETERS) | set(LOG_PARAMETERS) | set(SET_PARAMETERS) | set(INT_PARAMETERS))
