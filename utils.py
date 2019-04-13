__all__ = [
    'loguniform',
    'read_json_log',
    'log_json',
    'summarize_logs',
    'read_summarized_logs',
    'check_omitted_parameters',
    'N_FOLDS',
    'N_SEEDS',
    'EVAL_AT',
    'METRICS',
    'DATA_METRICS',
    'SPLITS',
    'SPLIT_METRICS',
    'WHOLE_METRICS',
    'CONT_PARAMETERS',
    'LOG_PARAMETERS',
    'SET_PARAMETERS',
    'INT_PARAMETERS',
    'drop_boring_columns',
    'read_full_logs',
    'unfold_iterations',
    'shaderdots',
    'read_narrow',
    'top_mean_dev_auc',
    'top_mean_validation_auc',
    'rolling_min_dev_auc',
    'narrow_filter',
    'parse_args',
    'read_telecom_churn',
    'run_pool'
]

import warnings
import traceback
import sys
from datetime import datetime
import json
from itertools import product
from multiprocessing import Pool
import logging
import argparse

import numpy as np
import pandas as pd

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import ParameterSampler
import lightgbm as lgb

from bokeh.plotting import figure

try:
    import datashader as ds
    import datashader.transfer_functions as tf
    from datashader.bokeh_ext import InteractiveImage
except ImportError as e:
    warnings.warn(f'exception "{e}" while importing datashader, plotting will be unavailable')


logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore",
                        category=UserWarning,
                        module='lightgbm.basic')

N_FOLDS = 5
N_SEEDS = 3

EVAL_AT = [10, 100, 1000]

METRICS = ['binary_logloss', 'auc', 'binary_error', 'kldiv']\
    + [f'map_{i}' for i in EVAL_AT]

DATA_METRICS = ['_'.join([d, m])
                for d, m
                in product(['train', 'dev', 'validation'], METRICS)]

SPLITS = ['split' + str(i) for i in range(N_FOLDS)]

SPLIT_METRICS = ['_'.join([s, m]) for s, m in product(SPLITS, DATA_METRICS)]

WHOLE_METRICS = ['_'.join([d, m])
                 for d, m
                 in product(['whole_train', 'whole_validation'], METRICS)]

CONT_PARAMETERS = [
    'param_bagging_fraction',
    'param_feature_fraction',
    'iteration'
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
    'param_boost_from_average'
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


def read_telecom_churn():
    df = pd.read_csv(
        './data/WA_Fn-UseC_-Telco-Customer-Churn.csv',
        index_col='customerID',
        dtype={
            'gender': 'category',
            'SeniorCitizen': 'category',
            'Partner': 'category',
            'Dependents': 'category',
            'PhoneService': 'category',
            'MultipleLines': 'category',
            'InternetService': 'category',
            'OnlineSecurity': 'category',
            'OnlineBackup': 'category',
            'DeviceProtection': 'category',
            'TechSupport': 'category',
            'StreamingTV': 'category',
            'StreamingMovies': 'category',
            'Contract': 'category',
            'PaperlessBilling': 'category',
            'PaymentMethod': 'category'
        }).sample(
            frac=1, random_state=102984)
    df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')

    y = (df.Churn == 'Yes').astype(np.int8)
    X = df.drop(['Churn', 'tenure', 'TotalCharges'], axis='columns')

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=834936)

    whole_train = lgb.Dataset(
        X_train,
        label=y_train,
        group=[X_train.shape[0]],
        free_raw_data=False
    )
    validation = lgb.Dataset(
        X_val,
        label=y_val,
        group=[X_val.shape[0]],
        free_raw_data=False
    )
    folds = [
        [lgb.Dataset(X_train.iloc[train_idx],
                     label=y_train.iloc[train_idx],
                     group=[len(train_idx)],
                     free_raw_data=False),
         lgb.Dataset(X_train.iloc[test_idx],
                     label=y_train.iloc[test_idx],
                     group=[len(test_idx)],
                     free_raw_data=False)]
        for train_idx, test_idx
        in StratifiedKFold(n_splits=N_FOLDS,
                           random_state=9342).split(X_train, y_train)
    ]
    return folds, validation, whole_train


def parse_args():
    parser = argparse.ArgumentParser(
        description='Random search LightGBM parameters')
    parser.add_argument('--name', required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--chunksize', type=int, default=10)
    return parser.parse_args()


def run_pool(parameter_space, args, evaluator):
    with Pool(processes=args.processes) as pool:
        results = pool.imap_unordered(
            evaluator,
            enumerate(ParameterSampler(parameter_space, args.iterations)),
            chunksize=args.chunksize)
        for r in results:
            print('.', end='', flush=True)


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


def read_json_log(f, chunksize=None):
    return pd.read_json(f, typ='frame', orient='records', lines=True,
                        chunksize=chunksize, dtype={'success': 'object'})


def log_json(file, data):
    with open(file, 'at') as output:
        data['timestamp'] = datetime.now().isoformat()
        print(json.dumps(data), file=output, flush=True)


def evaluate_experiment(experiment, folds, validation, whole_train,
                        experiment_name, log_file, log_lock, num_boost_round):

    experiment_id, parameters = experiment

    if parameters['is_unbalance']:
        parameters['scale_pos_weight'] = None

    log_data = {}
    log_data['name'] = experiment_name
    log_data['experiment_id'] = experiment_id

    root_seed = parameters['seed']
    for sub_seed in range(N_SEEDS):
        parameters['seed'] = root_seed + sub_seed
        log_data = {'param_' + k: v for k, v in parameters.items()}

        try:
            metrics = {}
            for fold in range(len(folds)):
                train, dev = folds[fold]

                split_result = {}
                lgb.train(parameters,
                          train,
                          valid_sets=[train, dev, validation],
                          valid_names=['train', 'dev', 'validation'],
                          evals_result=split_result,
                          num_boost_round=num_boost_round,
                          verbose_eval=False)
                for data_name, scores in split_result.items():
                    for score_name, score_values in scores.items():
                        metrics[f'split{fold}_{data_name}_{score_name}'] = score_values

            whole_result = {}
            lgb.train(parameters,
                      whole_train,
                      valid_sets=[whole_train, validation],
                      valid_names=['train', 'validation'],
                      evals_result=whole_result,
                      num_boost_round=num_boost_round,
                      verbose_eval=False)
            for data_name, scores in whole_result.items():
                for score_name, score_values in scores.items():
                    metrics[f'whole_{data_name}_{score_name}'] = score_values

            metrics['success'] = True
            log_data.update(metrics)

        except Exception as e:
            warnings.warn(f'got Exception "{e}" for parameters {parameters}')
            traceback.print_exc(file=sys.stderr)  # TODO use logger instead
        finally:
            with log_lock:
                log_json(log_file, log_data)


def summarize_logs(df):
    df = df[df.success.fillna(False)].rename(columns=lambda x: x.replace('@', '_'))

    rows = []
    for row in df.itertuples():
        iterations = pd.DataFrame(
            {k: getattr(row, k)
             for k in row._fields
             if k in SPLIT_METRICS + WHOLE_METRICS})

        for m in DATA_METRICS:
            c = ['_'.join([s, m]) for s in SPLITS]
            iterations['mean_' + m] = iterations[c].mean(axis=1)
            iterations.drop(c, axis=1, inplace=True)  # TODO make optional

        for m in METRICS:
            iterations['dev_train_diff_' + m] =\
                iterations['mean_dev_' + m] - iterations['mean_train_' + m]

        if 'experiment_id' in row._fields:
            iterations['experiment_id'] = row.experiment_id
        else:
            iterations['experiment_id'] = row.Index

        iterations.index.name = 'iteration'
        iterations.reset_index(inplace=True)
        iterations.iteration += 1

        rows.append(iterations)

    split_summaries = pd.concat(rows, ignore_index=True, copy=False)

    return split_summaries.join(df.drop(columns=(SPLIT_METRICS + WHOLE_METRICS)), on='experiment_id')


def unfold_iterations(df):
    df = df[df.success.fillna(False)].rename(columns=lambda x: x.replace('@', '_'))

    rows = []
    for row in df.itertuples():

        for s in range(N_FOLDS):
            one_split_metrics = {'split' + str(s) + '_' + m: m for m in DATA_METRICS}

            iterations = pd.DataFrame(
                {one_split_metrics[k]: getattr(row, k)
                 for k in row._fields if k in one_split_metrics.keys()})

            for m in METRICS:
                iterations['dev_train_diff_' + m] =\
                    iterations['dev_' + m] - iterations['train_' + m]
                iterations['val_dev_diff_' + m] =\
                    iterations['validation_' + m] - iterations['dev_' + m]

            if 'experiment_id' in row._fields:
                iterations['experiment_id'] = row.experiment_id
            else:
                iterations['experiment_id'] = row.Index

            iterations['split'] = s

            iterations.index.name = 'iteration'
            iterations.reset_index(inplace=True)
            iterations.iteration += 1

            rows.append(iterations)

    split_data = pd.concat(rows, ignore_index=True, copy=False)

    return split_data.join(df.drop(columns=SPLIT_METRICS), on='experiment_id')


def drop_boring_columns(df):
    return df.dropna(how='all', axis='columns')\
        .drop(columns=['param_eval_at', 'param_metric'], errors='ignore')\
        .pipe(lambda x: x.loc[:, x.nunique() != 1])


def read_summarized_logs(f, chunksize=1000):
    logs = read_json_log(f, chunksize)
    unfolded = map(summarize_logs, logs)
    cleaned = map(drop_boring_columns, unfolded)
    return pd.concat(list(cleaned), ignore_index=True, sort=True)


def read_full_logs(f, chunksize=1000):
    logs = read_json_log(f, chunksize)
    tidy = map(unfold_iterations, logs)
    cleaned = map(drop_boring_columns, tidy)
    return pd.concat(list(cleaned), ignore_index=True, sort=True)


def check_omitted_parameters(df):
    all_parameters = df.filter(like='param_', axis='columns')\
        .drop('param_seed', axis='columns')\
        .pipe(drop_boring_columns)\
        .columns.values

    return set(all_parameters) - (set(CONT_PARAMETERS) | set(LOG_PARAMETERS) | set(SET_PARAMETERS) | set(INT_PARAMETERS))


def shaderdots(df, x, y, plot_width, plot_height, x_axis_type='linear'):
    def image_callback(x_range, y_range, w, h):
        return tf.dynspread(
            tf.shade(
                ds.Canvas(
                    plot_width=w, plot_height=h, 
                    x_range=x_range, y_range=y_range,
                    x_axis_type=x_axis_type)\
                .points(df, x, y),
            ),
            max_px=1,  threshold=0.5)

    p = figure(plot_width=plot_width, plot_height=plot_height, x_axis_type=x_axis_type,
              x_range=df[x].agg(['min', 'max']).values,
              y_range=df[y].agg(['min', 'max']).values)
    p.xaxis[0].axis_label = x
    p.yaxis[0].axis_label = y
    
    return InteractiveImage(p, image_callback)


def narrow_filter(df):
    return df.query('''1e-4 <= param_learning_rate and param_learning_rate <= 1e-1 \
                    and 600 <= param_min_data_in_leaf and param_min_data_in_leaf <= 1000 \
                    and 1e-10 <= param_min_sum_hessian_in_leaf and param_min_sum_hessian_in_leaf <= 316 \
                    and 0.4 <= param_bagging_fraction and param_bagging_fraction <= 0.8 \
                    and 0.3 <= param_feature_fraction and param_feature_fraction <= 0.8 \
                    and 1e-4 <= param_max_delta_step and  param_max_delta_step <= 1 \
                    and 1 <= param_lambda_l1 and param_lambda_l1 <= 252 \
                    and 1e4 <= param_lambda_l2 and param_lambda_l2 <= 1e6 \
                    and 1e-1 <= param_min_gain_to_split and param_min_gain_to_split <= 1 \
                    and 1 <= param_min_data_per_group and param_min_data_per_group <= 3000 \
                    and ((1 <= param_scale_pos_weight and param_scale_pos_weight <= 8) \
                         or (param_scale_pos_weight != param_scale_pos_weight)) \
                    and 1 <= param_min_data_in_bin and param_min_data_in_bin <= 3000''')


def read_narrow(files):
    for f in files:
        yield pd.read_pickle(f)\
            .assign(file=f)\
            .pipe(narrow_filter)


def top_mean_dev_auc(dfs, n):
    return pd.concat(list(map(lambda df: df.sort_values('mean_dev_auc', ascending=False).iloc[:n], dfs)),
                  ignore_index=True, sort=True)\
            .sort_values('mean_dev_auc', ascending=False).iloc[:n]


def top_mean_validation_auc(dfs, n):
    return pd.concat(list(map(lambda df: df.sort_values('mean_validation_auc', ascending=False).iloc[:n], dfs)),
                  ignore_index=True, sort=True)\
            .sort_values('mean_validation_auc', ascending=False).iloc[:n]


def rolling_min_dev_auc(dfs, n, window):
    return pd.concat(
        list(map(lambda df: df
                 .sort_values('iteration')\
                 .assign(rolling_min_dev_auc=lambda x: x\
                         .groupby('experiment_id')\
                         .rolling(window, min_periods=1, center=True)\
                         .mean_dev_auc.min()\
                         .reset_index(0,drop=True)
                        )\
                 .sort_values('rolling_min_dev_auc', ascending=False).iloc[:n], dfs)),
        ignore_index=True, sort=True)\
    .sort_values('rolling_min_dev_auc', ascending=False).iloc[:n]
