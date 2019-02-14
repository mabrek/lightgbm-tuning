#!/usr/bin/env python

import argparse
import warnings
from multiprocessing import Pool, Lock
import logging
from functools import partial

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import ParameterSampler
from sklearn.exceptions import UndefinedMetricWarning
from scipy.stats import randint as randint
from scipy.stats import uniform as uniform
import lightgbm as lgb

from utils import loguniform, evaluate_parameters

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore",
                        category=UndefinedMetricWarning,
                        module='sklearn.metrics')

warnings.filterwarnings("ignore",
                        category=UserWarning,
                        module='lightgbm.basic')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Random search lightgbm parameters')
    parser.add_argument('--name', required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--chunksize', type=int, default=1)
    args = parser.parse_args()

    log_lock = Lock()

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
        X, y, test_size=0.2, stratify=y)

    # TODO don't copy data, pass indexes only
    folds = [
        [lgb.Dataset(X_train.iloc[train_idx], label=y_train.iloc[train_idx],
                     free_raw_data=False),
         lgb.Dataset(X_train.iloc[test_idx], label=y_train.iloc[test_idx],
                     free_raw_data=False)]
        for train_idx, test_idx
        in StratifiedKFold(n_splits=5, random_state=9342).split(X_train, y_train)]

    # TODO split into wide and narrow (for better metric range) sets
    parameter_space = {
        'objective': ['binary'],
        'boosting': ['gbdt'],
        'learning_rate': loguniform(low=-4, high=1, base=10),
        'num_leaves': randint(2, 200),
        'tree_learner': ['serial'],
        'num_threads': [1],  # will use per-parameter-set threads
        'device_type': ['cpu'],
        'seed': randint(1, 100000),

        'max_depth': randint(1, 100),
        'min_data_in_leaf': randint(1, 1000),
        'min_sum_hessian_in_leaf': loguniform(low=-5, high=1, base=10),
        'bagging_fraction': uniform(loc=0.1, scale=0.9),
        'bagging_freq': randint(0, 1000),
        'feature_fraction': uniform(loc=0.2, scale=0.6),
        'max_delta_step': loguniform(low=-4, high=1, base=10),
        'lambda_l1': loguniform(low=-6, high=3, base=10),
        'lambda_l2': loguniform(low=-6, high=3, base=10),
        'min_gain_to_split': loguniform(low=-6, high=3, base=10),

        'min_data_per_group': randint(50, 1000),
        'max_cat_threshold': randint(5, 500),
        'cat_l2': loguniform(low=-1, high=3, base=10),
        'cat_smooth': randint(1, 100),
        'max_cat_to_onehot': randint(1, 10),

        'is_unbalance': [False, True],
        'scale_pos_weight': loguniform(low=-1, high=2, base=10),
        'boost_from_average': [False, True],

        'metric': ['binary_logloss'],

        'verbosity': [-1],
        'max_bin': randint(4, 1024),
        'min_data_in_bin': randint(2, 100),
        'bin_construct_sample_cnt': randint(10, 1000000),
    }

    def parameters_evaluator(parameters):
        evaluate_parameters(
            parameters,
            folds=folds,
            X_val=X_val, y_val=y_val,
            experiment_name=args.name,
            log_file=args.log,
            log_lock=log_lock)

    with Pool(processes=args.processes) as pool:
        results = pool.imap_unordered(
            parameters_evaluator,
            ParameterSampler(parameter_space, args.iterations),
            chunksize=args.chunksize)
        for r in results:
            print('.', end='', flush=True)
