#!/usr/bin/env python

from multiprocessing import Lock

from scipy.stats import randint as randint
from scipy.stats import uniform as uniform

from utils import loguniform, EVAL_AT, parse_args, read_telecom_churn,\
    evaluate_parameters, run_pool

if __name__ == "__main__":
    parameter_space = {
        'objective': ['binary'],
        'boosting': ['gbdt'],
        'tree_learner': ['serial'],
        'num_threads': [1],  # will spread different parameter sets across cores
        'device_type': ['cpu'],
        'seed': randint(1, 100000),
        'metric': [['binary_logloss', 'auc', 'map', 'binary_error', 'kldiv']],
        'eval_at': [EVAL_AT],
        'verbosity': [-1],

        'bagging_fraction': uniform(loc=0.58, scale=0.01),
        'bagging_freq': randint(265, 275),
        'bin_construct_sample_cnt': randint(1600, 1700),
        'boost_from_average': [False],
        'cat_l2': uniform(loc=0.84, scale=0.01),
        'cat_smooth': uniform(loc=10, scale=1),
        'feature_fraction': uniform(loc=0.31, scale=0.01),
        'is_unbalance': [True],
        'lambda_l1': uniform(loc=2.3, scale=0.1),
        'lambda_l2': uniform(loc=26000, scale=1000),
        'learning_rate': uniform(loc=0.0087, scale=0.0001),
        'max_bin': randint(1300, 1400),
        'max_cat_threshold': randint(1300, 1400),
        'max_cat_to_onehot': randint(30, 40),
        'max_delta_step': uniform(loc=0.81, scale=0.01),
        'max_depth': randint(280, 290),
        'min_data_in_bin': randint(465, 475),
        'min_data_in_leaf': randint(780, 790),
        'min_data_per_group': randint(910, 920),
        'min_gain_to_split': uniform(loc=0.13, scale=0.01),
        'min_sum_hessian_in_leaf': uniform(loc=5.26409e-08, scale=1e-09),
        'num_leaves': randint(1400, 1500),
        'scale_pos_weight': [None]
    }

    args = parse_args()
    log_lock = Lock()
    folds, validation = read_telecom_churn()

    def parameters_evaluator(parameters):
        evaluate_parameters(
            parameters,
            folds=folds,
            validation=validation,
            experiment_name=args.name,
            log_file=args.log,
            log_lock=log_lock,
            num_boost_round=500
        )

    run_pool(parameter_space, args, parameters_evaluator)
