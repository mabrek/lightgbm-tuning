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

        'bagging_fraction': [0.585696304992511],
        'bagging_freq': [270],
        'bin_construct_sample_cnt': [1653],
        'boost_from_average': [False],
        'cat_l2': [0.8428665084986411],
        'cat_smooth': [10.318258820870582],
        'feature_fraction': [0.313659590764367],
        'is_unbalance': [True],
        'lambda_l1': [2.374574287201392],
        'lambda_l2': [26497.176002719025],
        'learning_rate': [0.008779600608473001],
        'max_bin': [1361],
        'max_cat_threshold': [1328],
        'max_cat_to_onehot': [34],
        'max_delta_step': [0.8112792283390121],
        'max_depth': [283],
        'min_data_in_bin': [470],
        'min_data_in_leaf': [787],
        'min_data_per_group': [914],
        'min_gain_to_split': [0.136626601532149],
        'min_sum_hessian_in_leaf': [5.264091232156215e-08],
        'num_leaves': [1495],
        'scale_pos_weight': [None],
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
