#!/usr/bin/env python

from multiprocessing import Lock

from scipy.stats import randint as randint
from scipy.stats import uniform as uniform

from utils import loguniform, EVAL_AT, parse_args, read_telecom_churn,\
    evaluate_experiment, run_pool, generate_random_experiments

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

        'bagging_enable': [False],
        'boost_from_average': [False, True],
        'is_unbalance': [True],

        'bagging_fraction': uniform(loc=0.4, scale=0.4),
        'bagging_freq': randint(1, 50),
        'bin_construct_sample_cnt': randint(2140, 7123),
        'cat_l2': loguniform(low=-3, high=5, base=10),
        'cat_smooth': loguniform(low=-4, high=5, base=10),
        'feature_fraction': uniform(loc=0.4, scale=0.2),
        'lambda_l1': loguniform(low=-7, high=0, base=10),
        'lambda_l2': loguniform(low=-7, high=3, base=10),
        'learning_rate': loguniform(low=-4, high=-2, base=10),
        'max_bin': randint(525, 1578),
        'max_cat_threshold': randint(481, 1521),
        'max_cat_to_onehot': randint(24, 72),
        'max_delta_step': loguniform(low=0, high=5, base=10),
        'max_depth': randint(93, 271),
        'min_data_in_bin': randint(1030, 3180),
        'min_data_in_leaf': randint(350, 892),
        'min_data_per_group': randint(1029, 2914),
        'min_gain_to_split': loguniform(low=-8, high=-2, base=10),
        'min_sum_hessian_in_leaf': loguniform(low=-7, high=0, base=10),
        'num_leaves': randint(1115, 3357),
        'scale_pos_weight': uniform(loc=0.8, scale=5),
    }

    args = parse_args()
    log_lock = Lock()
    folds, validation, whole_train = read_telecom_churn()

    # captures data and lock to use in forked processes
    def evaluator(experiment):
        evaluate_experiment(
            experiment,
            folds=folds,
            validation=validation,
            whole_train=whole_train,
            experiment_name=args.name,
            log_file=args.log,
            log_lock=log_lock,
            num_boost_round=500
        )

    run_pool(
        generate_random_experiments(parameter_space, args.iterations),
        args,
        evaluator)
