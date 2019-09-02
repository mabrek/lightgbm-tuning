#!/usr/bin/env python

from multiprocessing import Lock

from scipy.stats import randint as randint
from scipy.stats import uniform as uniform

from utils import loguniform, EVAL_AT, parse_args, read_telecom_churn,\
    evaluate_lgb_experiment, run_pool, generate_random_experiments

if __name__ == "__main__":
    parameter_space = {
        'objective': ['binary'],
        'boosting': ['gbdt'],
        'learning_rate': loguniform(low=-8, high=6, base=10),
        'num_leaves': randint(2, 4000),
        'tree_learner': ['serial'],
        'num_threads': [1],  # will spread different parameter sets across cores
        'device_type': ['cpu'],
        'seed': randint(1, 100000),

        'max_depth': randint(1, 400),
        'min_data_in_leaf': randint(1, 2000),
        'min_sum_hessian_in_leaf': loguniform(low=-10, high=6, base=10),
        'bagging_enable' : [False, True],
        'bagging_fraction': uniform(loc=0.1, scale=0.9),
        'bagging_freq': randint(1, 50),
        'feature_fraction': uniform(loc=0.2, scale=0.8),
        'max_delta_step': loguniform(low=-8, high=6, base=10),
        'lambda_l1': loguniform(low=-10, high=6, base=10),
        'lambda_l2': loguniform(low=-10, high=10, base=10),
        'min_gain_to_split': loguniform(low=-10, high=6, base=10),

        'min_data_per_group': randint(1, 4000),
        'max_cat_threshold': randint(1, 2000),
        'cat_l2': loguniform(low=-10, high=10, base=10),
        'cat_smooth': loguniform(low=-10, high=10, base=10),
        'max_cat_to_onehot': randint(1, 100),

        'is_unbalance': [False, True],
        'scale_pos_weight': uniform(loc=0.1, scale=99.9),
        'boost_from_average': [False, True],

        'metric': [['binary_logloss', 'auc']],

        'verbosity': [-1],
        'max_bin': randint(4, 2048),
        'min_data_in_bin': randint(1, 5000),
        'bin_construct_sample_cnt': randint(5, 10000),
    }

    args = parse_args()
    log_lock = Lock()
    X_train, X_val, y_train, y_val, folds = \
        read_telecom_churn(args.n_folds, args.split_kind)

    # captures data and lock to use in forked processes
    def evaluator(experiment):
        evaluate_lgb_experiment(
            experiment,
            experiment_name=args.name,
            log_file=args.log,
            log_lock=log_lock,
            num_boost_round=500,
            n_seeds=args.n_seeds,
            X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
            folds=folds
        )

    run_pool(
        generate_random_experiments(parameter_space, args.iterations),
        args,
        evaluator)
