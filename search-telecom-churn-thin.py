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
        'metric': [['binary_logloss', 'auc', 'map', 'binary_error']],
        'eval_at': [EVAL_AT],
        'verbosity': [-1],

        'bagging_enable': [False],
        'boost_from_average': [False],
        'is_unbalance': [True],

        'bin_construct_sample_cnt': randint(3357, 7645),
        'cat_l2': loguniform(low=-4.886096861957528, high=5.4365807176181855, base=10),
        'cat_smooth': loguniform(low=-4.009517873392765, high=4.81807269806813, base=10),
        'feature_fraction': uniform(loc=0.398439335534611, scale=0.37058744001234006),
        'lambda_l1': loguniform(low=-7.029977194193349, high=-0.9548637530058061, base=10),
        'lambda_l2': loguniform(low=-5.536631605374113, high=2.4622076214533113, base=10),
        'learning_rate': loguniform(low=-2.3281646891804666, high=-2, base=10),
        'max_bin': randint(458, 1440),
        'max_cat_threshold': randint(491, 1463),
        'max_cat_to_onehot': randint(26, 78),
        'max_delta_step': loguniform(low=0, high=4.062618590539585, base=10),
        'max_depth': randint(84, 290),
        'min_data_in_bin': randint(428, 953),
        'min_data_in_leaf': randint(450, 855),
        'min_data_per_group': randint(964, 3096),
        'min_gain_to_split': loguniform(low=-7.5192986266532795, high=-2.791566771390993, base=10),
        'min_sum_hessian_in_leaf': loguniform(low=-7.071165914217148, high=-0.41988311890357377, base=10),
        'num_leaves': randint(1111, 3088),
    }

    args = parse_args()
    log_lock = Lock()
    X_train, X_val, y_train, y_val, folds = read_telecom_churn()

    # captures data and lock to use in forked processes
    def evaluator(experiment):
        evaluate_experiment(
            experiment,
            experiment_name=args.name,
            log_file=args.log,
            log_lock=log_lock,
            num_boost_round=500,
            X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
            folds=folds
        )

    run_pool(
        generate_random_experiments(parameter_space, args.iterations),
        args,
        evaluator)
