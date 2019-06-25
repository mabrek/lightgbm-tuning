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

        'bagging_fraction': uniform(loc=0.9287591331277941, scale=0.07124086687220588),
        'bagging_freq': randint(0, 8),
        'bin_construct_sample_cnt': randint(3355, 7652),
        'cat_l2': loguniform(low=-4.914233959561014, high=5.454017036321304, base=10),
        'cat_smooth': loguniform(low=-4.012747598147916, high=4.830227942667468, base=10),
        'feature_fraction': uniform(loc=0.39824195739607804, scale=0.371582892101745),
        'lambda_l1': loguniform(low=-7.030818530779058, high=-0.94983071304177, base=10),
        'lambda_l2': loguniform(low=-5.558030603231208, high=2.5586664697473793, base=10),
        'learning_rate': loguniform(low=-5.39817703836328, high=-0.10934704016171805, base=10),
        'max_bin': randint(457, 1445),
        'max_cat_threshold': randint(488, 1477),
        'max_cat_to_onehot': randint(26, 79),
        'max_delta_step': loguniform(low=-1.207194065517524, high=4.10065031033168, base=10),
        'max_depth': randint(84, 292),
        'min_data_in_bin': randint(427, 3325),
        'min_data_in_leaf': randint(237, 856),
        'min_data_per_group': randint(962, 3097),
        'min_gain_to_split': loguniform(low=-7.5265973221927425, high=-2.7505834252425427, base=10),
        'min_sum_hessian_in_leaf': loguniform(low=-7.071784017222152, high=-0.37707683233134637, base=10),
        'num_leaves': randint(1111, 3093),
        'scale_pos_weight': uniform(loc=1.017324295134583, scale=4.946012583757598),
    }

    args = parse_args()
    log_lock = Lock()
    X_train, X_val, y_train, y_val, folds = read_telecom_churn(args.n_folds)

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
