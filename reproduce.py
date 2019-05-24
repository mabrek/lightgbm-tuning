#!/usr/bin/env python

from multiprocessing import Lock
import argparse

import numpy as np

from utils import read_json_log, read_telecom_churn, run_pool,\
    evaluate_parameters, log_json

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Reproduce recorded parameters')
    parser.add_argument('--input-log', required=True)
    parser.add_argument('--output-log', required=True)
    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument('--chunksize', type=int, default=10)
    args = parser.parse_args()

    log_lock = Lock()
    folds, validation, whole_train = read_telecom_churn()

    input_log = read_json_log(args.input_log)

    def generator():
        for _, row in input_log.iterrows():
            parameters = row.filter(regex='^param_')\
                            .rename(lambda x: x.replace('param_', ''))\
                            .to_dict()
            if np.isnan(parameters['scale_pos_weight']):
                parameters['scale_pos_weight'] = None
            yield (row['name'],
                   row['experiment_id'],
                   parameters)

    def evaluator(experiment):
        name, experiment_id, parameters = experiment
        log_data = {}
        log_data['name'] = name
        log_data['experiment_id'] = experiment_id
        log_data.update({'param_' + k: v for k, v in parameters.items()})
        metrics = evaluate_parameters(parameters, folds, validation,
                                      whole_train, num_boost_round=500)
        metrics['success'] = True
        log_data.update(metrics)
        with log_lock:
            log_json(args.output_log, log_data)

    run_pool(generator(), args, evaluator)
