# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import glob

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
# %matplotlib inline

from pylab import rcParams

from utils import METRICS, CONT_PARAMETERS, LOG_PARAMETERS, SET_PARAMETERS, INT_PARAMETERS, read_files,\
    top_mean_dev_auc, top_min_whole_validation_auc, top_min_dev_auc, read_files, LOGREG_LOG_PARAMETERS
# -

rcParams['figure.figsize'] = 20, 5
pd.set_option('display.max_columns', None)

# %load_ext autoreload
# %autoreload 2

files = ['./experiments/logreg.pkl']

# +
top_k = 1000

best_mean_dev = top_mean_dev_auc(read_files(files), top_k)
best_min_dev = top_min_dev_auc(read_files(files), top_k)
true_best = top_min_whole_validation_auc(read_files(files), top_k)
# -

compare_columns = ['mean_dev_auc', 'mean_validation_auc',
               'mean_whole_validation_auc', 'max_overfit_auc',
                   'min_dev_auc', 'min_validation_auc', 'min_whole_validation_auc']

best_mean_dev[compare_columns].describe().T

best_min_dev[compare_columns].describe().T

true_best[compare_columns].describe().T

best_mean_dev.merge(true_best, on=['file', 'experiment_id'])\
    .groupby(['file', 'experiment_id']).ngroups

best_mean_dev.merge(true_best, on=['file', 'experiment_id']).shape

best_min_dev.merge(true_best, on=['file', 'experiment_id'])\
    .groupby(['file', 'experiment_id']).ngroups

best_min_dev.merge(true_best, on=['file', 'experiment_id']).shape

best = best_mean_dev

best.shape

best.min_whole_validation_auc.hist(bins=100);

true_best.min_whole_validation_auc.hist(bins=100);

best.groupby('param_clf__fit_intercept').size()

pd.plotting.scatter_matrix(
    pd.concat([np.log10(best[list(set(LOGREG_LOG_PARAMETERS))])],
              axis='columns',
              sort=True
             ).rename(lambda x: x.replace('param_clf__', ''), axis='columns').sort_index(axis=1),
    alpha=1, figsize=(25, 25), hist_kwds={'bins': 50});

# ### top parameters

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(best.sort_values('mean_dev_auc', ascending=False).head().T)


