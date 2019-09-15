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
    top_mean_dev_auc, top_min_whole_validation_auc, top_min_dev_auc, read_files
# -

rcParams['figure.figsize'] = 20, 5
pd.set_option('display.max_columns', None)

# %load_ext autoreload
# %autoreload 2

files = ['./experiments/wide-20shuffle-3seed.pkl']

top_k = 100000
min_iterations = 50

best_mean_dev = top_mean_dev_auc(read_files(files), top_k)\
    .pipe(lambda x: x[x.groupby(['file', 'experiment_id']).iteration.transform('size') > min_iterations])
best_mean_dev.groupby(['file', 'experiment_id']).ngroups

best_min_dev = top_min_dev_auc(read_files(files), top_k)\
    .pipe(lambda x: x[x.groupby(['file', 'experiment_id']).iteration.transform('size') > min_iterations])
best_min_dev.groupby(['file', 'experiment_id']).ngroups

true_best = top_min_whole_validation_auc(read_files(files), top_k)\
    .pipe(lambda x: x[x.groupby(['file', 'experiment_id']).iteration.transform('size') > min_iterations])
true_best.groupby(['file', 'experiment_id']).ngroups

compare_columns = ['mean_dev_auc', 'mean_validation_auc',
               'mean_whole_validation_auc', 'min_dev_auc', 'min_validation_auc', 'min_whole_validation_auc']

best_mean_dev[compare_columns].describe().T

best_mean_dev.min_whole_validation_auc.corr(best_mean_dev.mean_dev_auc, method='spearman')

best_mean_dev.mean_whole_validation_auc.corr(best_mean_dev.mean_dev_auc, method='spearman')

best_min_dev[compare_columns].describe().T

best_min_dev.min_whole_validation_auc.corr(best_min_dev.min_dev_auc, method='spearman')

best_min_dev.min_whole_validation_auc.corr(best_min_dev.mean_dev_auc, method='spearman')

true_best[compare_columns].describe().T

true_best.min_whole_validation_auc.corr(true_best.min_dev_auc, method='spearman')

true_best.min_whole_validation_auc.corr(true_best.mean_dev_auc, method='spearman')

true_best.mean_whole_validation_auc.corr(true_best.min_dev_auc, method='spearman')

true_best.mean_whole_validation_auc.corr(true_best.mean_dev_auc, method='spearman')

best_mean_dev.merge(true_best, on=['file', 'experiment_id', 'iteration'])\
    .groupby(['file', 'experiment_id']).ngroups

best_mean_dev.merge(true_best, on=['file', 'experiment_id', 'iteration']).shape

best_min_dev.merge(true_best, on=['file', 'experiment_id', 'iteration'])\
    .groupby(['file', 'experiment_id']).ngroups

best_min_dev.merge(true_best, on=['file', 'experiment_id', 'iteration']).shape

best = best_mean_dev

best.shape

best.groupby(SET_PARAMETERS).min_whole_validation_auc.describe()

best.groupby(SET_PARAMETERS + ['file', 'experiment_id']).size().groupby(SET_PARAMETERS).size()

# +
best['param_bagging_enable'] = (best.param_bagging_freq != 0)

best.groupby('param_bagging_enable').size()
# -

best.min_whole_validation_auc.hist(bins=100);

true_best.min_whole_validation_auc.hist(bins=100);

best_iteration = best.sort_values('mean_dev_auc').groupby(['file', 'experiment_id']).last()

pd.plotting.scatter_matrix(
    pd.concat([best_iteration[CONT_PARAMETERS + INT_PARAMETERS], 
               np.log10(best_iteration[list(set(LOG_PARAMETERS))])],
              axis='columns',
              sort=True
             ).rename(lambda x: x.replace('param_', ''), axis='columns').sort_index(axis=1),
    alpha=1, figsize=(50, 50), hist_kwds={'bins': 50});

# ### check failures in best selected parameter range

best_range = best\
    [[c for c in best.columns if c.startswith('param_') and not c in SET_PARAMETERS]]\
    .drop(columns=['param_seed', 'param_bagging_enable'])\
    .quantile([0, 1]).T
best_range

# +
dfs = []
for f in files:
    df = pd.read_pickle(f).assign(file=f)
    for (n, l, h) in best_range.itertuples():
        df = df[(((df[n] >= l) & (df[n] <= h))
                 | df[n].isna())].copy()
    dfs.append(df)

limited = pd.concat(dfs, ignore_index=True, sort=True)
del dfs
limited.shape
# -

limited_best_iteration = limited.sort_values('min_whole_validation_auc').groupby(['file', 'experiment_id']).last()
limited_best_iteration.shape

limited_best_iteration['param_bagging_enable'] = (limited_best_iteration.param_bagging_freq != 0)

limited_best_iteration.min_whole_validation_auc.describe()

limited_bad = limited_best_iteration[limited_best_iteration.min_whole_validation_auc < 0.6]

limited_bad.shape

# TODO pikachu with bruises reaction

limited_bad.groupby(SET_PARAMETERS).size()

limited_bad.groupby('param_bagging_enable').size()

limited_best_iteration.groupby(SET_PARAMETERS).min_whole_validation_auc.describe()

pd.plotting.scatter_matrix(
    pd.concat([limited_bad[CONT_PARAMETERS + INT_PARAMETERS], 
               np.log10(limited_bad[list(set(LOG_PARAMETERS))])],
              axis='columns',
              sort=True
             ).rename(lambda x: x.replace('param_', ''), axis='columns').sort_index(axis=1),
    alpha=0.8, figsize=(50, 50), hist_kwds={'bins': 50});

# ### top parameters

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(best_iteration.sort_values('mean_dev_auc', ascending=False).head().T)


