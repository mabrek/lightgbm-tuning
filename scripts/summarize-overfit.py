#!/usr/bin/env python

from lightgbm_tuning import *

summarize_to_chunks(
    "experiments/overfit.log",
    "experiments/overfit-",
    5,
    chunksize=500,
    verbose=True,
)
df = aggregate_chunks("experiments/overfit-???.pkl")
df.to_pickle("experiments/overfit.pkl")
