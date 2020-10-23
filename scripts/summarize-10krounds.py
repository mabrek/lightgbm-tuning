#!/usr/bin/env python

from utils import *

summarize_to_chunks(
    "experiments/wide-10krounds-5folds.log.xz",
    "experiments/wide-10krounds-5folds-",
    5,
    chunksize=500,
    verbose=True,
)
df = aggregate_chunks("experiments/wide-10krounds-5folds-???.pkl")
df.to_pickle("experiments/wide-10krounds-5folds.pkl")
