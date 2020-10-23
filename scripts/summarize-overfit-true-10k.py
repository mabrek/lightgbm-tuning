#!/usr/bin/env python

from utils import *

summarize_to_chunks(
    "experiments/overfit-true-10k-3.log.xz",
    "experiments/overfit-true-10k-3-",
    5,
    chunksize=250,
    verbose=True,
)
df = aggregate_chunks("experiments/overfit-true-10k-3-???.pkl")
df.to_pickle("experiments/overfit-true-10k-3.pkl")
