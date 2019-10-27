#!/usr/bin/env python

from utils import *

summarize_to_chunks('experiments/overfit-true-10k.log', 'experiments/overfit-true-10k-', 5, chunksize=500, verbose=True)
df = aggregate_chunks('experiments/overfit-true-10k-???.pkl')
df.to_pickle('experiments/overfit-true-10k.pkl')
