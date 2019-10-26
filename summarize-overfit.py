#!/usr/bin/env python

from utils import *

summarize_to_chunks('experiments/overfit-10k.log', 'experiments/overfit-10k-', 5, chunksize=500, verbose=True)
df = aggregate_chunks('experiments/overfit-10k-???.pkl')
df.to_pickle('experiments/overfit-10k.pkl')
