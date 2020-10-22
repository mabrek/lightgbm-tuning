from utils import *

summarize_to_chunks('experiments/good-5fold-3seed.log.xz', 'experiments/good-5fold-3seed-', 5, verbose=True)
df = aggregate_chunks('experiments/good-5fold-3seed-???.pkl')
df.to_pickle('experiments/good-5fold-3seed.pkl')
