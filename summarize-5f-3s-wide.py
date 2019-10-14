from utils import *

summarize_to_chunks('experiments/wide-5fold-3seed.log', 'experiments/wide-5fold-3seed-', 5, verbose=True)
df = aggregate_chunks('experiments/wide-5fold-3seed-???.pkl')
df.to_pickle('experiments/wide-5fold-3seed.pkl')
