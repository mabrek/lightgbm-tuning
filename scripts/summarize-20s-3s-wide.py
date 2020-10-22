from utils import *

summarize_to_chunks('experiments/wide-20shuffle-3seed.log.xz', 'experiments/wide-20shuffle-3seed-', 20, verbose=True)
df = aggregate_chunks('experiments/wide-20shuffle-3seed-???.pkl')
df.to_pickle('experiments/wide-20shuffle-3seed.pkl')
