from utils import *

summarize_to_chunks('experiments/good-20shuffle-3seed.log', 'experiments/good-20shuffle-3seed-', 20, verbose=True)
df = aggregate_chunks('experiments/good-20shuffle-3seed-???.pkl')
df.to_pickle('experiments/good-20shuffle-3seed.pkl')
