from lightgbm_tuning import *

summarize_to_chunks(
    "experiments/better-best-coordinates.log",
    "experiments/better-best-coordinates-",
    20,
    verbose=True,
)
df = aggregate_chunks("experiments/better-best-coordinates-???.pkl")
df.to_pickle("experiments/better-best-coordinates.pkl")
