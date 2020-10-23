from utils import *

summarize_to_chunks(
    "experiments/better.log", "experiments/better-", 20, verbose=True
)
df = aggregate_chunks("experiments/better-???.pkl")
df.to_pickle("experiments/better.pkl")
