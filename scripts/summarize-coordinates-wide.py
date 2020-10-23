from utils import *

summarize_to_chunks(
    "experiments/wide-best-coordinates.log",
    "experiments/wide-best-coordinates-",
    20,
    verbose=True,
)
df = aggregate_chunks("experiments/wide-best-coordinates-???.pkl")
df.to_pickle("experiments/wide-best-coordinates.pkl")
