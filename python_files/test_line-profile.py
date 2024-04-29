# %%
%load_ext line_profiler
import numpy as np

def bad_sum(N):
    i = 0
    j = 1
    for _ in range(N):
        i = i+1

    print(i)

def good_sum(N):
    i = np.arange(N).sum()

    print(i)

def both(N):
    bad_sum(N)
    good_sum(N)

N = 10
%lprun -f bad_sum bad_sum(N)
%lprun -f good_sum good_sum(N)
%lprun -f both both(N)


# %%
