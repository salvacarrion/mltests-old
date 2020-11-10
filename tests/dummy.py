import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

src_freqs = np.load("src_freqs.npy")
trg_freqs = np.load("trg_freqs.npy")

data = src_freqs
# evaluate the histogram
#evaluate the cumulative
cumulative = np.cumsum(data)
cumulative_per = 100*cumulative/cumulative[-1]

top_k_range = [1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
for top_k in top_k_range:
    c_per = cumulative_per[:top_k]
    plt.plot(range(len(c_per)), c_per, c='blue')
    plt.savefig("vocab_topk_{}_{:.2f}%.eps".format(top_k, c_per[-1]))
    print("With the top {} most common words, we will cover {:.2f}% of the words".format(top_k, c_per[-1]))

asde = 3