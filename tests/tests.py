import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

t = [
    0, 1, 0, 4, 5,
    2, 3, 2, 1, 3,
    4, 4, 0, 4, 3,
    2, 5, 2, 6, 4,
    1, 0, 0, 5, 7
]
t = torch.tensor(t, dtype=torch.float)
t = torch.reshape(t, (1, 1, 5, 5))

kernel=[2, 2]
weights = torch.ones([1, 1]+kernel, dtype=torch.float)
biases = torch.zeros((1,), dtype=torch.float)

conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=tuple(kernel), stride=(2, 2), padding=(1,1))

with torch.no_grad():
    conv1.weight = nn.Parameter(weights)
    conv1.bias = nn.Parameter(biases)

# Compute
x = conv1(t)
x = x.detach().numpy()
x = x.flatten()
print(list(x))
asd = 3