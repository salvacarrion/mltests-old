import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

t = [
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 2, 1, 0, 1, 0,
    0, 2, 2, 1, 0, 0, 0,
    0, 0, 0, 2, 0, 1, 0,
    0, 0, 2, 1, 2, 0, 0,
    0, 2, 2, 0, 2, 0, 0,
    0, 0, 0, 0, 0, 0, 0,

    0, 0, 0, 0, 0, 0, 0,
    0, 2, 0, 2, 1, 2, 0,
    0, 0, 2, 0, 1, 0, 0,
    0, 1, 2, 0, 2, 2, 0,
    0, 0, 0, 2, 1, 2, 0,
    0, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0,

    0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 2, 0, 0, 0,
    0, 2, 0, 2, 0, 0, 0,
    0, 2, 1, 1, 2, 1, 0,
    0, 0, 2, 0, 1, 2, 0,
    0, 1, 1, 1, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0,
]
t = torch.tensor(t, dtype=torch.float)
t = torch.reshape(t, (1, 3, 7, 7))

weights = [
    -1, 1, 0,
    1, -1, 0,
    -1, -1, -1,

    -1, 0, 0,
    -1, 0, 1,
    -1, -1, 0,

    -1, 0, -1,
    0, 0, 0,
    1, 0, 0,

    0, -1, 0,
    -1, 0, 1,
    1, 0, -1,

    0, 0, -1,
    1, 0, 0,
    1, 0, -1,

    1, 1, 0,
    0, 0, 1,
    0, 0, 0,
]
weights = torch.tensor(weights, dtype=torch.float)
weights = torch.reshape(weights, (2, 3, 3, 3))

biases = [1, 0]
biases = torch.tensor(biases, dtype=torch.float)
biases = torch.reshape(biases, (2,))

conv1 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))

with torch.no_grad():
    conv1.weight = nn.Parameter(weights)
    conv1.bias = nn.Parameter(biases)

# Compute
x = conv1(t)
x = x.detach().numpy()
x = x.flatten()
print(list(x))
asd = 3