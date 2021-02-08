import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

image = [[
    [[-0.66, 1.88, -0.09, 2.00, -1.26],
[-0.96, 1.49, -0.34, -0.12, -0.09],
[-0.19, -0.60, -1.60, -0.84, -1.44],
[-0.83, -0.06, 0.01, -0.81, -0.90],
[0.43, 0.82, -0.46, -0.10, -0.17]],

[[0.32, -1.09, 0.52, 1.19, 0.76],
[0.16, 1.07, -1.08, 0.14, -2.00],
[0.94, 1.24, 0.23, -0.77, 0.23],
[0.09, -1.64, 2.31, 0.09, 0.98],
[0.23, -2.14, -1.47, 1.18, -0.02]],

[[0.75, -0.97, 0.47, 0.67, -0.03],
[0.77, 0.27, 1.16, 0.62, 1.39],
[-0.23, 0.51, 0.26, 0.75, -0.08],
[0.14, 0.17, 0.89, 0.19, -0.44],
[0.98, 0.66, 0.48, -0.20, 0.10]]
      ]]

a2 = np.array(image)
b2 = np.array(image)

a2 = a2.reshape((3, 25))
b2 = b2.reshape((3, 25))

a2 = np.mean(a2, axis=1)
b2 = np.var(b2, axis=1)
b3 = 1/np.sqrt(b2 + 1e-5)

channels = 3
m = nn.BatchNorm2d(channels, affine=False)
_input = torch.tensor(image, dtype=torch.float)
# _input = torch.randn(1, channels, 5, 5)
output = m(_input)

a = np.array(_input.cpu())
b = np.array(output.cpu())

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
print(a)
print("-------------------")
print(b)
asd = 3