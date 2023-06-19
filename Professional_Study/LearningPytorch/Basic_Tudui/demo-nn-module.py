import torch
from torch import nn


class Wu(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

wu = Wu()
x = torch.tensor(1.0)
output = wu(x)
print(output)