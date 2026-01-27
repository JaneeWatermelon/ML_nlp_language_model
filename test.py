import torch
from torch import nn

m = nn.Dropout(p=0.2)
input = torch.randn(5, 5)
output = m(input)
print(input)
print(output)