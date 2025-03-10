import torch
import torch.nn as nn

a = torch.randn(7)
print(a)
print(a.size())
b = nn.AvgPool1d(1)(a)
print(b)
