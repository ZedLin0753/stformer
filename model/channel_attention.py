import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.full_connect = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        N, C, T, V = x.size()
        y = self.pool(x).view(N, C)
        y = self.full_connect(y).view(N, C, 1, 1)
        return x * y
    
class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1  # ensure odd kernel size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, T, V = x.size()
        y = self.avg_pool(x).squeeze(-1).squeeze(-1)  # (N, C)
        y = self.conv(y.unsqueeze(1)).squeeze(1)      # (N, C)
        y = self.sigmoid(y).view(N, C, 1, 1)
        return x * y