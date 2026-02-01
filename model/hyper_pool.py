import torch
import torch.nn as nn
import torch.nn.functional as F
from .hypergraph import normalize_incidence


class HyperPoolStatic(nn.Module):
    """
    固定 25→E 分組 (可自行修改 GROUP)。
    """
    GROUP = {
        "head":      [2,3],
        "torso":     [0,1,20],
        "l_arm":     [4,5,6,7,21,22],
        "r_arm":     [8,9,10,11,23,24],
        "l_leg":     [12,13,14,15],
        "r_leg":     [16,17,18,19],
    }
    def __init__(self, V=25):
        super().__init__()
        H = torch.zeros(V, len(self.GROUP))
        for col, idxs in enumerate(self.GROUP.values()):
            H[idxs, col] = 1.
        self.register_buffer("H", H)  # (25,E)

    def forward(self, x):              # x: (B,C,T,25)
        B,C,T,V = x.shape
        H = self.H.to(x)
        x = x.permute(0,2,1,3).reshape(B*T, C, V)    # (B*T,C,V)
        parts = torch.matmul(x, H) / (H.sum(0)+1e-6) # (B*T,C,E)
        return parts.view(B, T, C, -1).permute(0,2,1,3)  # (B,C,T,E)

    def upsample(self, parts):         # (B,C,T,E) → (B,C,T,25)
        H = self.H.to(parts)
        W_up = H / (H.sum(0)+1e-6)
        B,C,T,E = parts.shape
        tmp = parts.permute(0,2,1,3).reshape(B*T, C, E)
        out = torch.matmul(tmp, W_up.t()).view(B, T, C, 25).permute(0,2,1,3)
        return out

class HyperPoolLearn(nn.Module):
    """
    Learnable Static ‑ 每個節點 Softmax 到 E 個超邊，訓練時自動學分組。
    """
    def __init__(self, V=25, E=8):
        super().__init__()
        self.H_raw = nn.Parameter(torch.randn(V, E))  # (25,E)

    def forward(self, x):
        B,C,T,V = x.shape
        H = torch.softmax(self.H_raw, dim=1)          # row‑wise softmax
        x_flat = x.permute(0,2,1,3).reshape(B*T, C, V)
        out = torch.matmul(x_flat, H)                 # (B*T,C,E)
        return out.view(B, T, C, -1).permute(0,2,1,3)

    def upsample(self, parts):
        H = torch.softmax(self.H_raw, dim=1)
        W_up = H / (H.sum(0)+1e-6)
        B,C,T,E = parts.shape
        tmp = parts.permute(0,2,1,3).reshape(B*T, C, E)
        out = torch.matmul(tmp, W_up.t()).view(B, T, C, 25).permute(0,2,1,3)
        return out


class HyperPoolDynamic(nn.Module):
    """
    Dynamic Hyper‑Pooling：利用當前特徵自適應產生 H。
    這裡用最簡單的 1×1 Conv → Softmax，效果已不錯。
    """
    def __init__(self, in_channels, V=25, E=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, E, 1)

    def forward(self, x):              # x: (B,C,T,V)
        B,C,T,V = x.shape
        # 利用特徵產生分配權重  α ∈ ℝ^{B,E,T,V}
        alpha = torch.softmax(self.conv(x).permute(0,2,3,1), dim=-1)  # (B,T,V,E)
        x_flat = x.permute(0,2,1,3)                                  # (B,T,C,V)
        out = torch.einsum('btcv,btve->btce', x_flat, alpha)         # sum over V
        return out.permute(0,2,1,3)                                  # (B,C,T,E)