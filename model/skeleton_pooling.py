import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP = {
    "head" :[2, 3], 
    "torso" :[0, 1, 20],
    "left_arm" :[4, 5, 6, 7, 21, 22],
    "right_arm" :[8, 9, 10, 11, 23, 24],
    "left_leg" :[12, 13, 14, 15],
    "right_leg" :[16, 17, 18, 19]
}

def build_incidence(device=None):
    H = torch.zeros(25, 6, device=device)
    for p, idxs in enumerate(GROUP.values()):
        H[idxs, p] = 1.
    print(H)
    return H   #(25, 6)

class StaticGroupWeightPool(nn.Module):
    def __init__(self, init_std: float = 0.01):
        super().__init__()
        self.register_buffer("H_mask", build_incidence())
        W = torch.ones_like(self.H_mask) + init_std*torch.randn_like(self.H_mask)
        self.W = nn.Parameter(W)  #learnable weights

    #25 -> 6
    def forward(self, x):  # x:(B,C,25) or (B,C,T,25)
        if x.dim() == 4:
            B0, C0, T0, V0 = x.shape
            x2 = x.permute(0, 2, 1, 3).reshape(B0*T0, C0, V0)
        else:
            B0, T0 = x.shape[0], 1
            x2 = x
        H = self.H_mask * self.W  #(25, 6)
        parts = torch.matmul(x2, H) / H.sum(0).clamp_min(1e-6) # (B*T, C, 6)
        parts = parts.reshape(B0, T0, C0, 6).permute(0, 2, 1, 3) # (B, C, T, 6)
        return parts
    
    #6 -> 25
    def upsample(self, parts):
        H = self.H_mask * self.W  #(25, 6)
        W_up = H / H.sum(0).clamp_min(1e-6)  # (25, 6)
        B1, C1, T1, P1 = parts.shape
        x3 = parts.permute(0, 2, 1, 3).reshape(B1*T1, C1, P1)  # (B*T, C, P)
        up = torch.matmul(x3, W_up.T).reshape(B1, T1, C1, 25).permute(0, 2, 1, 3)  # (B, C, T, 25)
        return up
