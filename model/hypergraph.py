import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_incidence(H, eps=1e-6):
    """
    返回超圖卷積核  D_v^{-½} H W_e D_e^{-1} H^T D_v^{-½}
    若 W_e = I，則 W_e 可省略。
    """
    # H: 超圖的 incidence matrix
    # eps: 防止除以零的常數
    V, E = H.size() # 頂點數量 V 和邊數量 E
    D_v = torch.sum(H, dim=1)  # 頂點的度數
    D_e = torch.sum(H, dim=0)  # 邊的度數

    D_v_inv_sqrt = torch.diag(torch.pow(D_v + eps, -0.5)) #(V, V)
    D_e_inv = torch.diag(torch.pow(D_e + eps, -1)) #(E, E)

    H_norm = D_v_inv_sqrt @ H @ D_e_inv # (V, E) @ (E, E) = (V, E)
    return H_norm @ H_norm.t()  # 返回 H_norm @ H_norm^T    (V, V)


class HypergraphConv(nn.Module):
    """
    HGConv  (V,E) 參照 HGNN / DH‑GCN.
    X': (B,C_out,T,V)
    """
    def __init__(self, V, E, in_channels, out_channels, H, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.H = nn.Parameter(torch.randn(V, E)) #learnable incidence matrix
        
        self.theta = nn.Conv2d(self.in_channels, self.out_channels, 1, bias=bias)

    def forward(self, x):  # x: (B, C_in, T, V)
        B, C, T, V = x.size()
        H_norm = normalize_incidence(torch.softmax(self.H, dim=1))  # (V,V)
        # 把 HG 卷積寫成一次 matmul： X @ H_norm
        x = x.permute(0,2,1,3).reshape(B*T, C, V)       # (B*T,C,V)
        x = torch.matmul(x, H_norm).reshape(B, T, C, V) # (B,T,C,V)
        x = x.permute(0,2,1,3)                          # (B,C,T,V)
        return self.theta(x)