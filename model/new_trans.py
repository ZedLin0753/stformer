import torch
import torch.nn as nn
import torch.nn.functional as F
from .pos_embed import Pos_Embed

class TRANS_BLOCK(nn.Module):
    def __init__(self, in_channels, out_channels, qkv_dim,
                 num_frames, num_joints, num_heads,
                 kernel_size, use_pes=True, att_drop=0):
        super().__init__()
        self.qkv_dim = qkv_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_pes = use_pes
        pads = int((kernel_size[1] - 1) / 2)

        #Position Embedding
        if self.use_pes: self.pes = Pos_Embed(in_channels, out_channels, num_joints)


        #transfer to Q, K, V
        self.to_qkvs = nn.Linear(in_channels, 3 * num_heads * qkv_dim)

        self.out_nets = nn.Sequential(
            nn.Conv2d(in_channels * num_heads, out_channels, (1, kernel_size[1]), padding=(0, pads)),
            nn.BatchNorm2d(out_channels)
        )
        self.ff_net = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        if in_channels != out_channels:
            self.ress = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.Conv1d()
            )
        else:
            self.ress = lambda x: x

        self.drop = nn.Dropout(att_drop)

    def forward(self, x):
        #Input
        N, C, T, V = x.size()

        xs = self.pes(x) + x if self.use_pes else x

        # adjust the demension of input
        x_flattened = xs.permute(0, 2, 3, 1).contiguous().view(N * T * V, C)  # 調整形狀為 (N * T * V, C)
        
        # 生成 Q, K, V
        qkv = self.to_qkv(x_flattened).view(N, T, V, 3, self.num_heads, self.qkv_dim).permute(0, 3, 4, 1, 2, 5)  # 分割並調整形狀
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # 將 Q、K、V 分開

        # 將 Q 和 K 展開並進行矩陣乘法
        q = q.permute(0, 1, 3, 4, 2).contiguous().view(N * self.num_heads, T * V, self.qkv_dim)  # 變形為 (N * H, TV, qkv_dim)
        k = k.permute(0, 1, 3, 2, 4).contiguous().view(N * self.num_heads, self.qkv_dim, T * V)  # 變形為 (N * H, qkv_dim, TV)
        attention = torch.matmul(q, k) / (self.qkv_dim ** 0.5)  # QK^T / sqrt(d)
        attention = F.softmax(attention, dim=-1)
        attention = self.drop(attention)

        # 將注意力分數應用於 V
        v = v.permute(0, 1, 3, 4, 2).contiguous().view(N * self.num_heads, T * V, self.qkv_dim)  # 變形為 (N * H, TV, qkv_dim)
        xs = torch.matmul(attention, v)  # (N * H, TV, qkv_dim)
        xs = xs.view(N, self.num_heads, T, V, self.qkv_dim).permute(0, 1, 4, 2, 3).contiguous()  # 調整回 (N, H, qkv_dim, T, V)
        xs = xs.view(N, self.num_heads * self.in_channels, T, V)  # 展開為輸出形狀

        # 殘差連接和前饋網路
        x_ress = self.ress(x)
        xs = F.relu(self.out_nets(xs) + x_ress)
        xs = F.relu(self.ff_net(xs) + xs)

        return xs

