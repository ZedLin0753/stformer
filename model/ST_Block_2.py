import torch
import torch.nn as nn
from .pos_embed import Pos_Embed

class STA_Block(nn.Module):
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
        padt = int((kernel_size[0] - 1) / 2)

        # Spatio-Temporal Tuples Attention
        if self.use_pes: 
            self.pes = Pos_Embed(in_channels, num_frames, num_joints)
        
        # 输出 Q、K、V
        self.to_qkvs = nn.Conv2d(in_channels, 3 * num_heads * qkv_dim, 1, bias=True)
        
        self.alphas = nn.Parameter(torch.ones(1, num_heads, 1, 1), requires_grad=True)
        self.att0s = nn.Parameter(torch.ones(1, num_heads, num_joints, num_joints) / num_joints, requires_grad=True)
        
        self.out_nets = nn.Sequential(
            nn.Conv2d(in_channels * num_heads, out_channels, (1, kernel_size[1]), padding=(0, pads)),
            nn.BatchNorm2d(out_channels)
        )
        
        self.ff_net = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        # Inter-Frame Feature Aggregation
        self.out_nett = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), padding=(padt, 0)),
            nn.BatchNorm2d(out_channels)
        )

        if in_channels != out_channels:
            self.ress = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
            self.rest = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.ress = lambda x: x
            self.rest = lambda x: x

        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(att_drop)

    def forward(self, x):
        N, C, T, V = x.size()
        
        
        # Spatio-Temporal Tuples Attention
        xs = self.pes(x) + x if self.use_pes else x
        print('xs.shape: ', xs.shape)
        
        # 使用 Q、K、V 
        qkv = self.to_qkvs(xs).view(N, 3 * self.num_heads, self.qkv_dim, T, V)  # 输出Q、K、V
        q, k, v = torch.chunk(qkv, 3, dim=1)  # 分离出Q、K、V
        print('q.shape: ', q.shape)
        print('k.shape: ', k.shape)
        print('v.shape: ', v.shape)
        
        attention_scores = torch.einsum('nhctu,nhctv->nhuv', [q, k]) / ((self.qkv_dim ** 0.5)*T)  # 注意力得分
        print('attention_score.shape: ', attention_scores.shape)
        attention_scores += self.att0s.repeat(N, 1, 1, 1)
        print('attention_score.shape: ', attention_scores.shape)

        attention_weights = self.tan(attention_scores) * self.alphas  # 应用激活函数和缩放因子
        attention_weights = self.drop(attention_weights)
        
        xs = torch.einsum('nhctu,nhuv->nhctv', [v, attention_weights])
        print('xs.shape: ', xs.shape)
        xs = xs.contiguous().view(N, self.num_heads * self.in_channels, T, V)  # 加权求和
        x_ress = self.ress(x)
        xs = self.relu(self.out_nets(xs) + x_ress)
        xs = self.relu(self.ff_net(xs) + x_ress)

        # Inter-Frame Feature Aggregation
        xt = self.relu(self.out_nett(xs) + self.rest(xs))

        return xt