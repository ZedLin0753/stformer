import torch
import torch.nn as nn
from .pos_embed import Pos_Embed
from .channel_attention import SEBlock, ECABlock

class STA_Block(nn.Module):
    def __init__(self, in_channels, out_channels, qkv_dim,
                 num_frames, num_joints, num_heads,
                 kernel_size,SE=None, ECA=None, temporal_att=None, spatial_att=None,  use_pess=None, use_pest=None, att_drop=0):
        super().__init__()
        self.qkv_dim = qkv_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_pes = use_pess
        self.SE = SE
        self.ECA = ECA
        pads = int((kernel_size[1] - 1) / 2)
        padt = int((kernel_size[0] - 1) / 2)
        # Spatio-Temporal Tuples Attention
        if self.use_pes:
            self.pes = Pos_Embed(in_channels, num_frames, num_joints)
        self.to_qkvs = nn.Conv2d(
            in_channels, 2 * num_heads * qkv_dim, 1, bias=True)
        self.alphas = nn.Parameter(
            torch.ones(1, num_heads, 1, 1), requires_grad=True)
        self.att0s = nn.Parameter(
            torch.ones(1, num_heads, num_joints, num_joints) / num_joints,
            requires_grad=True
        )
        self.out_nets = nn.Sequential(
            nn.Conv2d(in_channels * num_heads, out_channels,
                      (1, kernel_size[1]), padding=(0, pads)),
            nn.BatchNorm2d(out_channels)
        )
        self.ff_net = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        # Inter-Frame Feature Aggregation
        self.out_nett = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      (kernel_size[0], 1), padding=(padt, 0)),
            nn.BatchNorm2d(out_channels)
        )
        if in_channels != out_channels:
            self.ress = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
            self.rest = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.ress = lambda x: x
            self.rest = lambda x: x
        
        # Channel Attention
        if self.SE:
            self.channel_att = SEBlock(out_channels, reduction=16)
        elif self.ECA:
            self.channel_att = ECABlock(out_channels)
        else:
            self.channel_att = None


        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(att_drop)

    def forward(self, x):
        N, C, T, V = x.size()
        # print('x.shape: ', x.shape)
        #  Spatio-Temporal Tuples Attention
        xs = self.pes(x) + x if self.use_pes else x
        # print('xs.shape: ', xs.shape)
        q, k = torch.chunk(
            self.to_qkvs(xs).view(N, 2 * self.num_heads, self.qkv_dim, T, V),
            2,
            dim=1
        )
        # print('q.shape: ', q.shape)
        attention = self.tan(torch.einsum(
            'nhctu,nhctv->nhuv', [q, k]) / (self.qkv_dim * T)) * self.alphas
        #  print('attention.shape: ', attention.shape)
        attention = attention + self.att0s.repeat(N, 1, 1, 1)
        attention = self.drop(attention)
        xs = torch.einsum('nctu,nhuv->nhctv', [x, attention])
        xs = xs.contiguous().view(N, self.num_heads * self.in_channels, T, V)
        x_ress = self.ress(x)
        xs = self.relu(self.out_nets(xs) + x_ress)
        xs = self.relu(self.ff_net(xs) + x_ress)
        xt = self.relu(self.out_nett(xs) + self.rest(xs))

        if self.channel_att is not None:
            xt = self.channel_att(xt)
        

        return xt