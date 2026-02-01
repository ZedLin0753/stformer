import torch
import torch.nn as nn
from .position_embed import Pos_Embed






class STA_Block(nn.Module):
    def __init__(self, in_channels, out_channels, qkv_dim,
                 num_frames, num_joints, num_heads,
                 kernel_size,temporal_att=None, spatial_att=None,  use_pess=None, use_pest=None, att_drop=0):
        super().__init__()
        self.qkv_dim = qkv_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_pess = use_pess
        self.use_pest = use_pest
        self.temporal_att = temporal_att
        self.spatial_att = spatial_att
        pads = int((kernel_size[1] - 1) / 2)
        padt = int((kernel_size[0] - 1) / 2)

        #Spatial Attention
        if self.spatial_att:
            if self.use_pess:
                self.pes = Pos_Embed(in_channels, num_frames, num_joints, domain='spatial')
            self.to_qkvs = nn.Conv2d(in_channels, 2 * num_heads * qkv_dim, 1, bias=True)
            self.alphas = nn.Parameter(torch.ones(1, num_heads, 1, 1), requires_grad=True)
            self.att0s = nn.Parameter(torch.ones(1, num_heads, num_joints, num_joints) / num_joints, requires_grad=True)
            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels * num_heads, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
            self.ff_nett = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, kernel_size[1]), padding=(0, 1)),
                nn.BatchNorm2d(out_channels)
            )
        #Temporal Attention
        if self.temporal_att:
            if self.use_pest:
                self.pet = Pos_Embed(out_channels, num_frames, num_joints, domain='temporal')
            self.to_qkvt = nn.Conv2d(out_channels, 2 * num_heads * qkv_dim, 1, bias=True)
            self.alphat = nn.Parameter(torch.ones(1, num_heads, 1, 1), requires_grad=True)
            self.att0t = nn.Parameter(torch.ones(1, num_heads, num_frames, num_frames) + torch.eye(num_frames), requires_grad=True)
            #self.att0t = nn.Parameter(torch.ones(1, num_heads, num_frames, num_frames) / num_frames, requires_grad=True)

            self.out_nett = nn.Sequential(
                nn.Conv2d(out_channels * num_heads, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
            self.ff_nets = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.out_nett = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), padding=(padt, 0)),
                nn.BatchNorm2d(out_channels)
            )
    
        #Residual Connection
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

        self.tan = nn.Tanh()
        #self.tan = nn.Softmax()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(att_drop)

    def forward(self, x):
        N, C, T, V = x.size() # N: people, C: channels, T: frames, V: joints

        #Spatial Attention
        if self.spatial_att:
            if self.use_pess:
                xs = self.pes(x) + x # (N, C, T, V)
            else:
                xs = x # (N, C, T, V)
            q, k = torch.chunk(
                self.to_qkvs(xs).view(N, 2 * self.num_heads, self.qkv_dim, T, V),
                2,
                dim=1
            )# (N, 2 * num_heads, qkv_dim, T, V)
            attention = self.tan(torch.einsum(
                'nhctu,nhctv->nhuv', [q, k]) / (self.qkv_dim * T)) * self.alphas # (N, H, T, V)
            #attention = attention + self.att0s.repeat(N, 1, 1, 1)
            attention = attention + self.att0s[:, :, :V, :V].repeat(N, 1, 1, 1) # (N, H, T, V)
            attention = self.drop(attention)
            xs = torch.einsum('nctu,nhuv->nhctv', [x, attention]) # (N, H, C, T, V)
            xs = xs.contiguous().view(N, self.num_heads * self.in_channels, T, V) # (N, H*C, T, V)
            x_ress = self.ress(x) # (N, C, T, V)
            
            xs = self.relu(self.out_nets(xs) + x_ress) # (N, C, T, V)
            xs = self.relu(self.ff_nets(xs) + x_ress) # (N, C, T, V)
        else:
            xs = self.out_nets(x)
            xs = self.relu(xs)
        #Temporal Attention
        y = xs # (N, C, T, V)
        if self.temporal_att:
            if self.use_pest:
                xt = self.pet(y) + y # (N, C, T, V)
            else:
                xt = y # (N, C, T, V)
            q, k = torch.chunk(
                self.to_qkvt(xt).view(N, 2 * self.num_heads, self.qkv_dim, T, V),
                2,
                dim=1
            ) # (N, 2 * num_heads, qkv_dim, T, V)
            attention = self.tan(torch.einsum(
                'nhctv,nhcqv->nhtq', [q, k]) / (self.qkv_dim * V)) * self.alphat # (N, H, T, T)
            #attention = attention + self.att0t.repeat(N, 1, 1, 1)
            attention = attention + self.att0t[:, :, :T, :T].repeat(N, 1, 1, 1) # (N, H, T, T)
            attention = self.drop(attention) 
            xt = torch.einsum('nctv,nhtq->nhcvq', [y, attention]) # (N, H, C, V, T)
            xt = xt.contiguous().view(N, self.num_heads * self.out_channels, T, V) # (N, H*C, T, V)
            x_rest = self.rest(y)  # (N, C, T, V)
            xt = self.relu(self.out_nett(xt) + x_rest) # (N, C, T, V)
            xt = self.relu(self.ff_nett(xt) + x_rest ) # (N, C, T, V)
            #xt = self.relu(self.out_nett(xt) + x_rest + x_ress) # (N, C, T, V)
            #xt = self.relu(self.ff_nett(xt) + x_rest + x_ress) # (N, C, T, V)
        else:
            xt = self.out_nett(xs)
            xt = self.relu(xt)

        return xt
