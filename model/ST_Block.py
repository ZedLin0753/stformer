import torch
import torch.nn as nn
import torch.nn.functional as F
from .pos_embed import Pos_Embed


class SpatioTemporalTransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, qkv_dim,
                 num_frames, num_joints, num_heads,
                 kernel_size, dropout=0, use_pes=True):
        """
        :param in_channels: 输入通道數
        :param out_channels: 输出通道數
        :param qkv_dim: Q/K/V 的维度
        :param num_frames: 時間幀數
        :param num_joints: 關節數
        :param num_heads: 注意力頭數
        :param kernel_size: 局部卷積核(時間, 空間)
        :param dropout: Dropout 比例
        :param use_pes: 是否使用位置嵌入
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.qkv_dim = qkv_dim
        self.num_heads = num_heads
        self.use_pes = use_pes
        self.num_joints = num_joints
        self.num_frames = num_frames
        pads = int((kernel_size[1] - 1) / 2)
        padt = int((kernel_size[0] - 1) / 2)
        self.dropout = nn.Dropout(dropout)

        # 位置編碼
        if self.use_pes: self.pes = Pos_Embed(in_channels, num_frames, num_joints)

        # QKV 生成
        self.qkv_proj = nn.Conv2d(in_channels, 3 * num_heads * qkv_dim, kernel_size=1)
        self.attention_dropout = nn.Dropout(dropout)

        #可學習參數
        self.alphas = nn.Parameter(torch.ones(1, num_heads, 1, 1), requires_grad=True)
        self.att0s = nn.Parameter(torch.ones(1, num_heads, num_joints, num_joints) / num_joints, requires_grad=True)

        # 注意力输出层
        self.attention_output = nn.Conv2d(num_heads * qkv_dim, out_channels, kernel_size=1)

        # 時間卷積
        self.time_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size[0], 1), padding=(padt, 0))
        # 空間卷積
        self.spatial_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size[1]), padding=(0, pads))

        # 殘差連接
        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = lambda x: x

        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.tan = nn.Tanh()

    def forward(self, x):

        
        """
        :param x: 輸入張量, 形狀為 (N, C, T, V)
        :return: 輸出張量, 形狀為 (N, out_channels, T, V)
        """
        N, C, T, V = x.size()
        print('x.shape: ', x.shape)
        #print('x.shape: ', x.shape)
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input contains NaN or Inf!")

        # 添加位置編碼
        if self.use_pes:
            xs = x + self.pes(x)
        else:
            xs = x
        print('xs.shape: ', xs.shape)

        # 生成 Q, K, V
        #qkv = self.qkv_proj(xs).reshape(N, self.3 * num_heads, self.qkv_dim, T, V)
        #q, k, v = torch.chunk(qkv, 3, dim=2)  # q, k, v 形狀為 (N, num_heads, qkv_dim, T, V)
        q, k, v = torch.chunk(self.qkv_proj(xs).view(N, 3 * self.num_heads, self.qkv_dim, T, V), 3, dim=1)
        print('q.shape: ', q.shape)
        print('k.shape: ', k.shape)
        print('v.shape: ', v.shape)


        # 計算attention
        attention_scores = torch.einsum('nhctv,nhctu->nhuv', [q, k] / (self.qkv_dim ** 0.5) * T)  # (N, num_heads, V, V)
        print('attention_score.shape: ', attention_scores.shape)
        attention_probs = self.tan(attention_scores) * self.alphas + self.att0s.repeat(N, 1, 1, 1)# 在最後一维 (V) 上進行 softmax
        print('attention_probs.shape: ', attention_probs.shape)
        #attention_probs = F.softmax(attention_scores, dim=-1) * self.alphas + self.att0s.repeat(N, 1, 1, 1)# 在最後一维 (V) 上進行 softmax
        attention_probs = self.attention_dropout(attention_probs)


        # 乘上v
        attention_output = torch.einsum('nhuv,nhctv->nhctu', [attention_probs, v]).contiguous().view(N, self.num_heads * self.in_channels, T, V)
        print('attention_output.shape: ', attention_output.shape)

        # 轉成output維度
        attention_output = self.attention_output(attention_output)

        # 時間卷積與空間卷積
        #time_features = self.time_conv(attention_output)
        #spatial_features = self.spatial_conv(attention_output)

       

        # 添加殘差連接
        residual = self.residual_proj(x)
        
        # 空間特徵融合
        xs = self.spatial_conv(attention_output)
        xs = self.norm(xs)
        xs = self.activation(xs + residual)
        # 時間特徵融合
        xt = self.time_conv(xs)
        xt = self.norm(xt)
        xt = self.activation(xt + residual)

        

    

        return xt