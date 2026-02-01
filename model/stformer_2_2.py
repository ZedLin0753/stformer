
import torch
import torch.nn as nn
from .STA import STA_Block

GROUPS = {
    "head": [2, 4],
    "torso": [0, 1, 20],
    "left_arm": [4, 5, 6, 7, 21, 22],
    "right_arm": [8, 9, 10, 11, 23, 24],
    "left_leg": [12, 13, 14, 15],
    "right_leg": [16, 17, 18, 19]
}

# 定義一個將關節點分群為區域特徵的模組
class GroupToPoint(nn.Module):
    def __init__(self, groups, in_channels):
        super().__init__()
        self.groups = groups
        self.weights = nn.ParameterDict({
            group_name: nn.Parameter(torch.ones(in_channels, len(indices)))
            for group_name, indices in groups.items()
        })

    def forward(self, x):
        grouped_points = []
        for group_name, indices in self.groups.items():
            group_feature = x[:, :, :, indices]  # 提取每個群組的特徵
            weights = self.weights[group_name].unsqueeze(0).unsqueeze(2)
            weighted_feature = (group_feature * weights).sum(dim=-1, keepdim=True)
            grouped_points.append(weighted_feature)
        return torch.cat(grouped_points, dim=-1)

# CrossAttentionBlock 用於將細節關節特徵與粗略分群特徵做跨注意力交互融合
class CrossAttentionBlock(nn.Module):
    def __init__(self, joint_channels, group_channels, qkv_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv_dim = qkv_dim

        self.to_q = nn.Conv2d(joint_channels, num_heads * qkv_dim, 1)
        self.to_kv = nn.Conv2d(group_channels, 2 * num_heads * qkv_dim, 1)

        self.attention_out = nn.Sequential(
            nn.Conv2d(num_heads * qkv_dim, joint_channels, 1),
            nn.BatchNorm2d(joint_channels)
        )

        self.scale = qkv_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, joint_feat, group_feat):
        N, C, T, V = joint_feat.shape
        _, _, _, G = group_feat.shape

        # 計算Q (來自joint)
        q = self.to_q(joint_feat).view(N, self.num_heads, self.qkv_dim, T*V) # (N, H, qkv_dim, T*V)

        # 計算K,V (來自group)
        kv = self.to_kv(group_feat).view(N, 2 * self.num_heads, self.qkv_dim, T * G) # (N, H, 2, qkv_dim, T*G)
        k, v = kv.chunk(2, dim=1) # (N, H, qkv_dim, T*G), (N, H, qkv_dim, T*G)
        # 計算注意力權重
        attn_weights = torch.einsum('nhct,nhcs->nhts', q, k) * self.scale
        attn_weights = self.softmax(attn_weights)

        # 進行注意力加權計算，融合特徵
        attn_out = torch.einsum('nhts,nhcs->nhct', attn_weights, v)
        attn_out = attn_out.reshape(N, -1, T, V)

        # 整合後輸出
        out = self.attention_out(attn_out)

        return out


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)



# Model中資料流：

class Model(nn.Module):
    def __init__(self, num_classes, num_joints, num_frames, num_persons, 
                 num_heads, num_channels, kernel_size, att_drop=0, dropout=0, dropout2d=0):
        super().__init__()
        in_channels = 64
        self.out_channels = 256
        # 原始輸入經過input_map轉換
        self.input_map = nn.Sequential(
            nn.Conv2d(num_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        # 分群模組
        self.group_to_point = GroupToPoint(GROUPS, 64)

        # 主編碼器
        
        self.en0_0 = STA_Block(in_channels=64, out_channels=64, qkv_dim=16, num_frames=120, num_joints=25,num_heads=num_heads, kernel_size=kernel_size)
        self.en1_0 = STA_Block(in_channels=64, out_channels=64, qkv_dim=16, num_frames=120, num_joints=25,num_heads=num_heads, kernel_size=kernel_size)
        self.en2_0 = STA_Block(in_channels=64, out_channels=128, qkv_dim=32, num_frames=120, num_joints=25, num_heads=num_heads, kernel_size=kernel_size)
        self.en3_0 = STA_Block(in_channels=128, out_channels=128, qkv_dim=32, num_frames=120, num_joints=25, num_heads=num_heads, kernel_size=kernel_size)
        self.en4_0 = STA_Block(in_channels=128, out_channels=256, qkv_dim=64, num_frames=120, num_joints=25, num_heads=num_heads, kernel_size=kernel_size)
        self.en5_0 = STA_Block(in_channels=256, out_channels=256, qkv_dim=64, num_frames=120, num_joints=25, num_heads=num_heads, kernel_size=kernel_size)
        self.en6_0 = STA_Block(in_channels=256, out_channels=256, qkv_dim=64, num_frames=120, num_joints=25, num_heads=num_heads, kernel_size=kernel_size)
        self.en7_0 = STA_Block(in_channels=256, out_channels=256, qkv_dim=64, num_frames=120, num_joints=25, num_heads=num_heads, kernel_size=kernel_size) 

        self.en0_1 = STA_Block(in_channels=64, out_channels=64, qkv_dim=16, num_frames=120, num_joints=6,num_heads=num_heads, kernel_size=kernel_size)
        self.en1_1 = STA_Block(in_channels=64, out_channels=64, qkv_dim=16, num_frames=120, num_joints=6,num_heads=num_heads, kernel_size=kernel_size)
        self.en2_1 = STA_Block(in_channels=64, out_channels=128, qkv_dim=32, num_frames=120, num_joints=6, num_heads=num_heads, kernel_size=kernel_size)
        self.en3_1 = STA_Block(in_channels=128, out_channels=128, qkv_dim=32, num_frames=120, num_joints=6, num_heads=num_heads, kernel_size=kernel_size)
        self.en4_1 = STA_Block(in_channels=128, out_channels=256, qkv_dim=64, num_frames=120, num_joints=6, num_heads=num_heads, kernel_size=kernel_size)
        self.en5_1 = STA_Block(in_channels=256, out_channels=256, qkv_dim=64, num_frames=120, num_joints=6, num_heads=num_heads, kernel_size=kernel_size)
        #self.en6_1 = STA_Block(in_channels=256, out_channels=256, qkv_dim=64, num_frames=120, num_joints=6, num_heads=num_heads, kernel_size=kernel_size)
        #self.en7_1 = STA_Block(in_channels=256, out_channels=256, qkv_dim=64, num_frames=120, num_joints=6, num_heads=num_heads, kernel_size=kernel_size)

        # Cross-Attention模組
        self.c_att0_0 = CrossAttentionBlock(joint_channels=64, group_channels=64, qkv_dim=16, num_heads=num_heads)
        self.c_att1_0 = CrossAttentionBlock(joint_channels=128, group_channels=128, qkv_dim=32, num_heads=num_heads)
        self.c_att2_0 = CrossAttentionBlock(joint_channels=256, group_channels=256, qkv_dim=64, num_heads=num_heads)
        

        self.fc = nn.Linear(self.out_channels, num_classes)
        self.drop_out = nn.Dropout(dropout)
        self.drop_out2d = nn.Dropout2d(dropout2d)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)

    def forward(self, x):
        N, C, T, V, M = x.shape
        # 重塑為適合模型輸入的形式
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        x = x.view(x.size(0), x.size(1), T, V)

        # 原始關節特徵
        joint_feat = self.input_map(x)  # (N*M, 64, T, V)

        # 分群後特徵
        group_feat = self.group_to_point(joint_feat)  # (N*M, C, T, 6)

        # Encoder1
        joint_feat = self.en0_0(joint_feat)  #(N*M, 64, 120, 25)
        group_feat = self.en0_1(group_feat)  #(N*M, 64, 120, 6)

        # Encoder2
        joint_feat = self.en1_0(joint_feat)  #(N*M, 64, 120, 25)
        group_feat = self.en1_1(group_feat)  #(N*M, 64, 120, 6)

        # Cross-Attention1
        joint_feat = self.c_att0_0(joint_feat, group_feat) #(N*M, 64, 120, 25)

        # Encoder3
        joint_feat = self.en2_0(joint_feat)  #(N*M, 128, 120, 25)
        group_feat = self.en2_1(group_feat)  #(N*M, 128, 120, 6)

        # Encoder4
        joint_feat = self.en3_0(joint_feat)  #(N*M, 128, 120, 25)
        group_feat = self.en3_1(group_feat)  #(N*M, 128, 120, 6)

        # Cross-Attention2
        joint_feat = self.c_att1_0(joint_feat, group_feat) #(N*M, 128, 120, 25)

        # Encoder5
        joint_feat = self.en4_0(joint_feat)  #(N*M, 256, 120, 25)
        group_feat = self.en4_1(group_feat)  #(N*M, 256, 120, 6)

        # Encoder6
        joint_feat = self.en5_0(joint_feat)  #(N*M, 256, 120, 25)
        group_feat = self.en5_1(group_feat)  #(N*M, 256, 120, 6)

        # Cross-Attention3
        joint_feat = self.c_att2_0(joint_feat, group_feat) #(N*M, 256, 120, 25)

        # Encoder7
        joint_feat = self.en6_0(joint_feat)  #(N*M, 256, 120, 25)
        joint_feat = self.en7_0(joint_feat)  #(N*M, 256, 120, 25)

        

        # 全局池化與分類
        final = joint_feat.view(N, M, self.out_channels, -1)
        final = final.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)
        final = self.drop_out2d(final)
        final = final.mean(3).mean(1)
        final = self.drop_out(final)

        final = self.fc(final)

        return final

        