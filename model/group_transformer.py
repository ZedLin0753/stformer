import torch.nn as nn
import torch
#rom .sta_block import STA_Block
#from .trans_block_2 import TRANS_Block
from .ST_Block_3 import STA_Block
#rom .ST_Block_2 import STA_Block
from .down_sample import Downsample
from .up_sample import Upsample

GROUPS = {
    "head": [2, 4],
    "torso": [0, 1, 20],
    "left_arm": [4, 5, 6, 7, 21, 22],
    "right_arm": [8, 9, 10, 11, 23, 24],
    "left_leg": [12, 13, 14, 15],
    "right_leg": [16, 17, 18, 19]
}

def group_joints(x, groups):
    return {
        group_name: x[:, :, :, indices] for group_name, indices in groups.items()
    }

class GroupTransformerLayer(nn.Module):
    def __init__(self, in_channels, num_heads, kernel_size, num_frames, att_drop=0):
        super().__init__()
        self.group_attention = nn.ModuleDict({
            group: STA_Block(
                in_channels=in_channels, out_channels=in_channels, qkv_dim=32,
                num_frames=num_frames, num_joints=len(indices), num_heads=num_heads,
                kernel_size=kernel_size, att_drop=att_drop
            )
            for group, indices in GROUPS.items()
        })
        self.cross_attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads)

    def forward(self, x):
        # Group-wise Attention
        grouped_features = group_joints(x, GROUPS)
        group_features = {
            group_name: self.group_attention[group_name](feature)
            for group_name, feature in grouped_features.items()
        }
        # Compress group features
        embeddings = [feature.mean(dim=-1).mean(dim=-2) for feature in group_features.values()]  # (N, C) per group
        embeddings = torch.stack(embeddings, dim=0)  # (num_groups, N, C)
        # Cross-Group Attention
        global_features, _ = self.cross_attention(embeddings, embeddings, embeddings)
        return global_features.mean(dim=0)  # (N, C)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)


class Model(nn.Module):
    def __init__(self, num_classes, num_joints, num_frames, num_persons, 
                 num_heads, num_channels, kernel_size, use_pes=True, 
                 config=None, bottleneck_config=None, decoder_config=None,
                 att_drop=0, dropout=0, dropout2d=0):
        super().__init__()

        # self.use_pes = use_pes
        in_channels = 64
        self.out_channels = 128

        num_frames = num_frames
        # num_joints = num_joints

        # input Mapping
        self.input_map = nn.Sequential(
            nn.Conv2d(num_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU())
        
        self.transformer_layers = nn.ModuleList([
            GroupTransformerLayer(
                in_channels=in_channels, num_heads=num_heads, kernel_size=kernel_size,
                num_frames=num_frames, att_drop=att_drop
            ) for _ in range(3)  #num_layers
        ])

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
        #input
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        x = x.view(x.size(0), x.size(1), T, V)
        x = self.input_map(x) #(32, 32, 120, 25)
        print('x.shape: ', x.shape)

        # Apply multi-layer Group Transformer
        for layer in self.transformer_layers:
            x = layer(x)  # (N*M, in_channels)

        final_4 = x.view(N, M, self.out_channels, -1)
        final_4 = final_4.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)
        final_4 = self.drop_out2d(final_4)
        final_4 = final_4.mean(3).mean(1)
        final_4 = self.drop_out(final_4)

        final_4 = self.fc(final_4)

        return final_4
