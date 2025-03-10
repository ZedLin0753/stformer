import torch.nn as nn
import torch


GROUPS = {
    "head": [0, 1, 2],
    "torso": [3, 20, 9, 10, 11],
    "left_arm": [4, 5, 6, 7, 8],
    "right_arm": [12, 13, 14, 15, 16],
    "left_leg": [17, 18, 19, 20],
    "right_leg": [21, 22, 23, 24],
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