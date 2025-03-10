import torch.nn as nn
import torch
#rom .sta_block import STA_Block
#from .trans_block_2 import TRANS_Block
from .ST_Block_3 import STA_Block
#rom .ST_Block_2 import STA_Block
from .down_sample import Downsample
from .up_sample import Upsample

GROUPS = {
    "head": [0, 1, 2],
    "torso": [3, 20, 9, 10, 11],
    "left_arm": [4, 5, 6, 7, 8],
    "right_arm": [12, 13, 14, 15, 16],
    "left_leg": [17, 18, 19, 20],
    "right_leg": [21, 22, 23, 24],
}

def group_to_point(x, groups):
    """
    將每個分群的關節特徵壓縮成單一點特徵。
    x: (N, C, T, V) 輸入數據
    groups: 每個群組的關節索引
    返回: (N, C, T, num_groups)
    """
    grouped_points = []
    for indices in groups.values():
        group_feature = x[:, :, :, indices].mean(dim=-1, keepdim=True)  # 平均池化壓縮
        grouped_points.append(group_feature)
    return torch.cat(grouped_points, dim=-1)  # 合併為新的虛擬關節點


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
        self.out_channels = 256

        num_frames = num_frames
        # num_joints = num_joints

        # input Mapping
        self.input_map = nn.Sequential(
            nn.Conv2d(num_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU())
        
        


        #Encoder Blocks
        self.en0_0 = STA_Block(in_channels=64, out_channels=64, qkv_dim=16, num_frames=120, num_joints=6,num_heads=num_heads, kernel_size=kernel_size)
        self.en1_0 = STA_Block(in_channels=64, out_channels=64, qkv_dim=16, num_frames=120, num_joints=6,num_heads=num_heads, kernel_size=kernel_size)
        self.en2_0 = STA_Block(in_channels=64, out_channels=128, qkv_dim=32, num_frames=120, num_joints=6, num_heads=num_heads, kernel_size=kernel_size)
        self.en3_0 = STA_Block(in_channels=128, out_channels=128, qkv_dim=32, num_frames=120, num_joints=6, num_heads=num_heads, kernel_size=kernel_size)
        self.en4_0 = STA_Block(in_channels=128, out_channels=256, qkv_dim=64, num_frames=120, num_joints=6, num_heads=num_heads, kernel_size=kernel_size)
        self.en5_0 = STA_Block(in_channels=256, out_channels=256, qkv_dim=64, num_frames=120, num_joints=6, num_heads=num_heads, kernel_size=kernel_size)
        self.en6_0 = STA_Block(in_channels=256, out_channels=256, qkv_dim=64, num_frames=120, num_joints=6, num_heads=num_heads, kernel_size=kernel_size)
        self.en7_0 = STA_Block(in_channels=256, out_channels=256, qkv_dim=64, num_frames=120, num_joints=6, num_heads=num_heads, kernel_size=kernel_size)    
        

        
       

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
        x = self.input_map(x) #(64, 64, 120, 25)

        x = group_to_point(x, GROUPS)  # (64, 64, T, 6)


        #Encoder Blocks
        x = self.en0_0(x) #(64, 64, 120, 6)
        x = self.en1_0(x) #(64, 64, 120, 6)
        x = self.en2_0(x) #(64, 128, 120, 6)
        x = self.en3_0(x) #(128, 128, 120, 6)
        x = self.en4_0(x) #(128, 256, 120, 6)
        x = self.en5_0(x) #(256, 256, 120, 6)
        x = self.en6_0(x) #(256, 256, 120, 6)
        x = self.en7_0(x) #(256, 256, 120, 6)

        #Final Layer

        

        #final_4
        final_4 = x.view(N, M, self.out_channels, -1)
        final_4 = final_4.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)
        final_4 = self.drop_out2d(final_4)
        final_4 = final_4.mean(3).mean(1)
        final_4 = self.drop_out(final_4)

        final_4 = self.fc(final_4)

        return final_4
