import torch.nn as nn
import torch
#rom .sta_block import STA_Block
#from .trans_block_2 import TRANS_Block
#from .ST_Block_3 import STA_Block
#rom .ST_Block_2 import STA_Block
from .STA import STA_Block
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
        in_channels = 128
        self.out_channels = 256

        num_frames = num_frames
        # num_joints = num_joints

        # input Mapping
        self.input_map = nn.Sequential(
            nn.Conv2d(num_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU())
        

        #Encoder Blocks
        self.en0_0 = STA_Block(in_channels=128, out_channels=128, qkv_dim=32, num_frames=120, num_joints=25,num_heads=num_heads, kernel_size=kernel_size)
        self.en1_0 = STA_Block(in_channels=128, out_channels=256, qkv_dim=32, num_frames=120, num_joints=25,num_heads=num_heads, kernel_size=kernel_size)
        self.en2_0 = STA_Block(in_channels=256, out_channels=256, qkv_dim=64, num_frames=60, num_joints=6, num_heads=num_heads, kernel_size=kernel_size)
        self.en3_0 = STA_Block(in_channels=256, out_channels=512, qkv_dim=64, num_frames=60, num_joints=6, num_heads=num_heads, kernel_size=kernel_size)
       #self.en4_0 = STA_Block(in_channels=256, out_channels=256, qkv_dim=64, num_frames=30, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        
        #Bottleneck
        self.bottleneck = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1))
        

        #Decder Blocks
        self.de0_0 = STA_Block(in_channels=256, out_channels=256, qkv_dim=64, num_frames=120, num_joints=25,num_heads=num_heads, kernel_size=kernel_size)
        self.de0_1 = STA_Block(in_channels=256, out_channels=256, qkv_dim=64, num_frames=120, num_joints=25,num_heads=num_heads, kernel_size=kernel_size)
        self.de0_2 = STA_Block(in_channels=256, out_channels=256, qkv_dim=64, num_frames=60, num_joints=6,num_heads=num_heads, kernel_size=kernel_size)
        self.de0_3 = STA_Block(in_channels=512, out_channels=256, qkv_dim=64, num_frames=60, num_joints=6,num_heads=num_heads, kernel_size=kernel_size)

        

        
        self.t_down = nn.AvgPool2d(kernel_size=(2,1))
        self.t_up = nn.Upsample(scale_factor=(2,1), mode='nearest')

        
        self.s_down2 = nn.AdaptiveAvgPool2d((None, 1))

        self.s_up1 = Upsample(256)
        self.s_up2 = nn.ConvTranspose2d(512,512,kernel_size=(1, 6))



        self.up3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.up4 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        

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

        #Encoding
        
        x0_0 = self.en0_0(x)          #(64, 64, 120, 25)
        x1_0 = self.en1_0(x0_0)       #(64, 128, 120, 25)
        down1 = self.t_down(x1_0)     #(128, 128, 60, 25)
        down1 = group_to_point(down1, GROUPS)  # (128, 128, 60, 6)

        x2_0 = self.en2_0(down1)      #(128, 128, 60, 6)
        x3_0 = self.en3_0(x2_0)       #(128, 256, 60, 6)
        down2 = self.t_down(x3_0)     #(256, 256, 30, 6)
        down2 = self.s_down2(down2)   #(256, 256, 30, 1)
        
        #Bottleneck
        x4_0 = self.bottleneck(down2) #(256, 256, 30, 1)

     

        #Decoding
        up1 = self.t_up(x4_0)         #(256, 256, 60, 1)
        up1 = self.s_up2(up1)         #(256, 256, 60, 6)
        merge1 = torch.cat([up1, x3_0], dim=1) #(256, 512, 60, 6)
        merge1 = self.up4(merge1)      #(512, 256, 60, 6)
        x0_3 = self.de0_3(merge1)     #(256, 128, 60, 6)
        x0_2 = self.de0_2(x0_3)       #(128, 128, 60, 6)

        up2 = self.t_up(x0_2)         #(128, 128, 120, 6)
        up2 = self.s_up1(up2)         #(128, 128, 120, 25)
        merge2 = torch.cat([up2, x1_0], dim=1) #(128, 256, 120, 25)
        merge2 = self.up3(merge2)      #(256, 128, 120, 25)
        x0_1 = self.de0_1(merge2)     #(128, 128, 120, 25)
        final = self.de0_0(x0_1)      #(128, 128, 120, 25)

        #Final Layer

        

        #final_4
        final_4 = final.view(N, M, self.out_channels, -1)
        final_4 = final_4.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)
        final_4 = self.drop_out2d(final_4)
        final_4 = final_4.mean(3).mean(1)
        final_4 = self.drop_out(final_4)

        final_4 = self.fc(final_4)

        return final_4
