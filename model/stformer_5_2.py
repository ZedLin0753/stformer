import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .STA_3 import STA_Block
#from .STA_2 import STA_Block
from .skeleton_pooling import StaticGroupWeightPool
from .hyper_pool import HyperPoolDynamic
from .channel_attention import SEBlock, ECABlock


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    if fc.bias is not None:
        nn.init.constant_(fc.bias, 0)

class Model(nn.Module):
    def __init__(self, num_classes, num_joints, num_frames, num_persons, 
                 num_heads, num_channels, kernel_size, frame_stride=8, beta=8, e_coarse=6, e_fine=25, use_pess=None, use_pest=None,
                 spatial=None, temporal=None,SE=None,ECA=None,
                 config=None, bottleneck_config=None, decoder_config=None,
                 att_drop=0, dropout=0, dropout2d=0):
        super().__init__()

        # self.use_pes = use_pes
        in_channels = 64
        self.base_channels = in_channels
        self.out_channels = self.base_channels*16
        self.num_channels = num_channels

        self.use_pess = use_pess
        self.use_pest = use_pest
        self.spatial = spatial
        self.temporal = temporal
        self.SE = SE
        self.ECA = ECA
        self.frame_stride = frame_stride
        self.beta = beta
        self.E_COARSE = e_coarse
        self.E_FINE = e_fine

        self.num_frames = num_frames
        self.num_joints = num_joints

        # input Mapping
        self.input_map_slow = nn.Sequential(
            nn.Conv2d(num_channels, self.base_channels, 1),
            nn.BatchNorm2d(self.base_channels),
            nn.ReLU())
        
        self.input_map_fast = nn.Sequential(
            nn.Conv2d(num_channels, (self.base_channels//self.beta), 1),
            nn.BatchNorm2d((self.base_channels//self.beta)),
            nn.ReLU())
        


        #Encoder Blocks fast  
        self.en0_fast = nn.ModuleList([STA_Block(in_channels=(self.base_channels//self.beta), out_channels=(self.base_channels//self.beta), 
                                                 qkv_dim=32, num_frames=self.num_frames, num_joints=25,
                                                 num_heads=num_heads, kernel_size=kernel_size, temporal=self.temporal, spatial=self.spatial, 
                                                 use_pess=self.use_pess, use_pest=self.use_pest, att_drop=att_drop) for _ in range(2)])
        self.en1_fast = nn.ModuleList([STA_Block(in_channels=((self.base_channels*2)//self.beta), out_channels=((self.base_channels*2)//self.beta), 
                                                 qkv_dim=32, num_frames=self.num_frames, num_joints=25,
                                                 num_heads=num_heads, kernel_size=kernel_size, temporal=self.temporal, spatial=self.spatial, 
                                                 use_pess=self.use_pess, use_pest=self.use_pest, att_drop=att_drop) for _ in range(2)])
        self.en2_fast = nn.ModuleList([STA_Block(in_channels=((self.base_channels*4)//self.beta), out_channels=((self.base_channels*4)//self.beta), 
                                                 qkv_dim=32, num_frames=self.num_frames, num_joints=25, 
                                                 num_heads=num_heads, kernel_size=kernel_size, temporal=self.temporal, spatial=self.spatial, 
                                                 use_pess=self.use_pess, use_pest=self.use_pest, att_drop=att_drop) for _ in range(2)])
        self.en3_fast = nn.ModuleList([STA_Block(in_channels=((self.base_channels*8)//self.beta), out_channels=((self.base_channels*8)//self.beta),
                                                 qkv_dim=32, num_frames=self.num_frames, num_joints=25,
                                                 num_heads=num_heads, kernel_size=kernel_size, temporal=self.temporal, spatial=self.spatial,
                                                 use_pess=self.use_pess, use_pest=self.use_pest, att_drop=att_drop) for _ in range(2)])
        
        #Encoder Blocks slow
        self.en0_slow = nn.ModuleList([STA_Block(in_channels=self.base_channels, out_channels=self.base_channels, 
                                                 qkv_dim=32, num_frames=(self.num_frames//self.frame_stride), num_joints=self.E_COARSE,
                                                 num_heads=num_heads, kernel_size=kernel_size, temporal=self.temporal, spatial=self.spatial,
                                                 use_pess=self.use_pess, use_pest=self.use_pest, att_drop=att_drop) for _ in range(2)])
        self.en1_slow = nn.ModuleList([STA_Block(in_channels=(self.base_channels*2), out_channels=(self.base_channels*2), 
                                                 qkv_dim=32, num_frames=(self.num_frames//self.frame_stride), num_joints=self.E_COARSE,
                                                 num_heads=num_heads, kernel_size=kernel_size, temporal=self.temporal, spatial=self.spatial,
                                                 use_pess=self.use_pess, use_pest=self.use_pest, att_drop=att_drop) for _ in range(2)])
        self.en2_slow = nn.ModuleList([STA_Block(in_channels=(self.base_channels*4), out_channels=(self.base_channels*4), 
                                                 qkv_dim=32, num_frames=(self.num_frames//self.frame_stride), num_joints=self.E_COARSE,
                                                 num_heads=num_heads, kernel_size=kernel_size, temporal=self.temporal, spatial=self.spatial,
                                                 use_pess=self.use_pess, use_pest=self.use_pest, att_drop=att_drop) for _ in range(2)])
        self.en3_slow = nn.ModuleList([STA_Block(in_channels=(self.base_channels*8), out_channels=(self.base_channels*8),
                                                 qkv_dim=32, num_frames=(self.num_frames//self.frame_stride), num_joints=self.E_COARSE,
                                                 num_heads=num_heads, kernel_size=kernel_size, temporal=self.temporal, spatial=self.spatial,
                                                 use_pess=self.use_pess, use_pest=self.use_pest, att_drop=att_drop) for _ in range(2)])
        

        self.fusion0 = nn.Sequential(
            nn.Conv2d(in_channels=(self.base_channels*2), out_channels=(self.base_channels*2), kernel_size=1),
            nn.BatchNorm2d(self.base_channels*2),
            nn.ReLU()
        )
        self.fusion1 = nn.Sequential(
            nn.Conv2d(in_channels=(self.base_channels*4), out_channels=(self.base_channels*4), kernel_size=1),
            nn.BatchNorm2d(self.base_channels*4),
            nn.ReLU()
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(in_channels=(self.base_channels*8), out_channels=(self.base_channels*8), kernel_size=1),
            nn.BatchNorm2d(self.base_channels*8),
            nn.ReLU()
        )
        self.fusion3 = nn.Sequential(
            nn.Conv2d(in_channels=(self.base_channels*16), out_channels=(self.base_channels*16), kernel_size=1),
            nn.BatchNorm2d(self.base_channels*16),
            nn.ReLU()
        )

        self.t_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=(self.base_channels//self.beta),
                      out_channels=(self.base_channels),
                      kernel_size=(self.frame_stride, 1),
                      stride=(self.frame_stride, 1),
                      padding=(0, 0),bias=False),
            nn.BatchNorm2d(self.base_channels),
            nn.ReLU()
        )

        self.t_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=((self.base_channels*2)//self.beta),
                      out_channels=(self.base_channels*2),
                      kernel_size=(self.frame_stride, 1),
                      stride=(self.frame_stride, 1),
                      padding=(0, 0),bias=False),
            nn.BatchNorm2d(self.base_channels*2),
            nn.ReLU()
        )
        self.t_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=((self.base_channels*4)//self.beta),
                      out_channels=(self.base_channels*4),
                      kernel_size=(self.frame_stride, 1),
                      stride=(self.frame_stride, 1),
                      padding=(0, 0),bias=False),
            nn.BatchNorm2d(self.base_channels*4),
            nn.ReLU()
        )
        self.t_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=((self.base_channels*8)//self.beta),
                      out_channels=(self.base_channels*8),
                      kernel_size=(self.frame_stride, 1),
                      stride=(self.frame_stride, 1),
                      padding=(0, 0),bias=False),
            nn.BatchNorm2d(self.base_channels*8),
            nn.ReLU()
        )

        self.channel_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=(self.base_channels//self.beta),
                        out_channels=((self.base_channels*2)//self.beta),
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        padding=(0, 0),bias=False),
                        nn.BatchNorm2d((self.base_channels*2)//self.beta),
                        nn.ReLU()
        )
        self.channel_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=((self.base_channels*2)//self.beta),
                        out_channels=((self.base_channels*4)//self.beta),
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        padding=(0, 0),bias=False),
                        nn.BatchNorm2d((self.base_channels*4)//self.beta),
                        nn.ReLU()
        )
        self.channel_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=((self.base_channels*4)//self.beta),
                        out_channels=((self.base_channels*8)//self.beta),
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        padding=(0, 0),bias=False),
                        nn.BatchNorm2d((self.base_channels*8)//self.beta),
                        nn.ReLU()
        )

        self.part_pool1 = HyperPoolDynamic(in_channels=self.num_channels, V=self.num_joints, E=self.E_COARSE) 
        self.part_pool2 = HyperPoolDynamic(in_channels=self.base_channels, V=self.num_joints, E=self.E_COARSE)
        self.part_pool3 = HyperPoolDynamic(in_channels=(self.base_channels*2), V=self.num_joints, E=self.E_COARSE)
        self.part_pool4 = HyperPoolDynamic(in_channels=(self.base_channels*4), V=self.num_joints, E=self.E_COARSE)
        self.part_pool5 = HyperPoolDynamic(in_channels=(self.base_channels*8), V=self.num_joints, E=self.E_COARSE)
        
        if self.SE:
            self.channel_att0 = SEBlock(self.base_channels) #64
            self.channel_att1 = SEBlock(self.base_channels*2) #128 
            self.channel_att2 = SEBlock(self.base_channels*4) #256
            self.channel_att3 = SEBlock((self.base_channels//self.beta)) #8
            self.channel_att4 = SEBlock((self.base_channels*2//self.beta)) #16
            self.channel_att5 = SEBlock((self.base_channels*4//self.beta)) #32
            self.channel_att6 = SEBlock(self.base_channels*8) #512
            self.channel_att7 = SEBlock(self.base_channels) #64
            
        elif self.ECA:
            self.channel_att0 = ECABlock(self.base_channels) #64
            self.channel_att1 = ECABlock(self.base_channels*2) #128
            self.channel_att2 = ECABlock(self.base_channels*4) #256
            self.channel_att3 = ECABlock((self.base_channels//self.beta)) #8
            self.channel_att4 = ECABlock((self.base_channels*2//self.beta)) #16
            self.channel_att5 = ECABlock((self.base_channels*4//self.beta)) #32
            self.channel_att6 = ECABlock(self.base_channels*8) #512
            self.channel_att7 = ECABlock(self.base_channels) #64
        else:
            self.channel_att0 = None
            self.channel_att1 = None
            self.channel_att2 = None
            self.channel_att3 = None
            self.channel_att4 = None
            self.channel_att5 = None
            self.channel_att6 = None
            self.channel_att7 = None

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
        y = x[:, :, ::self.frame_stride, :]
        parts_y = self.part_pool1(y)  # (N*M, C, 25, 6)

        T_fast = T
        T_slow = T // self.frame_stride
        x = self.input_map_fast(x) #(3, 8, 200, 25)  (N*M, 8, 200, 25)
        #y = self.input_map_slow(y) #(3, 64, 25, 25)  (N*M, 64, 25, 25)
        parts_y = self.input_map_slow(parts_y)  #(3, 64, 25, 6)  (N*M, 64, 25, 6)

        
        #stage 1
        #fast
        for blk_fast0 in self.en0_fast:
            x = blk_fast0(x) #(8, 8, 200, 25)   (N*M, 8, 200, 25)

        x = self.channel_att3(x) if self.channel_att3 is not None else x # (N*M, 8, 200, 25)

        #slow
        for blk_slow0 in self.en0_slow:
            parts_y = blk_slow0(parts_y) #(64, 64, 25, 6)  (N*M, 64, 25, 6)

        parts_y = self.channel_att0(parts_y) if self.channel_att0 is not None else parts_y # (N*M, 64, 25, 6)

        #merge1
        #fast to slow
        x_up = self.t_conv0(x) #(8, 64, 25, 25)  (N*M, 64, 25, 25)
        x_up = self.part_pool2(x_up)  # (N*M, 64, 25, 6)
        
        parts_y = self.fusion0(torch.cat([x_up, parts_y], dim=1)) #(64+64, 128, 25, 6)  (N*M, 128, 25, 6)

        #stage 2
        #fast
        x = self.channel_conv0(x) #(8, 16, 200, 25)  (N*M, 16, 200, 25)

        for blk_fast1 in self.en1_fast:
            x = blk_fast1(x) #(16, 16, 200, 25)   (N*M, 16, 200, 25)

        x = self.channel_att4(x) if self.channel_att4 is not None else x # (N*M, 16, 200, 25)

        #slow
        for blk_slow1 in self.en1_slow:
            parts_y = blk_slow1(parts_y) #(128, 128, 25, 6)  (N*M, 128, 25, 6)

        parts_y = self.channel_att1(parts_y) if self.channel_att1 is not None else parts_y # (N*M, 128, 25, 6)

        #merge2
        #fast to slow
        x_up = self.t_conv1(x) #(16, 128, 25, 25)  (N*M, 128, 25, 25)
        x_up = self.part_pool3(x_up)  # (N*M, 128, 25, 6)

        parts_y = self.fusion1(torch.cat([x_up, parts_y], dim=1)) #(128+128, 256, 25, 6)  (N*M, 256, 25, 6)

        #stage 3
        #fast
        x = self.channel_conv1(x) #(16, 32, 200, 25)  (N*M, 32, 200, 25)

        for blk_fast2 in self.en2_fast:
            x = blk_fast2(x) #(32, 32, 200, 25)   (N*M, 32, 200, 25)

        x = self.channel_att5(x) if self.channel_att5 is not None else x # (N*M, 32, 200, 25)

        #slow
        for blk_slow2 in self.en2_slow:
            parts_y = blk_slow2(parts_y) #(256, 256, 25, 6)  (N*M, 256, 25, 6)

        parts_y = self.channel_att2(parts_y) if self.channel_att2 is not None else parts_y # (N*M, 256, 25, 6)

        #merge3
        #fast to slow
        x_up = self.t_conv2(x) #(32, 256, 25, 25)  (N*M, 256, 25, 25)
        x_up = self.part_pool4(x_up)  # (N*M, 256, 25, 6)

        parts_y = self.fusion2(torch.cat([x_up, parts_y], dim=1)) #(256+256, 512, 25, 6)  (N*M, 512, 25, 6)

        #stage 4
        #fast
        x = self.channel_conv2(x) #(32, 64, 200, 25)  (N*M, 64, 200, 25)

        for blk_fast3 in self.en3_fast:
            x = blk_fast3(x) #(64, 64, 200, 25)   (N*M, 64, 200, 25)

        x = self.channel_att7(x) if self.channel_att7 is not None else x # (N*M, 64, 200, 25)
        
        #slow
        for blk_slow3 in self.en3_slow:
            parts_y = blk_slow3(parts_y) #(512, 512, 25, 6)  (N*M, 512, 25, 6)

        parts_y = self.channel_att6(parts_y) if self.channel_att6 is not None else parts_y # (N*M, 512, 25, 6)

        #merge4
        #fast to slow
        x_up = self.t_conv3(x) #(64, 512, 25, 25)  (N*M, 512, 25, 25)
        x_up = self.part_pool5(x_up) # (N*M, 512, 25, 6)

        parts_y = self.fusion3(torch.cat([x_up, parts_y], dim=1)) #(512+512, 1024, 25, 6)  (N*M, 1024, 25, 6)

        #Final Layer

        final = parts_y.view(N, M, self.out_channels, -1)
        final = final.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)
        final = self.drop_out2d(final)
        final = final.mean(3).mean(1)
        final = self.drop_out(final)
        final = self.fc(final)

        return final
