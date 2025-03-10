import torch.nn as nn
import torch
# from .sta_block import STA_Block
# from .trans_block_2 import TRANS_Block
from .ST_Block_3 import STA_Block
from .down_sample import Downsample
from .up_sample import Upsample


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
        in_channels = 32
        self.out_channels = 64

        num_frames = num_frames
        # num_joints = num_joints

        # input Mapping
        self.input_map = nn.Sequential(
            nn.Conv2d(num_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU())
        
        # Encoder Blocks
        self.en0_0 = STA_Block(
            in_channels=32, 
            out_channels=64,
            qkv_dim=64,
            num_frames=num_frames,
            num_joints=25,
            num_heads=num_heads,
            kernel_size=kernel_size
        )
        self.en1_0 = STA_Block(
            in_channels=64,
            out_channels=64,
            qkv_dim=64,
            num_frames=num_frames,
            num_joints=25,
            num_heads=num_heads,
            kernel_size=kernel_size
        )
        self.en2_0 = STA_Block(
            in_channels=64,
            out_channels=128,
            qkv_dim=64,
            num_frames=num_frames,
            num_joints=6,
            num_heads=num_heads,
            kernel_size=kernel_size
        )
        self.en3_0 = STA_Block(
            in_channels=128,
            out_channels=128,
            qkv_dim=64,
            num_frames=num_frames,
            num_joints=6,
            num_heads=num_heads,
            kernel_size=kernel_size
        )
        self.en4_0 = STA_Block(
            in_channels=128,
            out_channels=256,
            qkv_dim=64,
            num_frames=num_frames,
            num_joints=1,
            num_heads=num_heads,
            kernel_size=kernel_size
        )
        self.en5_0 = STA_Block(
            in_channels=256,
            out_channels=256,
            qkv_dim=64,
            num_frames=num_frames,
            num_joints=1,
            num_heads=num_heads,
            kernel_size=kernel_size
        )

        # Bottleneck
        self.bottleneck = nn.Conv2d(256, 512, kernel_size=(1, 1))

        # Decoder Blocks
        self.de0_5 = STA_Block(
            in_channels=512,
            out_channels=256,
            qkv_dim=64,
            num_frames=num_frames,
            num_joints=1,
            num_heads=num_heads,
            kernel_size=kernel_size
        )
        self.de0_4 = STA_Block(
            in_channels=256,
            out_channels=128,
            qkv_dim=64,
            num_frames=num_frames,
            num_joints=1,
            num_heads=num_heads,
            kernel_size=kernel_size
        )
        self.de0_3 = STA_Block(
            in_channels=128,
            out_channels=128,
            qkv_dim=64,
            num_frames=num_frames,
            num_joints=6,
            num_heads=num_heads,
            kernel_size=kernel_size
        )
        self.de0_2 = STA_Block(
            in_channels=128,
            out_channels=64,
            qkv_dim=64,
            num_frames=num_frames,
            num_joints=6,
            num_heads=num_heads,
            kernel_size=kernel_size
        )
        self.de0_1 = STA_Block(
            in_channels=64,
            out_channels=64,
            qkv_dim=64,
            num_frames=num_frames,
            num_joints=25,
            num_heads=num_heads,
            kernel_size=kernel_size
        )
        self.de0_0 = STA_Block(
            in_channels=64,
            out_channels=64, 
            qkv_dim=64,
            num_frames=num_frames,
            num_joints=25,
            num_heads=num_heads,
            kernel_size=kernel_size
        )

        self.down1 = Downsample(num_channels=64)
        self.down2 = nn.AdaptiveAvgPool2d((None, 1))

        self.u1 = Upsample(64)
        self.u2 = nn.ConvTranspose2d(256, 128, kernel_size=(1, 6))

        # self.downsample = nn.AvgPool2d(kernel_size=(2,1))

        # self.upsample = nn.Upsample(scale_factor=(2,1), mode='nearest')
        # self.up1 = nn.Conv2d(in_channels=128, out_channels=64,
        #                      kernel_size=1)
        # self.up2 = nn.Conv2d(in_channels=256, out_channels=128,
        #                      kernel_size=1) 
        # self.up3 = nn.Conv2d(in_channels=512, out_channels=256,
        #                      kernel_size=1)
        # self.up4 = nn.Conv2d(in_channels=1024, out_channels=512,
        #                      kernel_size=1)

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
        # input
        N, C, T, V, M = x.shape
        # print('input shape 1: ',x.shape)
        # print(x[0, 0, 0, 0:, 0])
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        # print('input shape 2: ',x.shape)
        x = x.view(x.size(0), x.size(1), T, V)
        # print('input shape 3: ',x.shape)
        x = self.input_map(x)  # (32, 32, 120, 25)
        # print('input shape 4: ',x.shape)

        # Position Embedding
        # x = self.pes(x) + x if self.use_pes else x  #(32, 32, 120, 25)
        # print('input shape(position embed): ',x.shape)

        # Encoding

        x0_0 = self.en0_0(x)  # (32, 64, 120, 25)
        # print('x0_0.shape: ', x0_0.shape)

        x1_0 = self.en1_0(x0_0)  # (32, 64, 120, 25)
        # print('x1_0.shape: ', x1_0.shape)
        do1_0 = self.down1(x1_0)  # (32, 64, 120, 6)
        # print('do1_0.shape: ', do1_0.shape)
        x2_0 = self.en2_0(do1_0)  # (32, 128, 120, 6)
        # print('x2_0.shape: ', x2_0.shape)
        x3_0 = self.en3_0(x2_0)  # (32, 128, 120, 6)
        # print('x3_0.shape: ', x3_0.shape)
        do2_0 = self.down2(x3_0)  # (32, 128, 120, 1)
        # print('do2_0.shape: ', do2_0.shape)
        x4_0 = self.en4_0(do2_0)  # (32, 256, 120, 1)
        # print('x4_0.shape: ', x4_0.shape)
        x5_0 = self.en5_0(x4_0)  # (32, 256, 120, 1)
        # print('x5_0.shape: ', x5_0.shape)
        # Bottleneck
        bottle = self.bottleneck(x5_0)
        # print('bottleneck.shape: ', bottle.shape)

        # Decoding

        x0_5 = self.de0_5(bottle)  # (32, 256, 120, 1)
        # print('x0_5.shape: ', x0_5.shape)
        x0_4 = self.de0_4(x0_5)  # (32, 128, 120, 1)
        # print('x0_4.shape: ', x0_4.shape)
        merge1 = torch.cat((do2_0, x0_4), dim=1)  # (32, 256, 120, 1)
        # print('merge1.shape: ', merge1.shape)
        up_merge1 = self.u2(merge1)  # (32, 128, 120, 6)
        # print('up_merge1.shape: ', up_merge1.shape)
        x0_3 = self.de0_3(up_merge1)  # (32, 128, 120, 6)
        # print('x0_3.shape: ', x0_3.shape)
        x0_2 = self.de0_2(x0_3)  # (32, 64, 120, 6)
        # print('x0_2.shape: ', x0_2.shape)
        merge2 = torch.cat((do1_0, x0_2), dim=1)  # (32, 128, 120, 6)
        # print('merge2.shape: ', merge2.shape)
        up_merge2 = self.u1(merge2)  # (32, 64, 120, 25)
        # print('up_merge2.shape: ', up_merge2.shape)
        x0_4 = self.de0_1(up_merge2)  # (32, 64, 120, 25)
        # print('x0_4.shape: ', x0_4.shape)
        x0_5 = self.de0_0(x0_4)  # (32, 64, 120, 25)
        # print('x0_5.shape: ', x0_5.shape)

        # Final Layer

        # final_4
        final = x0_5.view(N, M, self.out_channels, -1)
        final = final.permute(0, 1, 3, 2).contiguous().view(
            N, -1, self.out_channels, 1)
        final = self.drop_out2d(final)
        final = final.mean(3).mean(1)
        final = self.drop_out(final)

        return self.fc(final)
