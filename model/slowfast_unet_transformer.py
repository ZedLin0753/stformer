
import torch.nn as nn
import torch
import torch.nn.functional as F
from .STA_2 import STA_Block
from .down_sample import Downsample
from .up_sample import Upsample

class SlowFastTransformerModel(nn.Module):
    def __init__(self, in_channels=3, num_joints=25, num_frames=60, slow_stride=4, num_classes=60, base_channel=64):
        super().__init__()
        self.slow_stride = slow_stride

        # Fast Path Encoder (U-Net style)
        self.fast_encoder1 = STA_Block(in_channels, base_channel)
        self.fast_encoder2 = STA_Block(base_channel, base_channel * 2)
        self.fast_encoder3 = STA_Block(base_channel * 2, base_channel * 4)

        self.down1 = Downsample()
        self.down2 = Downsample()

        # Slow Path Encoder (shares channel config but processes downsampled time)
        self.slow_encoder1 = STA_Block(in_channels, base_channel)
        self.slow_encoder2 = STA_Block(base_channel, base_channel * 2)
        self.slow_encoder3 = STA_Block(base_channel * 2, base_channel * 4)

        self.fusion = nn.Sequential(
            nn.Conv2d(base_channel * 4, base_channel * 4, kernel_size=1),
            nn.BatchNorm2d(base_channel * 4),
            nn.ReLU()
        )

        # Decoder
        self.up1 = Upsample()
        self.up2 = Upsample()
        self.decoder1 = STA_Block(base_channel * 4, base_channel * 2)
        self.decoder2 = STA_Block(base_channel * 2, base_channel)
        self.decoder3 = STA_Block(base_channel, base_channel)

        self.fc = nn.Linear(base_channel, num_classes)

    def forward(self, x):  # x: [N, C, T, V]
        # FAST PATH
        f1 = self.fast_encoder1(x)          # [N, C1, T, V]
        f2 = self.fast_encoder2(self.down1(f1))  # [N, C2, T/2, V]
        f3 = self.fast_encoder3(self.down2(f2))  # [N, C3, T/4, V]

        # SLOW PATH
        x_slow = x[:, :, ::self.slow_stride, :]  # 時間下採樣
        s1 = self.slow_encoder1(x_slow)
        s2 = self.slow_encoder2(self.down1(s1))
        s3 = self.slow_encoder3(self.down2(s2))
        s3_up = F.interpolate(s3, size=f3.shape[2], mode='linear', align_corners=False)

        # FUSION
        fused = self.fusion(f3 + s3_up)  # or torch.cat and conv

        # DECODER
        u1 = self.up1(fused) + f2
        d1 = self.decoder1(u1)
        u2 = self.up2(d1) + f1
        d2 = self.decoder2(u2)
        d3 = self.decoder3(d2)

        # POOL & CLASSIFY
        out = d3.mean(dim=[2, 3])  # GAP over T and V
        return self.fc(out)
