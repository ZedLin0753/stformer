import torch
import torch.nn as nn

class Upsample(nn.Module):
    def __init__(self, num_channels):#, num_joints, num_frames, kernel_size
        super().__init__()
        #self.conv_head = nn.ConvTranspose2d(num_channels*2, num_channels, kernel_size=(1, 2), stride=(1, 1), output_padding=(0, 0))
        #self.conv_torso = nn.ConvTranspose2d(num_channels*2, num_channels, kernel_size=(1, 3), stride=(1, 1), output_padding=(0, 0))
        #self.conv_left_arm = nn.ConvTranspose2d(num_channels*2, num_channels, kernel_size=(1, 6), stride=(1, 1), output_padding=(0, 0))
        #self.conv_right_arm = nn.ConvTranspose2d(num_channels*2, num_channels, kernel_size=(1, 6), stride=(1, 1), output_padding=(0, 0))
        #self.conv_left_leg = nn.ConvTranspose2d(num_channels*2, num_channels, kernel_size=(1, 4), stride=(1, 1), output_padding=(0, 0))
        #self.conv_right_leg = nn.ConvTranspose2d(num_channels*2, num_channels, kernel_size=(1, 4), stride=(1, 1), output_padding=(0, 0))

        self.upsample = nn.Upsample(size=(120, 25), mode = 'nearest')
        #self.up_head = nn.Upsample(size=(120, 2), mode = 'nearest')
        #self.up_torso = nn.Upsample(size=(120, 3), mode = 'nearest')
        #self.up_left_arm = nn.Upsample(size=(120, 6), mode = 'nearest')
        #self.up_right_arm = nn.Upsample(size=(120, 6), mode = 'nearest')
        #self.up_left_leg = nn.Upsample(size=(120, 4), mode = 'nearest')
        #self.up_right_leg = nn.Upsample(size=(120, 4), mode = 'nearest')
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=1)
    def forward(self, x):
        #head = self.conv_head(x[:, :, :, 0:1].expand(-1, -1, -1, 2))
        #print('head.shape: ', head.shape)
        #torso = self.conv_torso(x[:, :, :, [1]].expand(-1, -1, -1, 3))
        #print('torso.shape: ', torso.shape)
        #left_arm = self.conv_left_arm(x[:, :, :, [2]].expand(-1, -1, -1, 6))
        #print('left_arm.shape: ', left_arm.shape)
        #right_arm = self.conv_right_arm(x[:, :, :, [3]].expand(-1, -1, -1, 6))
        #print('right_arm.shape: ', right_arm.shape)
        #left_leg = self.conv_left_leg(x[:, :, :, [4]].expand(-1, -1, -1, 4))
        #print('left_leg.shape: ', left_leg.shape)
        #right_leg = self.conv_right_leg(x[:, :, :, [5]].expand(-1, -1, -1, 4))
        #print('right_leg.shape: ', right_leg.shape)

        #x = torch.cat([head, torso, left_arm, right_arm, left_leg, right_leg], dim=-1)

        x = self.upsample(x)
       #print('x.shape: ', x.shape)
        x = self.conv(x)
       #print('x.shape: ', x.shape)

        return x