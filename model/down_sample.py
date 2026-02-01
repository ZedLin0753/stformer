import torch
import torch.nn as nn

class Downsample(nn.Module):
    def __init__(self, num_channels):#, num_joints, num_frames, kernel_size
        super().__init__()
        self.conv_head = nn.Conv2d(num_channels, num_channels, kernel_size=(1, 2))
        self.conv_torso = nn.Conv2d(num_channels, num_channels, kernel_size=(1, 3))
        self.conv_left_arm = nn.Conv2d(num_channels, num_channels, kernel_size=(1, 6))
        self.conv_right_arm = nn.Conv2d(num_channels, num_channels, kernel_size=(1, 6))
        self.conv_left_leg = nn.Conv2d(num_channels, num_channels, kernel_size=(1, 4))
        self.conv_right_leg = nn.Conv2d(num_channels, num_channels, kernel_size=(1, 4))

    def forward(self, x):
        head = self.conv_head(x[:, :, :, [2, 3]]).mean(dim=-1)
        torso = self.conv_torso(x[:, :, :, [0, 1, 20]]).mean(dim=-1)
        left_arm = self.conv_left_arm(x[:, :, :, [4, 5, 6, 7, 21, 22]]).mean(dim=-1)
        right_arm = self.conv_right_arm(x[:, :, :, [8, 9, 10, 11, 23, 24]]).mean(dim=-1)
        left_leg = self.conv_left_leg(x[:, :, :, [12, 13, 14, 15]]).mean(dim=-1)
        right_leg = self.conv_right_leg(x[:, :, :, [16, 17, 18, 19]]).mean(dim=-1)

        x = torch.stack([head, torso, left_arm, right_arm, left_leg, right_leg], dim=-1)
        #print('x.shape: ', x.shape)

        return x