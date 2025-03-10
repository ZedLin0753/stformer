import torch
import torch.nn as nn
import math
import numpy as np

class Pos_Embed(nn.Module):
    def __init__(self, channels, num_frames, num_joints, domain):
        super(Pos_Embed, self).__init__()

        self.num_frames = num_frames
        self.num_joints = num_joints
        self.domain = domain

        if self.domain == 'spatial':# spatial
            pos_list = []
            for t in range(self.num_frames):
                for s in range(self.num_joints):
                    pos_list.append(s)

        elif self.domain == 'temporal':# temporal
            pos_list = []
            for t in range(self.num_frames):
                for s in range(self.num_joints):
                    pos_list.append(t)
        

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()

        pe = torch.zeros(self.num_frames * self.num_joints, channels)

        div_term = torch.exp(torch.arange(0, channels, 2).float() * -(math.log(10000.0) / channels)) 
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(self.num_frames, self.num_joints, channels).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = self.pe[:, :, :x.size(2)]
        return x