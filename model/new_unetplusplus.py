import torch.nn as nn
import torch
#from .sta_block import STA_Block
from .trans_block import TRANS_Block
from .pos_embed import Pos_Embed

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
    def __init__(self, num_classes, num_joints, 
                 num_frames, num_persons, num_heads, num_channels, 
                 kernel_size, use_pes = True, config=None, bottleneck_config=None, decoder_config=None,
                 att_drop=0, dropout=0, dropout2d=0):
        super().__init__()

        self.use_pes = use_pes
        in_channels = config[0][0]
        self.out_channels = decoder_config[3][1]
        

        num_frames = num_frames 
        num_joints = num_joints 

        #input Mapping
        self.input_map = nn.Sequential(
            nn.Conv2d(num_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU())
        
        #position embedding
        if self.use_pes: self.pes = Pos_Embed(in_channels, num_frames, num_joints)
        
        

        #dimension of encoder and decoder
        self.encoder_config = config
        self.bottleneck_config = bottleneck_config
        self.decoder_config = decoder_config


        self.encoders = nn.ModuleList()
        self.decoders0 = nn.ModuleList()
        self.decoders1 = nn.ModuleList()
        self.decoders2 = nn.ModuleList()
        self.decoders3 = nn.ModuleList()
        


        #Encoder Blocks
        self.en0_0 = TRANS_Block(in_channels=512, out_channels=256, qkv_dim=64, num_frames=num_frames, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        self.encoders.append(self.en0_0)
        self.en1_0 = TRANS_Block(in_channels=256, out_channels=128, qkv_dim=64, num_frames=num_frames, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        self.encoders.append(self.en1_0)
        self.en2_0 = TRANS_Block(in_channels=128, out_channels=64, qkv_dim=64, num_frames=num_frames, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        self.encoders.append(self.en2_0)
        self.en3_0 = TRANS_Block(in_channels=64, out_channels=32, qkv_dim=64, num_frames=num_frames, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        self.encoders.append(self.en3_0)
        self.en4_0 = TRANS_Block(in_channels=32, out_channels=16, qkv_dim=64, num_frames=num_frames, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        self.encoders.append(self.en4_0)
        
        

        #Decder Blocks
        self.de0_1 = TRANS_Block(in_channels=384, out_channels=512, qkv_dim=64, num_frames=num_frames, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        self.decoders0.append(self.de0_1)
        self.de1_1 = TRANS_Block(in_channels=192, out_channels=256, qkv_dim=64, num_frames=num_frames, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        self.decoders1.append(self.de1_1)
        self.de0_2 = TRANS_Block(in_channels=768, out_channels=512, qkv_dim=64, num_frames=num_frames, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        self.decoders1.append(self.de0_2)
        self.de2_1 = TRANS_Block(in_channels=96, out_channels=128, qkv_dim=64, num_frames=num_frames, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        self.decoders2.append(self.de2_1)
        self.de1_2 = TRANS_Block(in_channels=384, out_channels=256, qkv_dim=64, num_frames=num_frames, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        self.decoders2.append(self.de1_2)
        self.de0_3 = TRANS_Block(in_channels=768, out_channels=512, qkv_dim=64, num_frames=num_frames, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        self.decoders2.append(self.de0_3)
        self.de3_1 = TRANS_Block(in_channels=48, out_channels=64, qkv_dim=64, num_frames=num_frames, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        self.decoders3.append(self.de3_1)
        self.de2_2 = TRANS_Block(in_channels=192, out_channels=128, qkv_dim=64, num_frames=num_frames, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        self.decoders3.append(self.de2_2)
        self.de1_3 = TRANS_Block(in_channels=384, out_channels=256, qkv_dim=64, num_frames=num_frames, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        self.decoders3.append(self.de1_3)
        self.de0_4 = TRANS_Block(in_channels=768, out_channels=512, qkv_dim=64, num_frames=num_frames, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        self.decoders3.append(self.de0_4)
        

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
        #print('input shape 1: ',x.shape)
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        #print('input shape 2: ',x.shape)
        x = x.view(x.size(0), x.size(1), T, V)
        #print('input shape 3: ',x.shape)
        x = self.input_map(x)
        #print('input shape 4: ',x.shape)
        
        #Position Embedding
        x = self.pes(x) + x if self.use_pes else x
        #print('input shape(position embed): ',x.shape)


        #Encoding
        

        x0_0 = self.en0_0(x)
        #print('x0_0: ', x0_0.shape)
        #encoder_features.append(x0_0)
        x1_0 = self.en1_0(x0_0)
        #print('x1_0: ', x1_0.shape)
        #encoder_features.append(x1_0)
        x2_0 = self.en2_0(x1_0)
        #print('x2_0: ', x2_0.shape)
        #encoder_features.append(x2_0)
        x3_0 = self.en3_0(x2_0)
        #print('x3_0: ', x3_0.shape)
        #encoder_features.append(x3_0)
        x4_0 = self.en4_0(x3_0)

        #Decoding
        

        x0_1 = self.de0_1(torch.cat((x0_0, x1_0),dim=1))
        #print('x3_1: ', x3_1.shape)
        #decoder_features.append(x3_1)
        x1_1 = self.de1_1(torch.cat((x1_0, x2_0),dim=1))
        #print('x2_2: ', x2_2.shape)
        #decoder_features.append(x2_2)
        x0_2 = self.de0_2(torch.cat((x0_1, x1_1),dim=1))
        #print('x1_3: ', x1_3.shape)
        #decoder_features.append(x1_3)
        x2_1 = self.de2_1(torch.cat((x2_0, x3_0),dim=1))
        #print('x0_4: ', x0_4.shape)
        #decoder_features.append(x0_4)
        x1_2 = self.de1_2(torch.cat((x1_1, x2_1),dim=1))
        #print('x0_4: ', x0_4.shape)
        #decoder_features.append(x0_4)
        x0_3 = self.de0_3(torch.cat((x0_2, x1_2),dim=1))
        #print('x0_4: ', x0_4.shape)
        #decoder_features.append(x0_4)
        x3_1 = self.de3_1(torch.cat((x3_0, x4_0),dim=1))
        #print('x0_4: ', x0_4.shape)
        #decoder_features.append(x0_4)
        x2_2 = self.de2_2(torch.cat((x2_1, x3_1),dim=1))
        #print('x0_4: ', x0_4.shape)
        #decoder_features.append(x0_4)
        x1_3 = self.de1_3(torch.cat((x1_2, x2_2),dim=1))
        #print('x0_4: ', x0_4.shape)
        #decoder_features.append(x0_4)
        x0_4 = self.de0_4(torch.cat((x0_3, x1_3),dim=1))
        #print('x0_4: ', x0_4.shape)
        #decoder_features.append(x0_4)



        

        # NM, C, T, V
        x = x0_4.view(N, M, self.out_channels, -1)
        x = x.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)
        x = self.drop_out2d(x)
        x = x.mean(3).mean(1)

        x = self.drop_out(x)
        #print('output.shape: ', x)

        return self.fc(x)
