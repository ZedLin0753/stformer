import torch.nn as nn
import torch
#from .sta_block import STA_Block
from .trans_block_2 import TRANS_Block
from .pos_embed import Pos_Embed
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
    def __init__(self, num_classes, num_joints, 
                 num_frames, num_persons, num_heads, num_channels, 
                 kernel_size, use_pes = True, config=None, bottleneck_config=None, decoder_config=None,
                 att_drop=0, dropout=0, dropout2d=0):
        super().__init__()

        #self.use_pes = use_pes
        in_channels = 32
        self.out_channels = 64
        

        num_frames = num_frames 
        num_joints = num_joints 

        #input Mapping
        self.input_map = nn.Sequential(
            nn.Conv2d(num_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU())
        
        #position embedding
        #if self.use_pes: self.pes = Pos_Embed(in_channels, num_frames, num_joints)
        
        

      
        



        
        #Encoder Blocks
        self.en0_0 = TRANS_Block(in_channels=32, out_channels=64, qkv_dim=64, num_frames=120, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        self.en1_0 = TRANS_Block(in_channels=64, out_channels=128, qkv_dim=64, num_frames=60, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        self.en2_0 = TRANS_Block(in_channels=128, out_channels=256, qkv_dim=64, num_frames=30, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        self.en3_0 = TRANS_Block(in_channels=256, out_channels=512, qkv_dim=64, num_frames=15, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        self.en4_0 = TRANS_Block(in_channels=512, out_channels=1024, qkv_dim=64, num_frames=15, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        
        

        #Decder Blocks
        self.de0_1 = TRANS_Block(in_channels=128, out_channels=64, qkv_dim=64, num_frames=120, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        self.de0_2 = TRANS_Block(in_channels=256, out_channels=128, qkv_dim=64, num_frames=60, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        self.de0_3 = TRANS_Block(in_channels=512, out_channels=256, qkv_dim=64, num_frames=30, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)
        self.de0_4 = TRANS_Block(in_channels=1024, out_channels=512, qkv_dim=64, num_frames=15, num_joints=num_joints, num_heads=num_heads, kernel_size=kernel_size)

        

        
        

        self.downsample = nn.AvgPool2d(kernel_size=(2,1))

        self.upsample = nn.Upsample(scale_factor=(2,1), mode='nearest')

        self.up1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.up2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
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
        #print('input shape 1: ',x.shape)
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        #print('input shape 2: ',x.shape)
        x = x.view(x.size(0), x.size(1), T, V)
        #print('input shape 3: ',x.shape)
        x = self.input_map(x) #(32, 32, 120, 25)
        #print('input shape 4: ',x.shape)
        
        #Position Embedding
        #x = self.pes(x) + x if self.use_pes else x  #(32, 32, 120, 25)
        #print('input shape(position embed): ',x.shape)


        #Encoding
        
        x0_0 = self.en0_0(x)          #(32, 64, 120, 25)
        
        do0_0 = self.downsample(x0_0) #(32, 64, 60, 25)

        x1_0 = self.en1_0(do0_0)      #(32, 128, 60, 25)
        do1_0 = self.downsample(x1_0) #(32, 128, 30, 25)

        x2_0 = self.en2_0(do1_0)      #(32, 256, 30, 25)
        do2_0 = self.downsample(x2_0) #(32, 256, 15, 25)
  
        x3_0 = self.en3_0(do2_0)      #(32, 512, 15, 25)
        
        x4_0 = self.en4_0(x3_0)       #(32, 1024, 15, 25)

        #Decoding
  
        up4_0 = self.up4(x4_0)                  #(32, 512, 15, 25)
        x3_1 = self.de0_4(torch.cat((x3_0, up4_0), dim=1))       #(32, 512, 15, 25)
        
        up3_1 = self.upsample(x3_1)             #(32, 512, 30, 25)
        up3_1 = self.up3(up3_1)                 #(32, 256, 30, 25)
        x2_2 = self.de0_3(torch.cat((x2_0, up3_1), dim=1))       #(32, 256, 30, 25)
        #print('x1_0.shape: ', x1_0.shape)
        
        up2_2 = self.upsample(x2_2)             #(32, 256, 60, 25)
        up2_2 = self.up2(up2_2)                 #(32, 128, 60, 25)
        #print('up2_2.shape: ', up2_2.shape)
        x1_3 = self.de0_2(torch.cat((x1_0, up2_2), dim=1))        #(32, 128, 60, 25)
        
        #print('x1_3.shape: ', x1_3.shape)
       
        up1_3 = self.upsample(x1_3)             #(32, 128, 120, 25)
        up1_3 = self.up1(up1_3)                 #(32, 64, 120, 25)
        x0_4 = self.de0_1(torch.cat((x0_0, up1_3), dim=1))       #(32, 64, 120, 25)

        #Final Layer

        

        #final_4
        final_4 = x0_4.view(N, M, self.out_channels, -1)
        final_4 = final_4.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)
        final_4 = self.drop_out2d(final_4)
        final_4 = final_4.mean(3).mean(1)
        final_4 = self.drop_out(final_4)

        #final_4 = self.fc(final_4)

        #final = (final_1 + final_2 + final_3 + final_4) /4
        



        

        # NM, C, T, V
        #x = x0_4.view(N, M, self.out_channels, -1)
        #x = x.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)
        #x = self.drop_out2d(x)
        #x = x.mean(3).mean(1)

        #x = self.drop_out(x)
        #print('output.shape: ', x)

        return self.fc(final_4)
