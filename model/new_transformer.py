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
        
        self.decoders = nn.ModuleList()
        self.dense_connections = nn.ModuleList()
        


        #Encoder Blocks
        for idx, (in_channels, out_channels, qkv_dim) in enumerate(self.encoder_config):
            print('in_channel: ', in_channels)
            print('out_channel: ', out_channels)
            print('qkv_dim: ', qkv_dim)
            print('########################')
            self.encoders.append(TRANS_Block(in_channels, out_channels, qkv_dim, 
                                         num_frames=num_frames, 
                                         num_joints=num_joints, 
                                         num_heads=num_heads,
                                         kernel_size=kernel_size,
                                         use_pes=use_pes,
                                         att_drop=att_drop))
            if idx > 0:
                self.dense_connections.append(
                    nn.Conv2d(self.encoder_config[idx - 1][1], out_channels, 1)
                )
        print('bottleneck_config: ', bottleneck_config)
            


        self.bottleneck = TRANS_Block(in_channels=bottleneck_config[0], out_channels=bottleneck_config[1], qkv_dim=bottleneck_config[2], 
                                         num_frames=num_frames, 
                                         num_joints=num_joints, 
                                         num_heads=num_heads,
                                         kernel_size=kernel_size,
                                         use_pes=use_pes,
                                         att_drop=att_drop)
            
        #Decder Blocks
        for index, (in_channels, out_channels, qkv_dim) in enumerate(self.decoder_config):
            print('in_channel: ', in_channels)
            print('out_channel: ', out_channels)
            print('qkv_dim: ', qkv_dim)
            print('########################')
            self.decoders.append(TRANS_Block(in_channels, out_channels, qkv_dim, 
                                         num_frames=num_frames, 
                                         num_joints=num_joints, 
                                         num_heads=num_heads,
                                         kernel_size=kernel_size,
                                         use_pes=use_pes,
                                         att_drop=att_drop))   

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
        encoder_features = []

        for iex, encoder in enumerate(self.encoders):
            x = encoder(x)
            #print('x.shape: ', x.shape)
            #print('iex: ', iex)
            encoder_features.append(x)
            if iex > 0:
                x = x + self.dense_connections[iex - 1](encoder_features[iex -1])
          

        #BottleNeck
        x = self.bottleneck(x)
            #print('x.shape: ', x.shape)

        #merge1
        x = x + encoder_features[3]
        
        

        #Decoding
        for idx, decoder in enumerate(self.decoders):
            #x = block(x)
            #print('x.shape: ', x.shape)
            #print('idx: ', idx)
            skip_connection = encoder_features[-(idx + 1)]
            x = decoder(x + skip_connection)
            
            #if idx < len(encoder_features):
                #x = x + encoder_features[-idx-1]


        

        # NM, C, T, V
        x = x.view(N, M, self.out_channels, -1)
        x = x.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)
        x = self.drop_out2d(x)
        x = x.mean(3).mean(1)

        x = self.drop_out(x)
        #print('output.shape: ', x)

        return self.fc(x)
