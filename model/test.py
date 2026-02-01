#test.py
import torch 
import torch.nn as nn


class TRANS_Block(nn.Module):
    def __init__(self, in_channels, out_channels, qkv_dim,
                 num_frames, num_joints, num_heads, 
                 kernel_size, use_pes=True, att_drop=0):#use_pes=True
        super().__init__()
        self.qkv_dim = qkv_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        #self.use_pes = use_pes
        pads = int((kernel_size[1] - 1) / 2)
        #padt = int((kernel_size[0] - 1) / 2)
       

        

        #transfer to Q,K,V
        self.to_qkvs = nn.Conv2d(in_channels, 2 * num_heads * qkv_dim, 1, bias=True)

        self.alphas = nn.Parameter(torch.ones(1, num_heads, 1, 1), requires_grad=True)

        self.att0s = nn.Parameter(torch.ones(1, num_heads, num_joints, num_joints) / num_joints, requires_grad=True)


        self.out_nets = nn.Sequential(nn.Conv2d(in_channels * num_heads, out_channels, (1, kernel_size[1]), padding=(0, pads)), nn.BatchNorm2d(out_channels))
        self.ff_net = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))

        self.tan = nn.Tanh()
        self.relu = nn.ReLU()


        if in_channels != out_channels:
            self.ress = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
            
        else:
            self.ress = lambda x: x
           

        self.drop = nn.Dropout(att_drop)



    def forward(self, x):
        N, C, T, V = x.size()

        # Position Embed
        #xs = self.pes(x) + x if self.use_pes else x

        #get Q, K
        q, k = torch.chunk(self.to_qkvs(x).view(N, 2 * self.num_heads, self.qkv_dim, T, V), 2, dim=1)
        #print('q.shape: ', q.shape)

        #attention
        attention = self.tan(torch.einsum('nhctu,nhctv->nhuv', [q, k]) / (self.qkv_dim * T)) * self.alphas
        #print('attention.shape: ', attention.shape)
        attention = attention + self.att0s.repeat(N, 1, 1, 1)
        #print('attention.shape: ', attention.shape)
        attention = self.drop(attention)
        #print('attention.shape: ', attention.shape)
        xs = torch.einsum('nctu,nhuv->nhctv', [x, attention]).contiguous().view(N, self.num_heads * self.in_channels, T, V)
        #print('xs.shape: ', xs.shape)
        x_ress = self.ress(x)
        #print('xs.shape: ', xs.shape)
        xs = self.relu(self.out_nets(xs) + x_ress)
        #print('xs.shape: ', xs.shape)
        xs = self.relu(self.ff_net(xs) + xs) #xs = self.relu(self.ff_net(xs) + x_ress)

        return xs


        
        

num_frames = num_frames 
num_joints = num_joints 

en1 = TRANS_Block(512, 256, 64, num_frames=120, num_joints=25, num_heads=3, kernel_size=[3, 5], att_drop=0)
en2 = TRANS_Block(256, 128, 64, num_frames=120, num_joints=25, num_heads=3, kernel_size=[3, 5], att_drop=0)
en3 = TRANS_Block(128, 64, 64, num_frames=120, num_joints=25, num_heads=3, kernel_size=[3, 5], att_drop=0)
en4 = TRANS_Block(64, 32, 64, num_frames=120, num_joints=25, num_heads=3, kernel_size=[3, 5], att_drop=0)
en5 = TRANS_Block(32, 16, 64, num_frames=120, num_joints=25, num_heads=3, kernel_size=[3, 5], att_drop=0)

de1 = TRANS_Block(16, 32, 128, num_frames=120, num_joints=25, num_heads=3, kernel_size=[3, 5], att_drop=0)
de2 = TRANS_Block(32, 64, 128, num_frames=120, num_joints=25, num_heads=3, kernel_size=[3, 5], att_drop=0)
de3 = TRANS_Block(64, 128, 128, num_frames=120, num_joints=25, num_heads=3, kernel_size=[3, 5], att_drop=0)
de4 = TRANS_Block(128, 256, 128, num_frames=120, num_joints=25, num_heads=3, kernel_size=[3, 5], att_drop=0)
de5 = TRANS_Block(256, 512, 128, num_frames=120, num_joints=25, num_heads=3, kernel_size=[3, 5], att_drop=0)

fc = nn.Linear(512, 120)
drop_out = nn.Dropout(dropout)
drop_out2d = nn.Dropout2d(dropout2d)

    def forward(self, x):
        x = torch.randn(32, 512, 120, 25)
        y = torch.randn(32, 512, 120, 25)
        x1 = torch.randn(2, 3)
        y1 = torch.randn(2, 3)
        #print('x: ', x)
        print('x.shape: ', x.shape)
        #print('y: ', y)
        print('y.shape: ', y.shape)
        print('x1: ', x1)
        print('x1.shape: ', x1.shape)
        print('y1: ', y1)
        print('y1.shape: ', y1.shape)

        x1_0 = self.en1(x)
        x2_0 = self.en2(x1_0)
        x3_0 = self.en3(x2_0)
        x4_0 = self.en4(x3_0)
        x5_0 = self.en5(x4_0)

        x0_1 = self.de1(x5_0)
        x0_2 = self.de2(x0_1 + x4_0)
        x0_3 = self.de3(x0_2 + x3_0)
        x0_4 = self.de4(x0_3 + x2_0)
        x0_5 = self.de5(x0_4 + x1_0)

        print('x1_0.shape: ', x1_0.shape)
        print('x2_0.shape: ', x2_0.shape)
        print('x3_0.shape: ', x3_0.shape)
        print('x4_0.shape: ', x4_0.shape)
        print('x5_0.shape: ', x5_0.shape)

        print('x0_1.shape: ', x0_1.shape)
        print('x0_2.shape: ', x0_2.shape)
        print('x0_3.shape: ', x0_3.shape)
        print('x0_4.shape: ', x0_4.shape)
        print('x0_5.shape: ', x0_5.shape)
       
        

        

