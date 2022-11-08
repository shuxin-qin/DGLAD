import torch
import torch.nn as nn
import math

from modules import ConvLayer, ReconstructionModel
from transformer import EncoderLayer
from graph import GraphEmbedding

'''
results, 2graph, with fusion  win=64  :    layer=1  transformer decoder
(F1, P, R)
--msl: (95.31,97.21,93.48)--20220906_125118    time:6.7     
--smap:(94.41,97.95,91.11)--20220906_165157    time:7.6
--swat:(89.18,95.05,83.99)--20220926_095734    time:10.7 
--wadi:(84.15,80.76,87.84)--20220919_112954    time:27.2
-------------------------------------------- 

results, 2graph, with fusion  win=8  :
(F1, P, R)
--msl: (95.34,95.24,95.43)--20220919_165900   1.6
--smap:(90.04,95.07,85.51)--20220919_172545   3.2
--swat:(88.06,91.53,84.85)--20220919_165015   2.8  1.3
--wadi:(81.09,85.52,77.10)--20220919_112139   4.3
-------------------------------------------
results, 2graph, with fusion  win=16  :
(F1, P, R)
--msl: (94.63,93.86,95.43)--20220919_181007   2.0
--smap:(89.10,85.57,92.93)--20220919_181631   3.6
--swat:(88.58,98.14,80.72)--20220926_102227   3.4  1.5
--wadi:(84.68,93.90,77.10)--20220919_112139   6.6
-------------------------------------------
results, 2graph, with fusion  win=32  :
(F1, P, R)
--msl: (93.96,95.80,92.18)--20220907_155021   3.2
--smap:(93.16,96.68,89.90)--20220907_155608   4.2
--swat:(91.07,96.53,86.19)--20220926_092253   5.2  2.4
--wadi:(82.45,77.69,87.84)--20220919_110830   12.7
--------------------------------------------
results, 2graph, with fusion  win=128  :
(F1, P, R)
--msl: (92.44,94.04,90.88)--20220907_165329   18.1
--smap:(90.24,84.17,97.26)--20220907_175200   18.5
--swat:(87.47,89.77,85.28)--20220926_123842   28.7
--wadi:(84.00,80.49,87.84)--20220926_134812   68.9
-------------------------------------------
-------------------------------------------


results, 2graph, without fusion  win=64  :
(F1, P, R)
--msl: (94.01,94.54,93.48)--20220914_181156  time: 6.6
--smap:(92.31,93.32,91.32)--20220915_091412        7.4
--swat:(85.95,95.00,78.48)--20220915_092728        10.6
--wadi:(82.94,78.56,87.84)--20220920_144535   27.1

results, only using intra graph, without fusion  win=64  :
(F1, P, R)
--msl: (87.26,82.83,92.18)--20220915_110950  time: 4.4
--smap:(84.96,96.07,76.15)--20220915_105211        4.7
--swat:(85.02,95.28,76.75)--20220915_103148        7.3
--wadi:(80.76,74.73,87.84)--20220920_135141   14.1

results, only using inter graph, without fusion  win=64  :
(F1, P, R)
--msl: (87.40,83.10,92.18)--20220915_114309 
--smap:(88.52,91.77,85.49)--20220915_122657
--swat:(83.36,89.93,77.69)--20220915_135556
--wadi:(76.48,67.72,87.84)--20220920_141439   19.1

results, without transformer (use rnn)  win=64  
(F1, P, R)
--msl: (95.08,96.06,94.13)--20220914_150134    time:6.5     
--smap:(91.07,91.98,90.19)--20220921_103000    time:7.3   20220914_160805
--swat:(88.18,90.32,86.13)--20220914_142650    time:10.3
--wadi:(79.84,73.17,87.86)--20220920_150811   27.5
--------------------------------------------

'''


class MODEL_2GRAPH_TRANS(nn.Module):
    """ MODEL_2GRAPH_TRANS model class.
    """

    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        kernel_size=7,
        dropout=0.2
    ):
        super(MODEL_2GRAPH_TRANS, self).__init__()
        self.window_size = window_size
        self.out_dim = out_dim
        self.f_dim = n_features

        self.use_inter_graph = True
        self.use_intra_graph = True
        self.use_fusion = True
        self.decoder_type = 0  # 0:transformer   1:rnn

        # preprocessing
        self.conv = ConvLayer(n_features, kernel_size)

        self.num_levels = 1
        # inter embedding module based on GNN
        if self.use_inter_graph:
            self.inter_module = GraphEmbedding(n_features, window_size, num_levels=self.num_levels, device=torch.device('cuda:0'))
        
        # intra embedding module based on GNN
        if self.use_intra_graph:
            self.intra_module = GraphEmbedding(window_size, n_features, num_levels=self.num_levels, device=torch.device('cuda:0'))

        # learned fusion parameter
        if self.use_fusion:
            self.inter_para = nn.Parameter(torch.FloatTensor(self.window_size, self.f_dim), requires_grad=True)
            self.intra_para = nn.Parameter(torch.FloatTensor(self.window_size, self.f_dim), requires_grad=True)
            self.init_parameters()

        # fuse two branch
        self.fusion_linear = nn.Linear(self.f_dim*2, self.f_dim)

        # decoder
        if self.decoder_type == 0:
            self.decoder = EncoderLayer(n_feature=self.f_dim, num_heads=1, hid_dim=self.f_dim, dropout=dropout)
        elif self.decoder_type == 1:
            self.decoder = ReconstructionModel(in_dim=self.f_dim, hid_dim=100, out_dim=self.f_dim, n_layers=1, dropout=dropout)
        self.linear = nn.Linear(self.f_dim, self.out_dim)


    def init_parameters(self):
        stdv = 1. / math.sqrt(self.inter_para.size(1))
        self.inter_para.data.uniform_(-stdv, stdv)
        self.intra_para.data.uniform_(-stdv, stdv)


    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # print('x:', x.shape)
        x = self.conv(x)

        # intra module
        if self.use_intra_graph:
            enc_intra = self.intra_module(x.permute(0, 2, 1)).permute(0, 2, 1)   # >> (b, n, k)
            #self.enc_intra = enc_intra
        # inter module
        if self.use_inter_graph:
            enc_inter = self.inter_module(x)  # >> (b, n, k)
            #self.enc_inter = enc_inter
        
        if self.use_inter_graph and self.use_intra_graph:
            if self.use_fusion:
                enc_inter = torch.mul(self.inter_para, enc_inter)
                enc_intra = torch.mul(self.intra_para, enc_intra)
            enc = torch.cat([enc_inter, enc_intra], dim=-1)
            enc = self.fusion_linear(enc)
        else:
            if self.use_intra_graph:
                enc = enc_intra
            elif self.use_inter_graph:
                enc = enc_inter

        # decoder
        if self.decoder_type == 0:
            dec, _ = self.decoder(enc)
        elif self.decoder_type == 1:
            dec = self.decoder(enc)
        out = self.linear(dec)

        return out



