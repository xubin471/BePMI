import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class Channel_att(nn.Module):
    def __init__(self, channel=512, b=1, gamma=2):
        super().__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.scaler = 0.05

        # channel attention
        self.conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=kernel_size,
                     padding=(kernel_size-1)//2, bias=False),
            nn.Sigmoid()
        )
        self.sim_scale = nn.Parameter(torch.tensor(1.0))  # 初始缩放=1
        self.sim_bias = nn.Parameter(torch.tensor(0.0))  # 初始偏移=0

        # adaptive weight
        self.weight_avg = nn.Parameter(torch.tensor(0.7))

    def forward(self, x,gl_proto):
        """
        Args:
            x = fts  (1 512 64 64)
            gl_proto  (1 512)
        Return:
            enhanced_fts: (1 512 64 64)
        """
        """Compute Similarity Score"""
        feat_norm = F.normalize(x, p=2, dim=1)
        protos_norm = F.normalize(gl_proto, p=2, dim=1)[:, :, None, None]
        sim = F.conv2d(feat_norm, protos_norm) # compute similarity score via conv
        sim = self.sim_scale * sim + self.sim_bias

        # feature merge
        avg_feat = self.avg_pool(x*sim) #(1 512 1 1)
        max_feat = self.max_pool(x*sim) #(1 512 1 1)
        attn = torch.sigmoid(self.weight_avg) * avg_feat + (1 - torch.sigmoid(self.weight_avg)) * max_feat
        attn = F.softmax(attn/self.scaler ,dim =1)

        return x * attn #(1 512 64 64)

