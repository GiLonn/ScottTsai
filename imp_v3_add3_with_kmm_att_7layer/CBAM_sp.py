import torch
import math
import torch.nn as nn
import torch.nn.functional as F
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self, channel):
        super(SpatialGate, self).__init__()
        kernel_size = 5
        k_size = 5
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = x * self.sigmoid(x_out) # broadcasting

        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        scale2 = x * self.sigmoid(y)

        return scale + scale2

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        #self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        #self.no_spatial=no_spatial
        #if not no_spatial:
        self.SpatialGate = SpatialGate(channel)
    def forward(self, x):
        #x_out = self.ChannelGate(x)
        #if not self.no_spatial:
        x = self.SpatialGate(x)
        return x