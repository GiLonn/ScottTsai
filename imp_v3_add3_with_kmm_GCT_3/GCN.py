import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import math

class GCN(nn.Module):
    def __init__(self,c,out_c,k=7): #out_Channel=21 in paper
        super(GCN, self).__init__()
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.conv_l1 = nn.Conv2d(c, out_c, kernel_size=(k,1), padding =(3,0))
        self.conv_l2 = nn.Conv2d(out_c, out_c, kernel_size=(1,k), padding =(0,3))
        self.conv_r1 = nn.Conv2d(c, out_c, kernel_size=(1,k), padding =(3,0))
        self.conv_r2 = nn.Conv2d(out_c, out_c, kernel_size=(k,1), padding =(0,3))
        nn.init.kaiming_normal_(self.conv_l1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_l2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_r1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_r2.weight, mode='fan_out', nonlinearity='relu')

        self.bn.weight.data.fill_(1)  
        self.bn.bias.data.zero_()



        
    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_l = self.bn(x_l)
        x_l = self.relu(x_l)
        
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x_r = self.bn(x_r)
        x_r = self.relu(x_r)
        
        x = x_l + x_r
        x = self.bn(x)
        
        return x