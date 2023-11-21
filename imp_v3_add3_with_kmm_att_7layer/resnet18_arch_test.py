import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
from torchinfo import summary
import backbone_multi as bk


#resnet18 = models.resnet18(pretrained = True)
model = bk.resnet18_multi()
summary(model, (8, 128, 28, 28))
#print (resnet18)
