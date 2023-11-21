import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
from torchinfo import summary
import backbone as bk


#resnet18 = models.resnet18(pretrained = True)
model = bk.ResNet18Fc()
summary(model, (8, 3, 224, 224))
#print (resnet18)
