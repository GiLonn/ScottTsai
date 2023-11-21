from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader
import resnet_parameter as models
import numpy as np
import time

import KMM_Lin
from sklearn.metrics import confusion_matrix

from torchsummary import summary


        



if __name__ == '__main__':
    model = models.MFSAN(num_classes=2).cuda()

    summary(model,[(3,224,224), (3,224,224)])