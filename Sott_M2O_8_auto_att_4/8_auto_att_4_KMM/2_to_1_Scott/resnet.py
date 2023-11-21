import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import mmd_AMRAN as mmd 
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
import torch
import backbone_multi as bk
import call_resnet18_multi as cl
import LabelSmoothing as LS
#import CBAM_sp as att
import ECANet as att
import ECANet2 as att2



__all__ = ['ResNet', 'resnet18']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    # 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, groups=groups, bias=False, dilation=dilation)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Deconv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(Deconv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, bias = False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Depthwise_conv2d(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(Depthwise_conv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, bias = False, groups = in_channels, **kwargs)
        #self.bn = nn.BatchNorm2d(in_channels, eps=0.001)
        #self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        #x = self.bn(x)
        #x = self.relu(x)
        return(x)

class Pointwise_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super (Pointwise_conv2d, self).__init__()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size = 1, bias = False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return(x)




class Deconv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(Deconv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, bias = False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

#def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    #return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     #padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class conv_M2(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_M2, self).__init__()
        self.res = conv1x1(in_channels, 256)
        self.branch1_1 = BasicConv2d(in_channels, 64, kernel_size = 1, stride = 1)
        self.branch1_2_1 = Depthwise_conv2d(64, kernel_size = 3, stride = 1, padding = 1, dilation = 1)
        self.branch1_2_2 = Pointwise_conv2d(64, 96, stride = 1, padding = 1, dilation = 1)
        self.branch1_3_1 = Depthwise_conv2d(96, kernel_size = 3, stride = 1, padding = 1, dilation = 1)
        self.branch1_3_2 = Pointwise_conv2d(96, 96, stride = 1, padding = 1, dilation = 1)
        att_list1 = [att.eca_layer(64)]
        self.att_layer1 = nn.Sequential(*att_list1)
        att_list2 = [att2.eca_layer(96)]
        self.att_layer2 = nn.Sequential(*att_list2)



        #self.branch1_4_1 = Depthwise_conv2d(160, kernel_size = 3, stride = 1, padding = 1, dilation = 1)
        #self.branch1_4_2 = Pointwise_conv2d(160, 192, stride = 1, padding = 1, dilation = 1)
        self.branch1_4 = nn.Dropout(0.2)

        self.branch2_1 = BasicConv2d(in_channels, 64, kernel_size = 1, stride = 1)
        self.branch2_2_1 = Depthwise_conv2d(64,  kernel_size = 3, stride = 1, padding = 2, dilation = 2)
        self.branch2_2_2 = Pointwise_conv2d(64, 96, stride = 1, padding = 2, dilation = 2)
        self.branch2_3_1 = Depthwise_conv2d(96, kernel_size = 3, stride = 1, padding = 2, dilation = 2)
        self.branch2_3_2 = Pointwise_conv2d(96, 96, stride = 1, padding = 2, dilation = 2)
        #self.branch2_4_1 = Depthwise_conv2d(160, kernel_size = 3, stride = 1, padding = 2, dilation = 2)
        #self.branch2_4_2 = Pointwise_conv2d(160, 192, stride = 1, padding = 2, dilation = 2)
        self.branch2_4 = nn.Dropout(0.2)
        att_list3 = [att.eca_layer(64)]
        self.att_layer3 = nn.Sequential(*att_list3)
        att_list4 = [att2.eca_layer(96)]
        self.att_layer4 = nn.Sequential(*att_list4)





        #self.branch3_1 = BasicConv2d(in_channels, 96, kernel_size = 1, stride = 1)
        #self.branch3_2_1 = Depthwise_conv2d(96,  kernel_size = 3, stride = 1, padding = 3, dilation = 3)
        #self.branch3_2_2 = Pointwise_conv2d(96, 160, stride = 1, padding = 2, dilation = 2)
        #self.branch3_3_1 = Depthwise_conv2d(160, kernel_size = 3, stride = 1, padding = 3, dilation = 3)
        #self.branch3_3_2 = Pointwise_conv2d(160, 160, stride = 1, padding = 2, dilation = 2)

        self.branch3_1 = BasicConv2d(in_channels, 64, kernel_size = 1, stride = 1)
        self.branch3_2 = Deconv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        self.branch3_3 = Deconv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        #self.branch3_4 = Deconv2d(96, 128, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        self.branch3_4 = nn.Dropout(0.2)
        #self.branch4 = nn.Dropout(0.2)

    def forward(self, x):
        x_residual = self.res(x)
        x_branch1 = self.branch1_1(x)
        x_branch1 = self.branch1_2_1(x_branch1)
        x_branch1 = self.att_layer1(x_branch1)
        x_branch1 = self.branch1_2_2(x_branch1)
        x_branch2 = self.branch2_1(x)
        x_branch2 = self.branch2_2_1(x_branch2)
        x_branch2 = self.att_layer3(x_branch2)

        x_branch2 = self.branch2_2_2(x_branch2)
        x_branch2 = x_branch1 + x_branch2

        x_branch1 = self.branch1_3_1(x_branch1)
        x_branch1 = self.att_layer2(x_branch1)



        x_branch1 = self.branch1_3_2(x_branch1)
        x_branch1 = self.branch1_4(x_branch1)

        #x_branch1 = self.branch1_4_2(x_branch1)
        #_branch1 = self.branch1_5(x_branch1)
        x_branch2 = self.branch2_3_1(x_branch2)
        x_branch2 = self.att_layer4(x_branch2)

        x_branch2 = self.branch2_3_2(x_branch2)
        x_branch2 = self.branch2_4(x_branch2)

        #x_branch2 = self.branch2_4_1(x_branch2)
        #x_branch2 = self.branch2_4_2(x_branch2)
        #x_branch2 = self.branch2_5(x_branch2)



        #x_branch2 = self.branch2_4(x_branch2)
        x_branch2 = x_branch1 + x_branch2

        x_branch3 = self.branch3_1(x)
        x_branch3 = self.branch3_2(x_branch3)
        #x_branch3 = self.branch3_2_2(x_branch3)

        x_branch3 = self.branch3_3(x_branch3)
        #x_branch3 = self.branch3_3_2(x_branch3)
        x_branch3 = self.branch3_4(x_branch3)
        #x_branch3 = self.branch3_5(x_branch3)


        #x_branch3 = self.branch3_4(x_branch3)

        #x = torch.cat([x_branch1, x_branch2, x_branch3], 1)
        x = torch.cat([x_branch1, x_branch2, x_branch3], 1)
        return x + x_residual

class ADDneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ADDneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, bias=False, padding = 1)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.dr = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        #self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, bias=False, padding = 1)
        #self.bn3 = nn.BatchNorm2d(planes)
        #self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, bias=False, padding = 1)
        #self.bn4 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride


        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        #nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        #nn.init.kaiming_normal_(self.conv4.weight, mode='fan_out', nonlinearity='relu')

        self.bn1.weight.data.fill_(1)  
        self.bn1.bias.data.zero_()
        self.bn2.weight.data.fill_(1)  
        self.bn2.bias.data.zero_()
        #self.bn3.weight.data.fill_(1)  
        #self.bn3.bias.data.zero_()
        #self.bn4.weight.data.fill_(1)  
        #self.bn4.bias.data.zero_()




    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #out = self.dr(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        #out = self.dr(out)

        #out = self.conv3(out)
        #out = self.bn3(out)
        #out = self.relu(out)

        #out = self.conv4(out)
        #out = self.bn4(out)
        #out = self.relu(out)
        #out = self.dr(out)
        

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class MFSAN(nn.Module):

    def __init__(self, num_classes=2):
        super(MFSAN, self).__init__()
        # self.sharedNet = resnet50(True)
        self.sharedNet = resnet18(True)
        #self.sharedNet = cl.load_resnet18_multi()
        # self.sonnet1 = ADDneck(2048, 256)
        # self.sonnet2 = ADDneck(2048, 256)
        multi_list = [conv_M2(512, 256), BasicConv2d(256, 256, kernel_size = 1, stride =1)]
        multi_list2 = [conv_M2(512, 256),BasicConv2d(256, 256, kernel_size = 1, stride =1)]
        self.sonnet1 = nn.Sequential(*multi_list)
        self.sonnet2 = nn.Sequential(*multi_list2)

        #self.sonnet1 = conv_M2(512, 256)
        #self.sonnet2 = conv_M2(512, 256)
        #self.sonnet1 = ADDneck(256, 128)
        #self.sonnet2 = ADDneck(256, 128)
        fc_list1 = [nn.Dropout(0.1), nn.Linear(256, num_classes)]
        fc_list2 = [nn.Dropout(0.1), nn.Linear(256, num_classes)]
        self.cls_fc_son1 = nn.Sequential(*fc_list1)
        self.cls_fc_son2 = nn.Sequential(*fc_list2)
        #print(self.sonnet1.conv1.weight)

        nn.init.xavier_normal_(self.cls_fc_son1[1].weight)
        nn.init.xavier_normal_(self.cls_fc_son2[1].weight)


        #self.cls_fc_son1[0].weight.data.uniform_(0, 1)
        #self.cls_fc_son1[0].bias.data.fill_(0.0)
        #self.cls_fc_son2[0].weight.data.uniform_(0, 1)
        #self.cls_fc_son2[0].bias.data.fill_(0.0)
        #self.classifier_layer[1].weight.data.normal_(0, 0.01)
        #self.classifier_layer[1].bias.data.fill_(0.0)


        #self.cls_fc_son1 = nn.Linear(256, num_classes)
        #self.cls_fc_son2 = nn.Linear(256, num_classes)

        self.avgpool = nn.AvgPool2d(7, stride=1)


    def forward(self, data_src, test_flag, data_tgt, label_src, mu1, mu2, mark):
        mmd_loss = 0
        cmmd_loss = 0
        #criterion = torch.nn.CrossEntropyLoss()
        criterion = LS.LabelSmoothingCrossEntropy(reduction = 'sum')
        if self.training == True:

            if mark == 1:

                # data_src = self.sharedNet(data_src)
                # data_tgt = self.sharedNet(data_tgt)

                #data_src = self.sharedNet.layer3(data_src)
                #data_tgt = self.sharedNet.layer3(data_tgt)

                data_src = self.sharedNet.layer4[1](data_src)
                data_tgt = self.sharedNet.layer4[1](data_tgt)

                #data_src = self.sharedNet(data_src, test_flag)
                #data_tgt = self.sharedNet(data_tgt, test_flag)
                #print(data_tgt.size())

                ########################################## sharedNet
                data_src = self.sonnet1(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)

                data_tgt_son1 = self.sonnet1(data_tgt)
                data_tgt_son1 = self.avgpool(data_tgt_son1)
                #print(data_tgt_son1.size())
                data_tgt_son1 = data_tgt_son1.view(data_tgt_son1.size(0), -1)
                #calculate cmmd
                

                #calculate mmd
                mmd_loss += mmd.mmd_rbf_noaccelerate(data_src, data_tgt_son1)
                
                
                data_tgt_cls1 = self.cls_fc_son1(data_tgt_son1)
                data_tgt_label1 = Variable(data_tgt_cls1.data.max(1)[1])

                cmmd_loss = Variable(torch.tensor([0]))
                cmmd_loss = cmmd_loss.cuda()
                cmmd_loss = mmd.cmmd(data_src, data_tgt_son1,label_src, data_tgt_label1)
                transfer_loss = (1- mu1) * cmmd_loss + mu1 * mmd_loss

                data_tgt_son2 = self.sonnet2(data_tgt)
                data_tgt_son2 = self.avgpool(data_tgt_son2)
                data_tgt_son2 = data_tgt_son2.view(data_tgt_son2.size(0), -1)
                data_tgt_cls2 = self.cls_fc_son2(data_tgt_son2)

                
                
                l1_loss = torch.abs(torch.nn.functional.softmax(data_tgt_cls1, dim=1) - torch.nn.functional.softmax(data_tgt_cls2, dim=1))
                l1_loss = torch.mean(l1_loss)
                pred_src = self.cls_fc_son1(data_src)

                #cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                cls_loss = criterion(pred_src, label_src)

                return data_src, data_tgt_son1, data_tgt_label1, cls_loss, transfer_loss, l1_loss

            if mark == 2:
                # data_src = self.sharedNet(data_src)
                # data_tgt = self.sharedNet(data_tgt)

                #data_src = self.sharedNet.layer3(data_src)
                #data_tgt = self.sharedNet.layer3(data_tgt)

                data_src = self.sharedNet.layer4[1](data_src)
                data_tgt = self.sharedNet.layer4[1](data_tgt)
                #data_src = self.sharedNet(data_src, test_flag)
                #data_tgt = self.sharedNet(data_tgt, test_flag)


                ##########################################

                data_src = self.sonnet2(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)

                data_tgt_son2 = self.sonnet2(data_tgt)
                data_tgt_son2 = self.avgpool(data_tgt_son2)
                data_tgt_son2 = data_tgt_son2.view(data_tgt_son2.size(0), -1)
         
                mmd_loss += mmd.mmd_rbf_noaccelerate(data_src, data_tgt_son2)

                data_tgt_cls2 = self.cls_fc_son2(data_tgt_son2)
                data_tgt_label2 = Variable(data_tgt_cls2.data.max(1)[1])

                cmmd_loss = Variable(torch.tensor([0]))
                cmmd_loss = cmmd_loss.cuda()
                cmmd_loss = mmd.cmmd(data_src, data_tgt_son2,label_src, data_tgt_label2)
                transfer_loss = (1- mu2) * cmmd_loss + mu2 * mmd_loss


                data_tgt_son1 = self.sonnet1(data_tgt)
                data_tgt_son1 = self.avgpool(data_tgt_son1)
                data_tgt_son1 = data_tgt_son1.view(data_tgt_son1.size(0), -1)
                data_tgt_cls1 = self.cls_fc_son1(data_tgt_son1)

                l1_loss = torch.abs(torch.nn.functional.softmax(data_tgt_cls1, dim=1) - torch.nn.functional.softmax(data_tgt_cls2, dim=1))
                l1_loss = torch.mean(l1_loss)

                #l1_loss = F.l1_loss(torch.nn.functional.softmax(data_tgt_son1, dim=1), torch.nn.functional.softmax(data_tgt_son2, dim=1))

                pred_src = self.cls_fc_son2(data_src)
                #cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                cls_loss = criterion(pred_src, label_src)

                return data_src, data_tgt_son1, data_tgt_label2, cls_loss, mmd_loss, l1_loss

        else:
            #data = self.sharedNet(data_src, test_flag)
            data = self.sharedNet(data_src)

            fea_son1 = self.sonnet1(data)
            fea_son1 = self.avgpool(fea_son1)
            fea_son1 = fea_son1.view(fea_son1.size(0), -1)
            pred1 = self.cls_fc_son1(fea_son1)

            fea_son2 = self.sonnet2(data)
            fea_son2 = self.avgpool(fea_son2)
            fea_son2 = fea_son2.view(fea_son2.size(0), -1)
            pred2 = self.cls_fc_son2(fea_son2)

            return pred1, pred2

# def resnet50(pretrained=False, **kwargs):
#     """Constructs a ResNet-50 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
#     return model

def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


#new
'''
class conv_M2(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_M2, self).__init__()
        self.res = conv1x1(in_channels, 320)
        self.branch1_1 = BasicConv2d(in_channels, 96, kernel_size = 1, stride = 1)
        self.branch1_2_1 = Depthwise_conv2d(96, kernel_size = 3, stride = 1, padding = 1, dilation = 1)
        self.branch1_2_2 = Pointwise_conv2d(96, 128, stride = 1, padding = 1, dilation = 1)
        self.branch1_3_1 = Depthwise_conv2d(128, kernel_size = 3, stride = 1, padding = 1, dilation = 1)
        self.branch1_3_2 = Pointwise_conv2d(128, 128, stride = 1, padding = 1, dilation = 1)
        self.branch1_4 = nn.Dropout(0.2)

        self.branch2_1 = BasicConv2d(in_channels, 96, kernel_size = 1, stride = 1)
        self.branch2_2_1 = Depthwise_conv2d(96,  kernel_size = 3, stride = 1, padding = 2, dilation = 2)
        self.branch2_2_2 = Pointwise_conv2d(96, 128, stride = 1, padding = 2, dilation = 2)
        self.branch2_3_1 = Depthwise_conv2d(128, kernel_size = 3, stride = 1, padding = 2, dilation = 2)
        self.branch2_3_2 = Pointwise_conv2d(128, 128, stride = 1, padding = 2, dilation = 2)
        self.branch2_4 = nn.Dropout(0.2)

        self.branch3_1 = BasicConv2d(in_channels, 64, kernel_size = 1, stride = 1)
        self.branch3_2 = Deconv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        self.branch3_3 = Deconv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        self.branch3_4 = nn.Dropout(0.2)
        #self.branch4 = nn.Dropout(0.15)


    def forward(self, x):
        x_residual = self.res(x)
        x_branch1 = self.branch1_1(x)
        x_branch1 = self.branch1_2_1(x_branch1)
        x_branch1 = self.branch1_2_2(x_branch1)
        x_branch2 = self.branch2_1(x)
        x_branch2 = self.branch2_2_1(x_branch2)
        x_branch2 = self.branch2_2_2(x_branch2)
        x_branch2 = x_branch1 + x_branch2

        x_branch1 = self.branch1_3_1(x_branch1)
        x_branch1 = self.branch1_3_2(x_branch1)
        x_branch1 = self.branch1_4(x_branch1)
        x_branch2 = self.branch2_3_1(x_branch2)
        x_branch2 = self.branch2_3_2(x_branch2)
        x_branch2 = self.branch2_4(x_branch2)
        x_branch2 = x_branch1 + x_branch2

        x_branch3 = self.branch3_1(x)
        x_branch3 = self.branch3_2(x_branch3)
        x_branch3 = self.branch3_3(x_branch3)
        x_branch3 = self.branch3_4(x_branch3)

        #x = torch.cat([x_branch1, x_branch2, x_branch3], 1)
        x = torch.cat([x_branch1, x_branch2, x_branch3], 1)
        return x + x_residual


class resnet18_multi(nn.Module):
    def __init__(self, use_att,  num_classes = 2):
        #self.inplanes = 64  
        super(resnet18_multi, self).__init__()  
        model_resnet18 = models.resnet18(pretrained=True) 
        self.use_att = use_att
        self.conv1 = model_resnet18.conv1
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool
        self.layer1 = model_resnet18.layer1
        self.layer2 = model_resnet18.layer2
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(14, stride=1) 
        convm2_list = [conv_M2(128, 320)]
        att_list = [att.ULSAM(128, 128, 28, 28, 4)]
        self.att_layer = nn.Sequential(*att_list) 
        self.convm2_layer = nn.Sequential(*convm2_list)
        self.linear_test = BasicConv2d(320, 256, kernel_size = 1, stride =1)
       

    def forward(self, x, test_flag):
        if test_flag:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool1(x)
            x = self.layer1(x)
            x = self.layer2(x)

        if self.use_att:
            #print("Y")
            x = self.att_layer(x)

        x = self.convm2_layer(x)
        #x = self.convm2_layer_2(x)
        x = self.maxpool1(x)
        #print(x.size())
        x = self.linear_test(x)

        #x = self.layer3(x)
        

        #x = self.layer4(x)
        
        #x = self.convm3_layer(x)
        #x = self.convm3_layer_2(x)
        #x = self.maxpool1(x)
        #x = self.linear_test_2(x)
        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)


        return x

'''