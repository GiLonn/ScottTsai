import torch
import torch.nn as nn
import torchvision
# from Coral import CORAL
from torch.autograd import Variable
from coral_pytorch import CORAL
import mmd_AMRAN as mmd
import mmd as md
# import mmd2_correct as mmd
import backbone_multi
import backbone
import call_resnet18_multi as cl
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ResNet-18在最初時沒有使用bottleneck
class Transfer_Net(nn.Module):


    def __init__(self, num_class, base_net='resnet18_multi_new', transfer_loss='cmmd', use_bottleneck=True, bottleneck_width=128, width=512):
        super(Transfer_Net, self).__init__()

        self.base_network = backbone_multi.network_dict[base_net]() # 從backbone已經定義好的resnet-18到fully-connected之前
        #print(self.base_network.weight)
        self.use_bottleneck = use_bottleneck # 是否用bottleneck (這邊指的不是resnet-18的bottleneck，而是論文DDC的fully-connected)
        self.transfer_loss = transfer_loss
        #import pdb; pdb.set_trace() 
        ''' 
        bottleneck_list = [nn.Linear(256, bottleneck_width),nn.ReLU(), nn.Dropout(0.15), nn.BatchNorm1d(bottleneck_width),nn.ReLU(),nn.Linear(bottleneck_width, 128), nn.Dropout(0.15)]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        self.bottleneck_layer[3].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[3].bias.data.fill_(0.1)
        '''
        
        if base_net == 'resnet18_multi_new':
            self.base_network = cl.load_resnet18_multi()
            #print("XXXX")


        
        #classifier_layer_list = [nn.Linear(256, width), nn.ReLU(), nn.Dropout(0.15), nn.Linear(width, 128), nn.ReLU(), nn.Dropout(0.15), nn.Linear(128, 2)]
        bottle_list = [nn.Linear(256, width), nn.Linear(width, bottleneck_width)]
        classifier_list = [nn.Dropout(0.25), nn.Linear(bottleneck_width, num_class)]

        self.bottle_layer = nn.Sequential(*bottle_list)
        self.classifier_layer = nn.Sequential(*classifier_list)
        '''
        for i in range(2): # classifier[0]、classifier[3] = nn.Linear
            self.classifier_layer[i * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[i * 3].bias.data.fill_(0.0)
        '''
        self.bottle_layer[0].weight.data.normal_(0, 0.01)
        self.bottle_layer[0].bias.data.fill_(0.0)
        self.bottle_layer[1].weight.data.normal_(0, 0.01)
        self.bottle_layer[1].bias.data.fill_(0.0)
        self.classifier_layer[1].weight.data.normal_(0, 0.01)
        self.classifier_layer[1].bias.data.fill_(0.0)

    def forward(self, source, target, s_label, mu, test_flag):
        ##source = self.base_network.convm2_layer(source)
        ##source = self.base_network.convm2_layer_2(source)
        ##source = self.base_network.maxpool1(source)
        ##source = self.base_network.linear_test(source)
        #source = self.base_network.layer4(source)

        #source = self.base_network.convm3_layer(source)
        #source = self.base_network.convm3_layer_2(source)
        #source = self.base_network.maxpool1(source)
        #source = self.base_network.linear_test_2(source)
        ##source = self.base_network.avgpool(source)


        ##target = self.base_network.convm2_layer(target)
        ##target = self.base_network.convm2_layer_2(target)
        ##target = self.base_network.maxpool1(target)
        ##target = self.base_network.linear_test(target)
        #target = self.base_network.layer4(target)
        #target = self.base_network.convm3_layer(target)
        #target = self.base_network.convm3_layer_2(target)
        #target = self.base_network.maxpool1(target)
        #target = self.base_network.linear_test_2(target)
        ##target = self.base_network.avgpool(target)
        
        ##source = source.view(source.size(0), -1)
        ##target = target.view(target.size(0), -1)
        #print(s_label.requires_grad)
        source = self.base_network(source, test_flag)
        #source = source.detach()
        source = self.bottle_layer(source)
        with torch.no_grad():
            source_bottle = source
        #source_bottle = source.clone().detach()
        #source_bottle.requires_grad_ = False
        #print(source_bottle.requires_grad)
        #print(source_bottle.requires_grad)
        source_clf = self.classifier_layer(source)
        target = self.base_network(target, test_flag)
        #target = target.detach()
        target = self.bottle_layer(target) 
        with torch.no_grad():
            target_bottle = target
        print(target_bottle.requires_grad)  
        #target_bottle = target.clone().detach()
        #target_bottle.requires_grad_ = False
        #Target_bottle
        t_label = self.classifier_layer(target)
        t_label = Variable(t_label.detach().max(1)[1])
        #print(source.requires_grad)
        transfer_loss = self.adapt_loss(source_bottle, target_bottle, self.transfer_loss, s_label, t_label, mu)
        #transfer_loss = self.adapt_loss(source_bottle, target_bottle, self.transfer_loss, s_label, mu)
        #del source_bottle
        #del target_bottle
        

         #predict target label using iteration for cmmd

        
        #if self.use_bottleneck:
            #source = self.bottleneck_layer(source)
            #target = self.bottleneck_layer(target)
        
        
        
        
        return source, target,source_clf, t_label, transfer_loss
        #return source, target,source_clf, transfer_loss

    def predict(self, x, test_flag):

        features = self.base_network(x, test_flag)
        #print(features.requires_grad)
        features = self.bottle_layer(features)
        #print(features.size())
        clf = self.classifier_layer(features)
        return clf

    def adapt_loss(self, X, Y, adapt_loss, s_label,t_label, mu):

        if adapt_loss == 'mmd':
            transfer_loss = mmd.mmd_rbf_noaccelerate(X, Y)
            #loss = mmd_loss(X, Y)
        elif adapt_loss == 'coral':
            transfer_loss = CORAL(X, Y)
        elif adapt_loss == 'cmmd':
            cmmd_loss = 0
            mmd_loss = mmd.mmd_rbf_noaccelerate(X, Y)
            if self.training:
                cmmd_loss = Variable(torch.Tensor([0]))
                cmmd_loss = cmmd_loss.to(DEVICE)
                cmmd_loss = mmd.cmmd(X, Y, s_label, t_label)
            transfer_loss = (1- mu) * cmmd_loss + mu * mmd_loss
        else:
            transfer_loss = 0
        return transfer_loss
