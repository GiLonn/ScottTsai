import torch.nn as nn
import torchvision
from Coral import CORAL
# import mmd3 as mmd
import mmd2_correct as mmd2
# import mmd
import backbone


# ResNet-18在最初時沒有使用bottleneck
class Transfer_Net(nn.Module):
    def __init__(self, num_class, base_net='resnet18', transfer_loss='mmd', use_bottleneck=True, bottleneck_width=256, width=1024):
        super(Transfer_Net, self).__init__()
        self.base_network = backbone.network_dict[base_net]() # 從backbone已經定義好的resnet-18到fully-connected之前
        self.use_bottleneck = use_bottleneck # 是否用bottleneck (這邊指的不是resnet-18的bottleneck，而是論文DDC的fully-connected)
        self.transfer_loss = transfer_loss
        
        bottleneck_list = [nn.Linear(self.base_network.output_num(), bottleneck_width), nn.BatchNorm1d(bottleneck_width), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        
        classifier_layer_list = [nn.Linear(self.base_network.output_num(), width), nn.ReLU(), nn.Dropout(0.5),nn.Linear(width, num_class)]
        self.classifier_layer = nn.Sequential(*classifier_layer_list)
        for i in range(2): # classifier[0]、classifier[3] = nn.Linear
            self.classifier_layer[i * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[i * 3].bias.data.fill_(0.0)

    def forward(self, source, target):
        source = self.base_network(source)
        target = self.base_network(target)
        
        source_clf = self.classifier_layer(source)
        
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)
        transfer_loss = self.adapt_loss(source, target, self.transfer_loss)
        
        return source_clf, transfer_loss

    def predict(self, x):
        features = self.base_network(x)
        clf = self.classifier_layer(features)
        return clf

    def adapt_loss(self, X, Y, adapt_loss):
        """Compute adaptation loss, currently we support mmd and coral

        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix
            adapt_loss {string} -- loss type, 'mmd' or 'coral'. You can add your own loss

        Returns:
            [tensor] -- adaptation loss tensor
        """
        if adapt_loss == 'mmd':
            # mmd_loss = mmd.MMD_loss()
            # loss = mmd_loss(X, Y)

            loss = mmd2.MMD_loss(X, Y)

        elif adapt_loss == 'coral':
            loss = CORAL(X, Y)
        else:
            loss = 0
        return loss
