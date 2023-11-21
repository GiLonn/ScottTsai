import os
import torch
import torch.nn as nn
import numpy as np
import backbone_multi as bk
from PIL import Image
import cv2
import shutil
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models
gpu_id = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print("a")

model_root_path = 'E:/AmigoChou/Q2/Training/O2M_amigo_feature map_terminal/5_coral/'
model_layer_path = 'result_lambda_30000/'
class_path = 'carambola_to/to_muskmelon/'
model_name = 'model_parameter_30000_log1.pkl'

kwargs = {'num_workers': 2, 'pin_memory': False, 'persistent_workers' : True} if gpu_id == 0 or 1 else {}
test_root_path = "D:/ScottTsai/O2M_datasets_backup/"
target_test_layer = "carambola_to_/to_muskmelon/"
target_test_layer2 = 'sn_200_sp_200/target/test/positive/'
target = 'Muskmelon 00285.png'

target_path = target[:-4] + '/'
print (target_path)
save_root_path = 'E:/ScottTsai/grad_cam++/'
save_layer_path = model_layer_path + '/' + 'layer4/'
save_fruit_path = target_test_layer + "positive/"

save_path = os.path.join(save_root_path, save_layer_path, save_fruit_path, target_path) 

model_path = os.path.join(model_root_path, model_layer_path, class_path, model_name)
target_test_path = os.path.join(test_root_path, target_test_layer, target_test_layer2, target)

img = Image.open(target_test_path)
img2 = cv2.imread(target_test_path, 1)[:, :, ::-1]
img2 = cv2.resize(img2, (224, 224))
img2 = np.float32(img2) / 255


transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
                                  
img = transform(img)
img = img.unsqueeze(0)

class ResNet18Fc(nn.Module):
    def __init__(self):
        super(ResNet18Fc, self).__init__() # 繼承父類，MRO(Method Resolution Order)
        model_resnet18 = models.resnet18(pretrained=False) # If pretrained=True, returns a model pre-trained on ImageNet

        self.conv1 = model_resnet18.conv1
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool

        self.layer1 = model_resnet18.layer1

        self.layer2 = model_resnet18.layer2

        self.layer3 = model_resnet18.layer3

        self.layer4 = model_resnet18.layer4

        self.avgpool = model_resnet18.avgpool
        self.__in_features = model_resnet18.fc.in_features # 最後一層fully connected 的 input_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1) 
        return x

    def output_num(self):
        return self.__in_features

class Net(nn.Module):
    def __init__(self, num_class = 2, bottleneck_width = 256, width = 1024):
        super(Net, self).__init__()
        
        self.base_network = ResNet18Fc()
        bottleneck_list = [nn.Linear(self.base_network.output_num(), bottleneck_width), nn.BatchNorm1d(bottleneck_width), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)
        classifier_layer_list = [nn.Linear(self.base_network.output_num(), width), nn.ReLU(), nn.Dropout(0.5),nn.Linear(width, num_class)]
        self.classifier_layer = nn.Sequential(*classifier_layer_list)

    def forward(self, x):
   	    features = self.base_network(x)
   	#features = self.bottleneck_layer(features)
   	    clf = self.classifier_layer(features)
   	    return clf


model = Net()
target_layers = [model.base_network.layer4]

model.load_state_dict(torch.load(model_path))
#cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
cam =  GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=False)
#print(cam)
targets = [ClassifierOutputTarget(1)]
grayscale_cam = cam(input_tensor=img, targets=targets, aug_smooth=True, eigen_smooth=False)
grayscale_cam = grayscale_cam[0, :]
#print(model)
model.eval()
output = model(img)

pred = torch.max(output, 1)[1]
print(output)
visualization = show_cam_on_image(img2, grayscale_cam, use_rgb=True)
plt.imshow(visualization)
if not os.path.exists(save_path):
	os.makedirs(save_path)
plt.savefig(save_path + target + '_heat.png')

#img2.save(save_path + target + '.png')
#cv2.imshow('target',visualization)
plt.show()



while True:
	a = input("input: s for save, q for quit")

	if a == 's':
		shutil.copyfile(target_test_path, save_path + target + '.png')
		break

	if a =='q':
		os.remove(save_path + target + '_heat.png')
		os.rmdir(save_path)
		break



