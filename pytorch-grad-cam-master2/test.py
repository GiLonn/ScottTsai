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
gpu_id = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
def load_testing(root_path, dir, kwargs, batch_size = 1):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader
'''
model_root_path = 'E:/ScottTsai/res_multi/auto_hyperparameter/implicit/'
model_layer_path = 'imp_v3_add3_with_kmm_dr_conv3_att2'
class_path = 'record/apple/to/muskmelon/'
model_name = '0.0001epoch150_1.pkl'

kwargs = {'num_workers': 2, 'pin_memory': False, 'persistent_workers' : True} if gpu_id == 0 or 1 else {}
test_root_path = "D:/ScottTsai/O2M_datasets_backup/"
target_test_layer = "carambola_to_/to_muskmelon/"
target_test_layer2 = 'sn_200_sp_200/target/test/positive/'
target = 'Muskmelon 00815.png'

target_path = target[:-4] + '/'
print (target_path)
save_root_path = 'E:/ScottTsai/grad_cam++/'
save_layer_path = model_layer_path + '/' + 'linear_test/'
save_fruit_path = target_test_layer + "positive/"

save_path = os.path.join(save_root_path, save_layer_path, save_fruit_path, target_path) 

model_path = os.path.join(model_root_path, model_layer_path, class_path, model_name)
target_test_path = os.path.join(test_root_path, target_test_layer, target_test_layer2, target)

img = Image.open(target_test_path)
img2 = cv2.imread(target_test_path, 1)[:, :, ::-1]
img2 = cv2.resize(img2, (224, 224))
img2 = np.float32(img2) / 255
#img2 = Image.open(target_test_path)
#img2 = img2.resize((224, 224))
#img2 = np.asarray(img2)
#img2 = img2 / 255

#img2 = np.array(img2.getdata())
#print(img2)

transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
                                  
img = transform(img)
#transform2 = transforms.Compose([transfroms.Resize([224, 224])])
#np_img = img.detach().numpy()
#img2 = transform2(img2)
img = img.unsqueeze(0)

#print(img)


#target_test_loader = load_testing(test_root_path, target_test_dir, kwargs)



class Net(nn.Module):
	def __init__(self, num_class = 2, bottleneck_width = 128, width = 512):
		super(Net, self).__init__()
		
		self.base_network = bk.resnet18_multi()
		bottle_list = [nn.Linear(256, width), nn.Linear(width, bottleneck_width)]
		classifier_list = [nn.Dropout(0.25), nn.Linear(bottleneck_width, num_class)]

		self.bottle_layer = nn.Sequential(*bottle_list)
		self.classifier_layer = nn.Sequential(*classifier_list)

	def forward(self, x):
		features = self.base_network(x)
		features = self.bottle_layer(features)
		clf = self.classifier_layer(features)
		return clf

		




model = Net()
target_layers = [model.base_network.convm2_layer]

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







