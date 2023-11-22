import models
import os
import argparse
import torch
import numpy as np
from torchinfo import summary
import torch.utils.data as Data
parser = argparse.ArgumentParser(description="ZYM_feature_4")
parser.add_argument('--train_root_path', type=str, default="/data/sihan.zhu/transfer learning/deep/dataset/RSTL/")
parser.add_argument('--source_dir', type=str, default="UCM")
parser.add_argument("--gpu_id", type=str, default='cuda:1', help='GPU id')
opt = parser.parse_args()

path_source_train = os.path.join(opt.train_root_path, opt.source_dir, 'source_train_feature.npy')
path_source_train_label = os.path.join(opt.train_root_path, opt.source_dir, 'source_train_feature_label.npy')
source_train = torch.from_numpy(np.load(path_source_train))
source_train_label = torch.from_numpy(np.load(path_source_train_label))	

source_dataset = Data.TensorDataset(source_train, source_train_label)




#target_dataset = Data.TensorDataset(target_train, target_train_label)
if __name__ == '__main__':
	source_loader = Data.DataLoader(
	    dataset=source_dataset,      # torch TensorDataset format
	    batch_size=8,      # mini batch size
	    shuffle=True,               # 要不要打乱数据 (打乱比较好)
	    num_workers=2,   # 多线程来读数据
	    drop_last = True,
	    persistent_workers=False, # 未到batch_size數量之樣本丟棄
	    )

	len_source_loader = len(source_loader)
	n_batch = len_source_loader
	iter_source = iter(source_loader)
	for i in range(n_batch):
		data_source, label_source = iter_source.next()

		label_source = torch.squeeze(label_source)
		#print(label_source.size())

	#print(models.Transfer_Net(num_class = 2))

	model = models.Transfer_Net(num_class = 2)
	model.cuda()
	summary(model, [(8, 128, 28, 28), (8, 128, 28, 28), 0.5])