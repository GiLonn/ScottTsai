import torch
import os
import math
import data_loader
import models
from config import CFG
import utils
import numpy as np
import KMM_Lin
from sklearn.metrics import confusion_matrix
import torch.utils.data as Data

import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser(description="ZYM_feature_4")
# parser.add_argument("--preprocess", type=str2bool, default=False, help='run prepare_data or not')

parser.add_argument("--pa_train_source",type=str, default="None",help='path to training source data')
parser.add_argument("--pa_train_source_label",type=str, default="None",help='path to training source data label')

parser.add_argument("--pa_train_target",type=str, default="None",help='path to training target data')
parser.add_argument("--pa_train_target_label",type=str, default="None",help='path to training target data label')

parser.add_argument("--pa_target_test",type=str, default="None",help='path to testing target data')

parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate")
parser.add_argument("--lambda_", type=int, default=1, help="lambda of MMD or Coral")
parser.add_argument("--epoch", type=int, default=150, help="Number of training epochs")
# parser.add_argument("--kmm_source_path", type=str, default="None", help="kmm_source_path")
# parser.add_argument("--kmm_target_path", type=str, default="None", help="kmm_target_path")
parser.add_argument("--save_parameter_path_name", type=str, default="None", help='path to save models and log files')
parser.add_argument("--save_test_name", type=str, default='test_log1.csv', help='path to save models and log files')
parser.add_argument("--save_train_loss_name", type=str, default='train_log.csv', help='path to save models and log files')
# parser.add_argument("--use_gpu", type=str2bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default='cuda:1', help='GPU id')

opt = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.cuda.set_device(opt.gpu_id)
logtrain = [] #創建一個空的list
logtest = [] #創建一個空的list

def test(model, target_test_loader):
    model.eval() # 讓模型變成測試模式，主要針對Dropout與Batch Normalization在train與eval的不同設置模式
    test_loss = utils.AverageMeter() # Computes and stores the average and current value
    correct_total = 0.
    count_stack = 0
    criterion = torch.nn.CrossEntropyLoss() # 定義一個標準準則，用來計算loss (讓輸出經過softmax，再進入Cross Entropy)
    len_target_dataset = len(target_test_loader.dataset) #所有test資料集的總數
    with torch.no_grad(): # 在做evaluation時，關閉計算導數來增加運行速度
        for data, target in target_test_loader: # data為test資料，target為test label
            data, target = data.to(DEVICE), target.to(DEVICE)
            s_output = model.predict(data) # 將data放入模型得到預測的輸出
            loss = criterion(s_output, target) #計算loss
            test_loss.update(loss.item()) # 更新值到紀錄中

            # torch.max(a,1) 返回每一列中最大值的那個元素，且返回其索引
            # troch.max()[1] 只返回最大值的每個索引
            pred = torch.max(s_output, 1)[1]
            correct_total += torch.sum(pred == target)

            pred_matrix = pred.data
            target_matrix = target.data
            # print("pred_matrix、target_matrix")
            # print(pred_matrix,target_matrix)
            if count_stack == 0:
                pred_matrix_total = pred_matrix
                target_matrix_total = target_matrix
                count_stack =1
            elif count_stack == 1:
                pred_matrix_total = torch.cat((pred_matrix_total,pred_matrix))
                target_matrix_total = torch.cat((target_matrix_total,target_matrix))

            # print("pred_matrix_total、target_matrix_total")
            # print(pred_matrix_total, target_matrix_total)
        target_matrix_total = target_matrix_total.cpu().numpy()
        pred_matrix_total = pred_matrix_total.cpu().numpy()
        tn, fp, fn, tp = confusion_matrix(target_matrix_total, pred_matrix_total,labels=[0,1]).ravel()
        # print(tn, fp, fn, tp)
    # correct = correct.cpu().numpy()
    # {:.2f}保留小數點後兩位
    # print(correct)
    # print(correct/300.)
    # print('{:.3f}'.format(correct/len_target_dataset))
    test_total = 100*correct_total.type(torch.float32)/len_target_dataset
    print('{} --> {}: test max correct: {}, test accuracy: {:.3f} % \n'.format(source_name, target_name, correct_total, test_total))

    logtest.append([test_total,tn, fp, fn, tp])
    np_log = np.array(logtest, dtype=float)
    # delimiter : 分隔浮號 ； %.6f 浮點型保留六位小數
    np.savetxt(opt.save_test_name, np_log, delimiter=',', fmt='%.6f')


def train(source_loader, target_train_loader, target_test_loader, model, optimizer, CFG):
    len_source_loader = len(source_loader) # 訓練資料來源的batch數量
    len_target_loader = len(target_train_loader) # 訓練資料目標batch的數量
    for e in range(opt.epoch):
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()

        # print("model.base_network.conv1.weight")
        # print(model.base_network.conv1.weight)
        # print("model.base_network.layer4[0].conv1.weight")
        # print(model.base_network.layer4[0].conv1.weight)


        # 讓pytorch知道切換到training mode
        model.train()

        # dataloader是可迭代對象，需先使用iter()訪問，返回一個迭代器後，再使用next()訪問
        iter_source, iter_target = iter(source_loader), iter(target_train_loader)

        n_batch = min(len_source_loader, len_target_loader) # 取train資料source與target的最小值 (batch_size設置的參數不是在這傳入)
        # n_batch = len_source_loader
        criterion = torch.nn.CrossEntropyLoss() # 使用Loss為CrossEntropy (pytorch crossentropy會自動先經過softmax function)
        scheduler.step()
        print(scheduler.get_lr())

        for i in range(n_batch):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            data_source, label_source = data_source.to(DEVICE), label_source.to(DEVICE)
            data_target = data_target.to(DEVICE)

            label_source = torch.squeeze(label_source)

            optimizer.zero_grad() # set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
            label_source_pred, transfer_loss = model(data_source, data_target) # 返回 model.forward 的 source_clf 與 transfer_loss
            clf_loss = criterion(label_source_pred, label_source) # source 的 classification loss
            #loss = clf_loss * KMM_weight + opt.lambda_ * transfer_loss  # classification loss + lambda * transfer loss
            loss = clf_loss * KMM_weight + 1 * transfer_loss # classification loss + lambda * transfer loss

            loss.backward() # computes dloss/dx for every parameter x which has requires_grad=True
            optimizer.step() # updates the value of x using the gradient x.grad
            train_loss_clf.update(clf_loss.item()) # 更新值到紀錄中
            train_loss_transfer.update(transfer_loss.item()) # 更新值到紀錄中
            train_loss_total.update(loss.item()) # 更新值到紀錄中

            if i % CFG['log_interval'] == 0:
                print('Train Epoch: [{}/{} ({:02d}%)], cls_Loss: {:.6f}, transfer_loss: {:.6f}, total_Loss: {:.6f}'.format(
                    e + 1,
                    opt.epoch,
                    int(100. * i / n_batch), train_loss_clf.avg, train_loss_transfer.avg*opt.lambda_, train_loss_total.avg))
        logtrain.append([train_loss_clf.avg, train_loss_transfer.avg*opt.lambda_,])
        np_log = np.array(logtrain, dtype=float)
        # delimiter : 分隔浮號 ； %.6f 浮點型保留六位小數
        np.savetxt(opt.save_train_loss_name, np_log, delimiter=',', fmt='%.12f')

        # Test
        test(model, target_test_loader)
    

def load_data(root_dir):
    folder_test = root_dir
    target_test_loader,test_index = data_loader.load_data(folder_test, opt.batch_size, False, CFG['kwargs'])
    return target_test_loader

if __name__ == '__main__':
    torch.manual_seed(0) # 為CPU设置隨機種子讓參數是從某一隨機種子初始化

    source_name = "/source/train"
    target_name = "/target/train"
    # test_name = "/target/test"

    print('Src: %s, Tar: %s' % (source_name, target_name))

    # root_path = E:\AmigoChou\Q2\Training\O2M_amigo\datasets\feature_map\apple\all_in_one\5

    path_train_source = opt.pa_train_source
    # path_train_source='E:\\AmigoChou\\Q2\\Training\\O2M_LXZ vs ZYM\\datasets_ZYM\\4\\apple_to_\\to_pear\\source\\train\\source_train_feature.npy'
    sample_source_train = torch.from_numpy(np.load(path_train_source))

    path_train_source_label = opt.pa_train_source_label
    # path_train_source_label='E:\\AmigoChou\\Q2\\Training\\O2M_LXZ vs ZYM\\datasets_ZYM\\4\\apple_to_\\to_pear\\source\\train\\source_train_feature_label.npy'
    sample_source_train_label = torch.from_numpy(np.load(path_train_source_label))

    path_train_target = opt.pa_train_target
    # path_train_target='E:\\AmigoChou\\Q2\\Training\\O2M_LXZ vs ZYM\\datasets_ZYM\\4\\apple_to_\\to_pear\\target\\train\\target_train_feature.npy'
    sample_target_train = torch.from_numpy(np.load(path_train_target))

    path_train_target_label = opt.pa_train_target_label
    # path_train_target_label='E:\\AmigoChou\\Q2\\Training\\O2M_LXZ vs ZYM\\datasets_ZYM\\4\\apple_to_\\to_pear\\target\\train\\target_train_feature_label.npy'
    sample_target_train_label = torch.from_numpy(np.load(path_train_target_label))

    source_dataset = Data.TensorDataset(sample_source_train, sample_source_train_label)
    target_dataset = Data.TensorDataset(sample_target_train, sample_target_train_label)

    source_loader = Data.DataLoader(
    dataset=source_dataset,      # torch TensorDataset format
    batch_size=opt.batch_size,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,   # 多线程来读数据
    drop_last = True, # 未到batch_size數量之樣本丟棄
	)

    target_loader = Data.DataLoader(
    dataset=target_dataset,      # torch TensorDataset format
    batch_size=opt.batch_size,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,   # 多线程来读数据
    drop_last = True, # 未到batch_size數量之樣本丟棄
	)

    # target_test_loader = load_data('E:\\AmigoChou\\Q2\\Training\\O2M_LXZ vs ZYM\\datasets\\O2M_datasets_backup\\apple_to_\\to_pear\\sn_200_sp_200\\target\\test')
    target_test_loader = load_data(opt.pa_target_test)

    # model = models.Transfer_Net(
    #     CFG['n_class'], transfer_loss='mmd', base_net='resnet50').to(DEVICE)
    model = models.Transfer_Net(CFG['n_class'], transfer_loss='coral', base_net='resnet18').to(DEVICE)


    # 如要關閉網路參數更新，將param.requires_grad = False 即可
    # for param in model.base_network.conv1.parameters():
    #     param.requires_grad = False
    
    # for param in model.base_network.bn1.parameters():
    #     param.requires_grad = False
    
    # for param in model.base_network.relu.parameters():
    #     param.requires_grad = False
    
    # for param in model.base_network.maxpool.parameters():
    #     param.requires_grad = False
    
    # for param in model.base_network.layer1[0].parameters():
    #     param.requires_grad = False

    # for param in model.base_network.layer1[1].parameters():
    #     param.requires_grad = False

    # for param in model.base_network.layer2[0].parameters():
    #     param.requires_grad = False

    # for param in model.base_network.layer2[1].parameters():
    #     param.requires_grad = False

    # for param in model.base_network.layer3[0].parameters():
    #     param.requires_grad = False

    # for param in model.base_network.layer3[1].parameters():
    #     param.requires_grad = False

    # for param in model.base_network.layer4[0].parameters():
    #     param.requires_grad = False

    # for param in model.base_network.layer4[1].parameters():
    #     param.requires_grad = False

    # for param in model.base_network.avgpool.parameters():
    #     param.requires_grad = False

    # Compute the KMM weight by Lin
    # source_positive_dir_path,target_positive_dir_path


    # KMM_weight = KMM_Lin.compute_kmm('E:\\AmigoChou\\Q2\\Training\\O2M_LXZ vs ZYM\\datasets_ZYM\\4\\apple_to_\\to_pear\\source\\train\\source_train_feature.npy'
    #                                  ,'E:\\AmigoChou\\Q2\\Training\\O2M_LXZ vs ZYM\\datasets_ZYM\\4\\apple_to_\\to_pear\\target\\train\\target_train_feature.npy'
    #                                  ,'E:\\AmigoChou\\Q2\\Training\\O2M_LXZ vs ZYM\\datasets_ZYM\\4\\apple_to_\\to_pear\\source\\train\\source_train_feature_label.npy')
    KMM_weight = KMM_Lin.compute_kmm(opt.pa_train_source, opt.pa_train_target, opt.pa_train_source_label)
    KMM_weight = torch.from_numpy(KMM_weight).float().to(DEVICE)
    print(KMM_weight)

    # optimizer = torch.optim.SGD([
    #     {'params': model.base_network.parameters()},
    #     {'params': model.bottleneck_layer.parameters(), 'lr': 10 * CFG['lr']},
    #     {'params': model.classifier_layer.parameters(), 'lr': 10 * CFG['lr']},
    # ], lr=CFG['lr'], momentum=CFG['momentum'], weight_decay=CFG['l2_decay'])

    # optimizer = torch.optim.Adam(model.parameters(),lr=CFG['lr'], betas=CFG['betas'], weight_decay=CFG['l2_decay'])
    optimizer = torch.optim.Adam([
        {'params': model.base_network.parameters()},
        {'params': model.bottleneck_layer.parameters(), 'lr': 10 * opt.lr},
        {'params': model.classifier_layer.parameters(), 'lr': 10 *  opt.lr},
    ], lr=opt.lr, betas=CFG['betas'], weight_decay=CFG['l2_decay'])

    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.16)

    train(source_loader, target_loader,
          target_test_loader, model, optimizer, CFG)

    # Test
    test(model, target_test_loader)

    torch.save(model.state_dict(), opt.save_parameter_path_name)
