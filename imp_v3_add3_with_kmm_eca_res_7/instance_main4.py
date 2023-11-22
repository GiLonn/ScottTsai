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
import call_resnet18_multi
from torch.utils.tensorboard import SummaryWriter

import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser(description="ZYM_feature_4")
parser.add_argument('--root_path', type=str, default="/data/sihan.zhu/transfer learning/deep/dataset/RSTL/")
parser.add_argument('--source_dir', type=str, default="UCM")
parser.add_argument('--target_train_dir', type=str, default="UCM")
parser.add_argument('--target_test_dir', type=str, default="RSSCN7")
# parser.add_argument("--preprocess", type=str2bool, default=False, help='run prepare_data or not')

#parser.add_argument("--pa_train_source",type=str, default="None",help='path to training source data')
#parser.add_argument("--pa_train_source_label",type=str, default="None",help='path to training source data label')

#parser.add_argument("--pa_train_target",type=str, default="None",help='path to training target data')
#parser.add_argument("--pa_train_target_label",type=str, default="None",help='path to training target data label')

#parser.add_argument("--pa_target_test",type=str, default="None",help='path to testing target data')

parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate")
#parser.add_argument("--lambda_", type=int, default=1, help="lambda of MMD or Coral")
parser.add_argument("--epoch", type=int, default=150, help="Number of training epochs")
parser.add_argument('--mu', type=float, default=0)
parser.add_argument('--source_class', type=str, default="None")
parser.add_argument('--test_class', type=str, default="None")

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


def test(model, target_test_loader, test_flag, mu):
    model.eval() # 讓模型變成測試模式，主要針對Dropout與Batch Normalization在train與eval的不同設置模式
    test_loss = utils.AverageMeter() # Computes and stores the average and current value
    correct_total = 0.
    count_stack = 0
    #import pdb;pdb.set_trace()
    test_flag = 1
    criterion = torch.nn.CrossEntropyLoss() # 定義一個標準準則，用來計算loss (讓輸出經過softmax，再進入Cross Entropy)
    len_target_dataset = len(target_test_loader.dataset) #所有test資料集的總數
    with torch.no_grad(): # 在做evaluation時，關閉計算導數來增加運行速度
        for data, target in target_test_loader: # data為test資料，target為test label
            data, target = data.to(DEVICE), target.to(DEVICE)
            #import pdb; pdb.set_trace()
            s_output, test_mmd_loss = model(data, data, target, mu) # 將data放入模型得到預測的輸出
            loss = criterion(s_output, target) + test_mmd_loss #計算loss
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
    test_total = 100*correct_total.type(torch.float32)/len_target_dataset
    print('{} --> {}: test max correct: {}, test accuracy: {:.3f} % \n'.format(source_name, target_name, correct_total, test_total))
    #print("test_loss = {}".format(test_loss.avg))

    logtest.append([test_total,tn, fp, fn, tp])
    np_test = np.array(logtest, dtype=float)
    # delimiter : 分隔浮號 ； %.6f 浮點型保留六位小數
    #np.savetxt(opt.save_test_name, np_log, delimiter=',', fmt='%.6f')
    test_flag = 0
    return np_test
    


def train(source_loader, target_train_loader, test_flag,  target_test_loader,KMM_weight, model, mu, CFG, save_test_path, save_log_path, optimizer):
    len_source_loader = len(source_loader) # 訓練資料來源的batch數量
    len_target_loader = len(target_train_loader) # 訓練資料目標batch的數量
    for e in range(opt.epoch):
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        LEARNING_RATE = opt.lr / math.pow((1 + 10 * (e - 1) / opt.epoch), 0.75) ##
        #print(LEARNING_RATE)
        
        
        
        
        scheduler.step()
        #scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.85)
        #print(optimizer.state_dict()['param_groups'][0]['lr'])
        #print(scheduler.get_last_lr())
        #print("model.base_network.conv1.weight")
        #print(model.base_network.conv1.weight)
        #print("model.base_network.layer4[0].conv1.weight")
        # print(model.base_network.layer4[0].conv1.weight)


        # 讓pytorch知道切換到training mode
        model.train()

        # dataloader是可迭代對象，需先使用iter()訪問，返回一個迭代器後，再使用next()訪問
        iter_source, iter_target = iter(source_loader), iter(target_train_loader)

        n_batch = min(len_source_loader, len_target_loader) # 取train資料source與target的最小值 (batch_size設置的參數不是在這傳入)
        # n_batch = len_source_loader
        criterion = torch.nn.CrossEntropyLoss() # 使用Loss為CrossEntropy (pytorch crossentropy會自動先經過softmax function)
        #scheduler.step()
        #print(scheduler.get_last_lr())

        for i in range(n_batch):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            data_source, label_source = data_source.to(DEVICE), label_source.to(DEVICE)
            data_target = data_target.to(DEVICE)
            #print(data_target.size())
            #print(label_source)

            label_source = torch.squeeze(label_source)
            #print(label_source)

            optimizer.zero_grad() # set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
            label_source_pred, transfer_loss = model(data_source, data_target, label_source, mu) # 返回 model.forward 的 source_clf 與 transfer_loss
            clf_loss = criterion(label_source_pred, label_source)
            gamma = 2 / (1 + math.exp(-10 * (e) / opt.epoch)) - 1 
            loss = KMM_weight * clf_loss + gamma * transfer_loss # classification loss + lambda * transfer loss
            loss.backward() # computes dloss/dx for every parameter x which has requires_grad=True
            optimizer.step() #updates the value of x using the gradient x.grad
             
            train_loss_clf.update(clf_loss.item()) # 更新值到紀錄中
            train_loss_transfer.update(transfer_loss.item()) # 更新值到紀錄中
            train_loss_total.update(loss.item()) # 更新值到紀錄中





            if i % CFG['log_interval'] == 0:
                print('Train Epoch: [{}/{} ({:02d}%)], cls_Loss: {:.6f}, transfer_loss: {:.6f}, total_Loss: {:.6f}'.format(
                    e + 1,
                    opt.epoch,
                    int(100. * i / n_batch), train_loss_clf.avg, train_loss_transfer.avg * 1, train_loss_total.avg))



        #writer.add_scalar('Loss/train', np.random.random(), e)
        #writer.add_scalar('Accuracy/train', np.random.random(), e)
        #writer.add_scalar('Accuracy/test', np.random.random(), e)
        #writer.close()
        logtrain.append([train_loss_clf.avg, train_loss_transfer.avg * gamma,])
        np_log = np.array(logtrain, dtype=float)
        save_log_add = os.path.join(save_log_path, str(opt.save_train_loss_name))
        # delimiter : 分隔浮號 ； %.6f 浮點型保留六位小數
        np.savetxt(save_log_add, np_log, delimiter=',', fmt='%.12f')
        print(mu)
        #return np_log


        #Test
        np_test = test(model, target_test_loader, test_flag, opt.mu)
        save_test_add = os.path.join(save_test_path, str (opt.save_test_name))
        np.savetxt(save_test_add, np_test, delimiter=',', fmt='%.6f')
    



def load_data(root_dir):
    folder_test = root_dir
    target_test_loader,test_index = data_loader.load_data(folder_test, opt.batch_size, False, CFG['kwargs'])
    return target_test_loader


if __name__ == '__main__':
    torch.manual_seed(0) # 為CPU设置隨機種子讓參數是從某一隨機種子初始化
    test_flag = 0
    source_name = opt.source_class
    target_name = opt.test_class
    print('Src: %s, Tar: %s' % (source_name, target_name))
    

    kwargs = {'num_workers': 1, 'pin_memory': True} if opt.gpu_id == 0 or 1 else {}


    source_loader = data_loader.load_training(opt.root_path, opt.source_dir, opt.batch_size, kwargs)
    target_train_loader = data_loader.load_training(opt.root_path, opt.target_train_dir, opt.batch_size, kwargs)
    target_test_loader = data_loader.load_testing(opt.root_path, opt.target_test_dir, opt.batch_size, kwargs)
    kmm_source_positive_path = os.path.join(opt.root_path, opt.source_dir, 'positive/')
    kmm_target_positive_path = os.path.join(opt.root_path, opt.target_train_dir, 'positive/')
    #print("kmm_source = {}".format(kmm_source_positive_path))
    #print("kmm_target = {}".format(kmm_target_positive_path))

    



    len_source_dataset = len(source_loader.dataset)
    len_target_dataset = len(target_test_loader.dataset)
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)

    
    save_parameter_path = './record/' + opt.source_class + '/' + 'to' + '/' + opt.test_class + '/' +str(opt.lr) + '/' 'mu=' + str(opt.mu)
    save_test_path = './accuracy/' + opt.source_class+ '/' + 'to' + '/' + opt.test_class + '/' +str(opt.lr) + '/' 'mu=' + str(opt.mu)
    save_log_path = './log/' + opt.source_class + '/' + 'to' + '/' + opt.test_class + '/' +str(opt.lr) + '/' 'mu=' + str(opt.mu)

    if not os.path.exists(save_parameter_path):
        os.makedirs(save_parameter_path)
    if not os.path.exists(save_test_path):
        os.makedirs(save_test_path)
    if not os.path.exists(save_log_path):
        os.makedirs(save_log_path)
    
    model = models.Transfer_Net(CFG['n_class'])
    model = model.cuda()


    KMM_weight = KMM_Lin.compute_kmm(kmm_source_positive_path, kmm_target_positive_path)
    KMM_weight = torch.from_numpy(KMM_weight).float().to(DEVICE)
    print(KMM_weight)


    #writer = SummaryWriter()


    optimizer = torch.optim.Adam([
            {'params': model.base_network.conv1.parameters(), 'lr' : opt.lr },
            {'params': model.base_network.bn1.parameters(), 'lr' : opt.lr },
            {'params': model.base_network.relu.parameters(), 'lr' : opt.lr },
            {'params': model.base_network.layer1.parameters(), 'lr' : opt.lr },
            {'params': model.base_network.layer2.parameters(), 'lr' : opt.lr },
            {'params': model.base_network.layer3.parameters(), 'lr' : opt.lr},
            

            {'params': model.base_network.convm3_layer.parameters(), 'lr' : 10 * opt.lr },
            {'params': model.base_network.maxpool1.parameters(), 'lr' : opt.lr },
            {'params': model.base_network.linear_test.parameters(), 'lr' : opt.lr },
        #{'params': model.base_network.layer2.parameters(), 'lr' : opt.lr},
        
            
        #{'params': model.base_network.layer4.parameters(), 'lr' : opt.lr},
            

            {'params': model.base_network.avgpool.parameters(), 'lr' :  opt.lr},
            {'params': model.classifier_layer.parameters(), 'lr' :   10 *  opt.lr},
            {'params': model.bottleneck_layer.parameters(), 'lr' :   opt.lr},
        ], lr=opt.lr , betas=CFG['betas'], weight_decay=CFG['l2_decay'])

    #scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=20, gamma=0.16)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.85)
    #for params in model.parameters():
        #print(params.data)
    
    '''
    optimizer = torch.optim.SGD([
        {'params': model.base_network.convm1_layer.parameters(), 'lr' : 500 * opt.lr},
        {'params': model.classifier_layer.parameters(), 'lr' : 100 * opt.lr},
        {'params': model.bottleneck_layer.parameters(), 'lr' : 100 * opt.lr},
    ], lr=opt.lr, momentum=0.9, weight_decay=CFG['l2_decay'])
    '''
    #scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.16)

    np_log = train(source_loader, target_train_loader, test_flag, 
          target_test_loader,KMM_weight, model, opt.mu, CFG, save_test_path, save_log_path, optimizer)
    #np.savetxt(opt.save_train_loss_name, np_log, delimiter=',', fmt='%.12f')

    # Test

    np_test = test(model, target_test_loader, test_flag, opt.mu)
    #np.savetxt(opt.save_test_name, np_log, delimiter=',', fmt='%.6f')
 
    save_parameter_add = os.path.join(save_parameter_path + str(opt.save_parameter_path_name))
    torch.save(model.state_dict(), save_parameter_add)