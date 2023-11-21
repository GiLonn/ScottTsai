# -*- coding: UTF-8 -*-
from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader
import resnet as models
import numpy as np
import time

import KMM_Lin
from sklearn.metrics import confusion_matrix

import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser(description="M2O_2_to_1_train")
# parser.add_argument("--preprocess", type=str2bool, default=False, help='run prepare_data or not')

parser.add_argument("--data_path",type=str, default="None",help='path to training data')
parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate")
# parser.add_argument("--lambda_", type=float, default=None, help="lambda of MMD or Coral")
parser.add_argument("--iteration", type=int, default=1000, help="Number of training epochs")
# parser.add_argument("--kmm_source1_path", type=str, default="None", help="kmm_source1_path")
# parser.add_argument("--kmm_source2_path", type=str, default="None", help="kmm_source2_path")
# parser.add_argument("--kmm_target_path", type=str, default="None", help="kmm_target_path")
parser.add_argument("--save_model_path_name", type=str, default="None", help='path to save models and log files')
parser.add_argument("--save_test_name", type=str, default='test_log1.csv', help='path to save models and log files')
parser.add_argument("--save_train_loss_name1", type=str, default='train_log1.csv', help='path to save models and log files')
parser.add_argument("--save_train_loss_name2", type=str, default='train_log2.csv', help='path to save models and log files')
# parser.add_argument("--use_gpu", type=str2bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default='cuda:0', help='GPU id')

opt = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.set_device(opt.gpu_id)
logtrain1 = [] #創建一個空的list
logtrain2 = [] #創建一個空的list
logtest = [] #創建一個空的list

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
# batch_size = 32
# iteration = 10000
# lr = 0.01
# momentum = 0.9
cuda = True
seed = 8
log_interval = 1
l2_decay = 1e-4
betas = [0.9, 0.999]
iteration = opt.iteration

root_path = opt.data_path
source1_name = "/source1/train"
source2_name = '/source2/train'
target_train_name = "/target/train"
target_test_name = "/target/test"

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

source1_loader = data_loader.load_training(root_path, source1_name, opt.batch_size, kwargs)
source2_loader = data_loader.load_training(root_path, source2_name, opt.batch_size, kwargs)
target_train_loader = data_loader.load_training(root_path, target_train_name, opt.batch_size, kwargs)
target_test_loader = data_loader.load_testing(root_path, target_test_name, opt.batch_size, kwargs)

def train(model):
    source1_iter = iter(source1_loader)
    source2_iter = iter(source2_loader)
    target_iter = iter(target_train_loader)
    correct = 0

    optimizer = torch.optim.Adam([
            {'params': model.sharedNet.parameters()},
            {'params': model.cls_fc_son1.parameters(), 'lr': opt.lr},
            {'params': model.cls_fc_son2.parameters(), 'lr': opt.lr},
            {'params': model.sonnet1.parameters(), 'lr': opt.lr},
            {'params': model.sonnet2.parameters(), 'lr': opt.lr},
        ], lr=opt.lr , betas=betas, weight_decay=l2_decay)

    # scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.16)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.25)


    for i in range(1, iteration + 1):
        model.train()
        # LEARNING_RATE = opt.lr
        # LEARNING_RATE = opt.lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        

        # optimizer = torch.optim.SGD([
        #     {'params': model.sharedNet.parameters()},
        #     {'params': model.cls_fc_son1.parameters(), 'lr': LEARNING_RATE},
        #     {'params': model.cls_fc_son2.parameters(), 'lr': LEARNING_RATE},
        #     {'params': model.sonnet1.parameters(), 'lr': LEARNING_RATE},
        #     {'params': model.sonnet2.parameters(), 'lr': LEARNING_RATE},
        # ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)


        torch.cuda.synchronize()
        tStart = time.time() #計時開始
        try:
            source_data, source_label = source1_iter.next()
        except Exception as err:
            source1_iter = iter(source1_loader)
            source_data, source_label = source1_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        cls_loss, mmd_loss, l1_loss = model(source_data, target_data, source_label, mark=1)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
        # gamma = opt.lambda_
        # print(gamma)
        # loss = cls_loss + gamma * (mmd_loss + l1_loss)
        loss = KMM_weight_source1 * cls_loss + gamma * mmd_loss
        loss.backward()
        optimizer.step()

        if i % (log_interval*2) == 0:
            # print("learning rate：", LEARNING_RATE)
            # print("gamma: ", gamma)
            print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tcls_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()))

        logtrain1.append([loss, cls_loss,KMM_weight_source1*cls_loss,mmd_loss,gamma*mmd_loss])
        np_log1 = np.array(logtrain1, dtype=float)
        np.savetxt(opt.save_train_loss_name1, np_log1, delimiter=',', fmt='%.12f')

        try:
            source_data, source_label = source2_iter.next()
        except Exception as err:
            source2_iter = iter(source2_loader)
            source_data, source_label = source2_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        cls_loss, mmd_loss, l1_loss = model(source_data, target_data, source_label, mark=2)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
        # gamma = opt.lambda_
        # print(gamma)
        # loss = cls_loss + gamma * (mmd_loss + l1_loss)
        loss = KMM_weight_source2 * cls_loss + gamma * mmd_loss

        loss.backward()
        optimizer.step()

        scheduler.step()
        # print(scheduler.get_lr())
        torch.cuda.synchronize()
        tEnd = time.time() #計時結束
        print (tEnd - tStart) #原型長這樣

        if i % (log_interval*2) == 0:
            print("\n"+"---------------------"+"\n")
            # print("learning rate：", LEARNING_RATE)
            # print("gamma: ", gamma)
            print('Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tcls_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                    i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()))

        if i % log_interval == 0:
            t_correct = test(model)
            if t_correct > correct:
                correct = t_correct
            print(source1_name, source2_name, "to", target_test_name, "%s max correct:" % target_test_name, correct.item(), "\n")

        logtrain2.append([loss, cls_loss,KMM_weight_source2*cls_loss,mmd_loss,gamma*mmd_loss])
        np_log2 = np.array(logtrain2, dtype=float)
        np.savetxt(opt.save_train_loss_name2, np_log2, delimiter=',', fmt='%.12f')

def test(model):
    model.eval()
    test_loss = 0
    correct_test = 0.
    correct1 = 0
    correct2 = 0
    count_stack = 0
    with torch.no_grad():
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred1, pred2 = model(data, mark = 0)

            pred1 = torch.nn.functional.softmax(pred1, dim=1)
            pred2 = torch.nn.functional.softmax(pred2, dim=1)

            pred = (pred1 + pred2) / 2
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()

            pred = pred.data.max(1)[1]
            pred_total = pred
            correct_test += pred.eq(target.data.view_as(pred)).cpu().sum()
            
            pred = pred1.data.max(1)[1]
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
            
            pred = pred2.data.max(1)[1]
            correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()

            pred_matrix = pred_total.data
            target_matrix = target.data
            if count_stack == 0:
                pred_matrix_total = pred_matrix
                target_matrix_total = target_matrix
                count_stack =1
            elif count_stack == 1:
                pred_matrix_total = torch.cat((pred_matrix_total,pred_matrix))
                target_matrix_total = torch.cat((target_matrix_total,target_matrix))

        target_matrix_total = target_matrix_total.cpu().numpy()
        pred_matrix_total = pred_matrix_total.cpu().numpy()
        tn, fp, fn, tp = confusion_matrix(target_matrix_total, pred_matrix_total,labels=[0,1]).ravel()

        test_loss /= len(target_test_loader.dataset)
        print(target_test_name, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(test_loss, correct_test.type(torch.float32), len(target_test_loader.dataset),100. * correct_test.type(torch.float32) / len(target_test_loader.dataset)))
        print('\nsource1 accnum {}, source2 accnum {}'.format(correct1.type(torch.float32), correct2.type(torch.float32)))

        test_total = 100*correct_test.type(torch.float32)/len(target_test_loader.dataset)
        
        logtest.append([test_total,tn, fp, fn, tp,correct1,correct2])
        np_log = np.array(logtest, dtype=float)
        np.savetxt(opt.save_test_name, np_log, delimiter=',', fmt='%.6f')

    return correct_test.type(torch.float32)

if __name__ == '__main__':
    model = models.MFSAN(num_classes=2)
    print(model)
    if cuda:
        model.cuda()

    kmm_source1_path = os.path.join(root_path,"source1/","train/","positive/")
    kmm_source2_path = os.path.join(root_path,"source2/","train/","positive/")
    kmm_target_path = os.path.join(root_path,"target/","train/","positive/")

    # print(kmm_source1_path,kmm_source2_path,kmm_target_path)

    KMM_weight_source1 = KMM_Lin.compute_kmm(kmm_source1_path,kmm_target_path)
    KMM_weight_source2 = KMM_Lin.compute_kmm(kmm_source2_path,kmm_target_path)

    KMM_weight_source1 = torch.from_numpy(KMM_weight_source1).float().to(DEVICE)
    print(KMM_weight_source1)

    KMM_weight_source2 = torch.from_numpy(KMM_weight_source2).float().to(DEVICE)
    print(KMM_weight_source2)

    train(model)

    torch.save(model.state_dict(), opt.save_model_path_name)
