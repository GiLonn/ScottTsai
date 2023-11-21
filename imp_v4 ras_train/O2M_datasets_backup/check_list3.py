from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os,sys
import glob
import shutil
import random
import argparse

parser = argparse.ArgumentParser(description="check2")

parser.add_argument("--check_fruit1_path",type=str, default="None",help='which fruit want to check')
parser.add_argument("--check_fruit2_path",type=str, default="None",help='which fruit want to check')
parser.add_argument("--check_fruit3_path",type=str, default="None",help='which fruit want to check')
parser.add_argument("--check_fruit4_path",type=str, default="None",help='which fruit want to check')
parser.add_argument("--check_fruit5_path",type=str, default="None",help='which fruit want to check')
parser.add_argument("--check_fruit6_path",type=str, default="None",help='which fruit want to check')
parser.add_argument("--check_fruit7_path",type=str, default="None",help='which fruit want to check')
parser.add_argument("--check_fruit8_path",type=str, default="None",help='which fruit want to check')

opt = parser.parse_args()

now_path = os.path.abspath('.') # 現在程式的位置
# train_list = ["apple","banana","carambola","guava","muskmelon","peach","pear","tomato"]



# 本程式是用來確認"來源與目標同種類時":source train negative 與 target test negative是否沒有重疊

# Get File List
def Get_List(fruit_path):

    source_train_negative_path = os.path.join(fruit_path,"sn_200_sp_200\\source\\train\\negative")
    target_test_negative_path = os.path.join(fruit_path,"sn_200_sp_200\\target\\test\\negative")

    source_train_negative_list = os.listdir(source_train_negative_path)
    target_test_positive_list = os.listdir(target_test_negative_path)

    return source_train_negative_list,target_test_positive_list

# check list
def check_list(f1_list1,f1_list2):
    flag = 0
    for elem in f1_list2:
        # print(elem) 
        if elem in f1_list1: 
            flag = 1
  
# checking condition 
    if flag == 1: 
        print("有重疊!!!!!!!!!!!!!!!!!!!!")  
    else : 
        print("沒重疊,ok") 



fruit1_list1,fruit1_list2 = Get_List(opt.check_fruit1_path)
fruit2_list1,fruit2_list2 = Get_List(opt.check_fruit2_path)
fruit3_list1,fruit3_list2 = Get_List(opt.check_fruit3_path)
fruit4_list1,fruit4_list2 = Get_List(opt.check_fruit4_path)
fruit5_list1,fruit5_list2 = Get_List(opt.check_fruit5_path)
fruit6_list1,fruit6_list2 = Get_List(opt.check_fruit6_path)
fruit7_list1,fruit7_list2 = Get_List(opt.check_fruit7_path)
fruit8_list1,fruit8_list2 = Get_List(opt.check_fruit8_path)

check_list(fruit1_list1,fruit1_list2)
check_list(fruit2_list1,fruit2_list2)
check_list(fruit3_list1,fruit3_list2)
check_list(fruit4_list1,fruit4_list2)
check_list(fruit5_list1,fruit5_list2)
check_list(fruit6_list1,fruit6_list2)
check_list(fruit7_list1,fruit7_list2)
check_list(fruit8_list1,fruit8_list2)