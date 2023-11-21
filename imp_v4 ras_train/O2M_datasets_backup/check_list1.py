from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os,sys
import glob
import shutil
import random
import argparse

parser = argparse.ArgumentParser(description="check1")

parser.add_argument("--check_fruit_path1",type=str, default="None",help='which fruit want to check')
parser.add_argument("--check_fruit_path2",type=str, default="None",help='which fruit want to check')
parser.add_argument("--check_fruit_path3",type=str, default="None",help='which fruit want to check')
parser.add_argument("--check_fruit_path4",type=str, default="None",help='which fruit want to check')
parser.add_argument("--check_fruit_path5",type=str, default="None",help='which fruit want to check')
parser.add_argument("--check_fruit_path6",type=str, default="None",help='which fruit want to check')
parser.add_argument("--check_fruit_path7",type=str, default="None",help='which fruit want to check')
parser.add_argument("--check_fruit_path8",type=str, default="None",help='which fruit want to check')

opt = parser.parse_args()

now_path = os.path.abspath('.') # 現在程式的位置
# train_list = ["apple","banana","carambola","guava","muskmelon","peach","pear","tomato"]



# 本程式是用來確認是否每個"同種類之目標模型中":目標測試正樣本、目標測試負樣本、目標訓練正樣本、來源負樣本是否一樣

# Get File List
def Get_List(fruit_path):
    target_test_positive_path = os.path.join(fruit_path,"sn_200_sp_200\\target\\test\\positive")
    target_test_negative_path = os.path.join(fruit_path,"sn_200_sp_200\\target\\test\\negative")
    target_train_positive_path = os.path.join(fruit_path,"sn_200_sp_200\\target\\train\\positive")
    source_train_negative_path = os.path.join(fruit_path,"sn_200_sp_200\\source\\train\\negative")

    target_test_positive_list = os.listdir(target_test_positive_path)
    target_test_negative_list = os.listdir(target_test_negative_path)
    target_train_positive_list = os.listdir(target_train_positive_path)
    source_train_negative_list = os.listdir(source_train_negative_path)

    return target_test_positive_list,target_test_negative_list,target_train_positive_list,source_train_negative_list

# check list
def check_list(f1_list1,f1_list2,f1_list3,f1_list4,f2_list1,f2_list2,f2_list3,f2_list4):
    if(f1_list1!=f2_list1):
        print("f1_list1 & f2_list1 are different")
    if(f1_list2!=f2_list2):
        print("f1_list2 & f2_list2 are different")
    if(f1_list3!=f2_list3):
        print("f1_list3 & f2_list3 are different")
    if(f1_list4!=f2_list4):
        print("f1_list4 & f2_list4 are different")
    

fruit1_list1,fruit1_list2,fruit1_list3,fruit1_list4 = Get_List(opt.check_fruit_path1)
fruit2_list1,fruit2_list2,fruit2_list3,fruit2_list4 = Get_List(opt.check_fruit_path2)
fruit3_list1,fruit3_list2,fruit3_list3,fruit3_list4 = Get_List(opt.check_fruit_path3)
fruit4_list1,fruit4_list2,fruit4_list3,fruit4_list4 = Get_List(opt.check_fruit_path4)
fruit5_list1,fruit5_list2,fruit5_list3,fruit5_list4 = Get_List(opt.check_fruit_path5)
fruit6_list1,fruit6_list2,fruit6_list3,fruit6_list4 = Get_List(opt.check_fruit_path6)
fruit7_list1,fruit7_list2,fruit7_list3,fruit7_list4 = Get_List(opt.check_fruit_path7)
fruit8_list1,fruit8_list2,fruit8_list3,fruit8_list4 = Get_List(opt.check_fruit_path8)

check_list(fruit1_list1,fruit1_list2,fruit1_list3,fruit1_list4,fruit2_list1,fruit2_list2,fruit2_list3,fruit2_list4)

check_list(fruit1_list1,fruit1_list2,fruit1_list3,fruit1_list4,fruit3_list1,fruit3_list2,fruit3_list3,fruit3_list4)

check_list(fruit1_list1,fruit1_list2,fruit1_list3,fruit1_list4,fruit4_list1,fruit4_list2,fruit4_list3,fruit4_list4)

check_list(fruit1_list1,fruit1_list2,fruit1_list3,fruit1_list4,fruit5_list1,fruit5_list2,fruit5_list3,fruit5_list4)

check_list(fruit1_list1,fruit1_list2,fruit1_list3,fruit1_list4,fruit6_list1,fruit6_list2,fruit6_list3,fruit6_list4)

check_list(fruit1_list1,fruit1_list2,fruit1_list3,fruit1_list4,fruit7_list1,fruit7_list2,fruit7_list3,fruit7_list4)

check_list(fruit1_list1,fruit1_list2,fruit1_list3,fruit1_list4,fruit8_list1,fruit8_list2,fruit8_list3,fruit8_list4)