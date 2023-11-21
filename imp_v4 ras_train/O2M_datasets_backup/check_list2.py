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

parser.add_argument("--check_fruit1",type=str, default="None",help='which fruit want to check')
parser.add_argument("--check_fruit2",type=str, default="None",help='which fruit want to check')
parser.add_argument("--check_fruit3",type=str, default="None",help='which fruit want to check')
parser.add_argument("--check_fruit4",type=str, default="None",help='which fruit want to check')
parser.add_argument("--check_fruit5",type=str, default="None",help='which fruit want to check')
parser.add_argument("--check_fruit6",type=str, default="None",help='which fruit want to check')
parser.add_argument("--check_fruit7",type=str, default="None",help='which fruit want to check')
parser.add_argument("--check_fruit8",type=str, default="None",help='which fruit want to check')

opt = parser.parse_args()

now_path = os.path.abspath('.') # 現在程式的位置
# train_list = ["apple","banana","carambola","guava","muskmelon","peach","pear","tomato"]



# 本程式是用來確認"來源與目標同種類時":source1500 與 target test positive 與 target train positive 是否沒有重疊

# Get File List
def Get_List(fruit):
    source_1500_path = os.path.join(now_path,"source_1500",fruit,"source_1500")

    if fruit == "apple":
        path1 = "apple_to_"
        path2 = "to_apple"
    
    if fruit == "banana":
        path1 = "banana_to_"
        path2 = "to_banana"
    
    if fruit == "carambola":
        path1 = "carambola_to_"
        path2 = "to_carambola"
    
    if fruit == "guava":
        path1 = "guava_to_"
        path2 = "to_guava"
    
    if fruit == "muskmelon":
        path1 = "muskmelon_to_"
        path2 = "to_muskmelon"
    
    if fruit == "peach":
        path1 = "peach_to_"
        path2 = "to_peach"
    
    if fruit == "pear":
        path1 = "pear_to_"
        path2 = "to_pear"
    
    if fruit == "tomato":
        path1 = "tomato_to_"
        path2 = "to_tomato"


    target_test_positive_path = os.path.join(now_path,path1,path2,"sn_200_sp_200\\target\\test\\positive")
    target_train_positive_path = os.path.join(now_path,path1,path2,"sn_200_sp_200\\target\\train\\positive")

    source_1500_list = os.listdir(source_1500_path)
    target_test_positive_list = os.listdir(target_test_positive_path)
    target_train_positive_list = os.listdir(target_train_positive_path)

    return source_1500_list,target_test_positive_list,target_train_positive_list

# check list
def check_list(f1_list1,f1_list2,f1_list3):
    flag = 0
    for elem in f1_list2:
        # print(elem) 
        if elem in f1_list1: 
            flag = 1
            print(elem) 
        if elem in f1_list3: 
            flag = 1
            print(elem) 
    for elem1 in f1_list1:
        # print(elem) 
        if elem1 in f1_list2: 
            flag = 1
            print(elem1) 
        if elem1 in f1_list3: 
            flag = 1
            print(elem1) 
    for elem2 in f1_list3:
        # print(elem) 
        if elem2 in f1_list1: 
            flag = 1
            print(elem2) 
        if elem2 in f1_list2: 
            flag = 1
            print(elem2) 
  
# checking condition 
    if flag == 1: 
        print("有重疊!!!!!!!!!!!!!!!!!!!!")  
    else : 
        print("沒重疊,ok") 
    

fruit1_list1,fruit1_list2,fruit1_list3 = Get_List(opt.check_fruit1)
fruit2_list1,fruit2_list2,fruit2_list3 = Get_List(opt.check_fruit2)
fruit3_list1,fruit3_list2,fruit3_list3 = Get_List(opt.check_fruit3)
fruit4_list1,fruit4_list2,fruit4_list3 = Get_List(opt.check_fruit4)
fruit5_list1,fruit5_list2,fruit5_list3 = Get_List(opt.check_fruit5)
fruit6_list1,fruit6_list2,fruit6_list3 = Get_List(opt.check_fruit6)
fruit7_list1,fruit7_list2,fruit7_list3 = Get_List(opt.check_fruit7)
fruit8_list1,fruit8_list2,fruit8_list3 = Get_List(opt.check_fruit8)

check_list(fruit1_list1,fruit1_list2,fruit1_list3)
check_list(fruit2_list1,fruit2_list2,fruit2_list3)
check_list(fruit3_list1,fruit3_list2,fruit3_list3)
check_list(fruit4_list1,fruit4_list2,fruit4_list3)
check_list(fruit5_list1,fruit5_list2,fruit5_list3)
check_list(fruit6_list1,fruit6_list2,fruit6_list3)
check_list(fruit7_list1,fruit7_list2,fruit7_list3)
check_list(fruit8_list1,fruit8_list2,fruit8_list3)