from PIL import Image
import numpy as np
import os
from os.path import join

def match(path1, path2):
    i1 = Image.open(path1)
    i2 = Image.open(path2)
    if i1.mode != i2.mode or i1.size != i2.size: return False
    a1 = np.array(i1)
    a1 = a1 - a1.mean()
    a1 = a1 / np.sqrt((a1 ** 2).mean())
    a2 = np.array(i2)
    a2 = a2 - a2.mean()
    a2 = a2 / np.sqrt((a2 ** 2).mean())
    a = ((a1 - a2) ** 2).mean()
    if a > 0.1: return False
    return True
#     image1=Image.open(path1)
#     image2=Image.open(path2)
#     if image1.mode != image2.mode or image1.size != image2.size: return False
#     a1 = np.array(image1)
#     a2 = np.array(image2)
#     std1 = np.std(a1)
#     std2 = np.std(a2)
#     diff = std1-std2
#     print diff
#     if np.abs(diff) < 0.1: return True
#     return False

train='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/train_backup/'
test='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/test_backup/'

same_list=[]

for test_name in os.listdir(test):
    test_path=join(test,test_name)
    for train_name in os.listdir(train):
        train_path=join(train,train_name)
        if True == match(test_path, train_path):
            same_list.append([test_path, train_path])
            print [test_path, train_path]

print same_list
            