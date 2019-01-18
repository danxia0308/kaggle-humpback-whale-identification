#coding=utf-8

import csv 
import numpy as np
import os
import util

path='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/train_160'
threshold=5
count=0
for dir in os.listdir(path):
    files=os.listdir(os.path.join(path,dir))
    size=len(files)
    if size >= threshold:
        count=count+1
print count

util.isZhishu(count)