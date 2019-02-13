#coding=utf-8

from pandas import read_csv
from os.path import join
import numpy as np
import pandas as pd
import random
from scipy.ndimage import affine_transform
import os

base_dir='/home/nemo/kaggle/data/'
# base_dir='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/'
logs_base_dir = '/home/nemo/logs/kaggle'
models_base_dir = '/home/nemo/models/kaggle'
train_csv_path=base_dir+'train.csv'
test_image_name_file=base_dir+'test_list.txt'
new_whale_name='new_whale'
P2H = base_dir+'metadata/p2h.pickle'
P2SIZE = base_dir+'metadata/p2size.pickle'
BB_DF = base_dir+'metadata/bounding_boxes.csv'

p2bb = pd.read_csv(BB_DF).set_index("Image")
img_shape = (384, 384, 3)  # The image shape used by the model
anisotropy = 2.15  # The horizontal compression ratio
crop_margin = 0.05

def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    """
    Build a transformation matrix with the specified characteristics.
    """
    rotation = np.deg2rad(rotation)
    shear = np.deg2rad(shear)
    rotation_matrix = np.array(
        [[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    shear_matrix = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    zoom_matrix = np.array([[1.0 / height_zoom, 0, 0], [0, 1.0 / width_zoom, 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
    return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))

def read_cropped_image(image, image_path, augment=True):
    
    size_x = image.shape[1]
    size_y = image.shape[0]
    image_name=os.path.basename(image_path)
    row = p2bb.loc[image_name]
    x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']
    dx = x1 - x0
    dy = y1 - y0
    x0 -= dx * crop_margin
    x1 += dx * crop_margin + 1
    y0 -= dy * crop_margin
    y1 += dy * crop_margin + 1
    if x0 < 0:
        x0 = 0
    if x1 > size_x:
        x1 = size_x
    if y0 < 0:
        y0 = 0
    if y1 > size_y:
        y1 = size_y
    dx = x1 - x0
    dy = y1 - y0
    if dx > dy * anisotropy:
        dy = 0.5 * (dx / anisotropy - dy)
        y0 -= dy
        y1 += dy
    else:
        dx = 0.5 * (dy * anisotropy - dx)
        x0 -= dx
        x1 += dx

    # Generate the transformation matrix
    trans = np.array([[1, 0, -0.5 * img_shape[0]], [0, 1, -0.5 * img_shape[1]], [0, 0, 1]])
    trans = np.dot(np.array([[(y1 - y0) / img_shape[0], 0, 0], [0, (x1 - x0) / img_shape[1], 0], [0, 0, 1]]), trans)
    if augment:
        trans = np.dot(build_transform(
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(0.8, 1.0),
            random.uniform(0.8, 1.0),
            random.uniform(-0.05 * (y1 - y0), 0.05 * (y1 - y0)),
            random.uniform(-0.05 * (x1 - x0), 0.05 * (x1 - x0))
        ), trans)
    trans = np.dot(np.array([[1, 0, 0.5 * (y1 + y0)], [0, 1, 0.5 * (x1 + x0)], [0, 0, 1]]), trans)

    # Apply affine transformation
    matrix = trans[:2, :2]
    offset = trans[:2, 2]
    
    img0 = image[:,:,0]
    img1 = image[:,:,1]
    img2 = image[:,:,2]
    img0 = affine_transform(img0, matrix, offset, output_shape=img_shape[:-1], order=1, mode='constant',
                           cval=np.average(img0))
    img1 = affine_transform(img1, matrix, offset, output_shape=img_shape[:-1], order=1, mode='constant',
                           cval=np.average(img1))
    img2 = affine_transform(img2, matrix, offset, output_shape=img_shape[:-1], order=1, mode='constant',
                           cval=np.average(img2))
    image = np.stack([img0,img1,img2],2)
    
#     image = image.reshape(img_shape)
    
    return image


image_name_2_class_name=dict([(y,z) for x,y,z in read_csv(train_csv_path).to_records()])
new_whale_name='new_whale'
def get_image_name_list(file_path,ignore_new_whale=True):
    with open(file_path) as f:
        whales = f.read().split(" ")
        if ignore_new_whale:
            clean_whale=[]
            for whale in whales:
                if image_name_2_class_name[whale]!=new_whale_name:
                    clean_whale.append(whale)
            return clean_whale
        else:
            return whales

def get_class_name_2_image_names(forAll=False):
    test_image_name_list=get_image_name_list(test_image_name_file)
    class_name_2_image_names={}
    for image_name in image_name_2_class_name.keys():
        class_name=image_name_2_class_name.get(image_name)
        if class_name==new_whale_name:
            continue
        if image_name in test_image_name_list:
            if not forAll:
                continue
        if class_name not in class_name_2_image_names:
            image_names=[]
        else:
            image_names=class_name_2_image_names.get(class_name)
        if image_name not in image_names:
            image_names.append(image_name)
        class_name_2_image_names[class_name]=image_names
    return class_name_2_image_names

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)

def get_data_set_train(path_dir, iggore_single_image, useAll=False, ignore_new_whale=True):
    if useAll:
        class_name_2_image_names=get_class_name_2_image_names(True)
    else:
        class_name_2_image_names=get_class_name_2_image_names(False)
    classes = class_name_2_image_names.keys()
    classes.sort()
    class_num = len(classes)
    dataset=[]
    for i in range(class_num):
        class_name=classes[i]
        if ignore_new_whale and class_name==new_whale_name:
            continue
        image_paths=[join(path_dir,img) for img in class_name_2_image_names.get(class_name)]
        if iggore_single_image:
            if len(image_paths)==1:
                continue
        dataset.append(ImageClass(class_name, image_paths))
    return dataset

def isZhishu(num):
    for i in range(2,num):
        if num % i == 0:
            print(i,num/i)
    else:
        print (num, '是质数')

