#coding=utf-8
from __future__ import absolute_import
import csv 
import numpy as np
import os
import shutil
from pandas import read_csv
import cv2
import aircv as ac
from PIL.ImageDraw import Draw
from scipy import misc
import matplotlib.pyplot as plt
from keras.preprocessing import image
import tensorflow as tf
from skimage import transform
import numpy as np
from tensorflow.python.framework import ops
from PIL import Image
from keras.preprocessing.image import img_to_array
from scipy.ndimage.interpolation import affine_transform

img1_path='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/clean_test/0a0ec5a23.jpg'
img=Image.open(img1_path).convert('L')
img = img_to_array(img)

shear=5
shear_rad=np.deg2rad(shear)
shear_matrix = np.array([[1, np.sin(shear_rad), 0], [0, np.cos(shear_rad), 0], [0, 0, 1]])

matrix = shear_matrix[:2, :2]
offset = shear_matrix[:2, 2]
img_shape=img.shape
img = img.reshape(img.shape[:-1])
img = affine_transform(img, matrix, offset, output_shape=img_shape[:-1], order=1, mode='constant',
                           cval=np.average(img))
print img_shape[:2]
img = img.reshape(img_shape[:2])
plt.subplot(121)
img1=misc.imread(img1_path)
plt.imshow(img1)
plt.subplot(122)
plt.imshow(img)
plt.show()

def XWarp(image,angle):
    a = math.tan(angle*math.pi/180.0)
    W = image.width
    H = int(image.height+W*a)
    size = (W,H)
    iWarp = cv2.CreateImage(size,image.depth,image.nChannels)
    for i in range(image.height):
        for j in range(image.width):
            x = int(i+j*a)
            iWarp[x,j] = image[i,j]
    return iWarp


# img1_path='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/clean_test_384/0a0ec5a23.jpg'
# file_contents = tf.read_file(img1_path)
# image = misc.imread(img1_path)
# print (image.shape)
# angle = np.random.uniform(low=-10.0, high=10.0)
# print ("image.shape",image.shape)
# image.set_shape((384, 384, 3))
# image=transform.resize(image,(384,384,3))
# image=transform.rotate(image, angle)
# image = misc.imrotate([image], angle, 'bicubic')

# def find_image_pos():               
#     img1_path='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/train_backup/0ef8ede21.jpg'
#     img2_path='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/clean_train_mine_v1/0ef8ede21.jpg'
#     
#     img1=ac.imread(img1_path)
#     img2=ac.imread(img2_path)
#     
#     pos=ac.find_template(img1,img2)
#     print (pos)
#     print (pos.get('rectangle'))
#     x0=pos.get('rectangle')[0][0]
#     y0=pos.get('rectangle')[0][1]
#     x1=pos.get('rectangle')[3][0]
#     y1=pos.get('rectangle')[3][1]
#     
#     print x0,y0,x1,y1
#     
#     
#     # img1=misc.imread(img1_path)
#     img1=image.load_img(img1_path)
#     img2=image.load_img(img2_path)
#     print img1.size
#     print img2.size
#     draw=Draw(img1)
#     draw.rectangle([x0,y0,x1,y1], outline='red')
#     plt.imshow(img1)
#     plt.show()





# def train_count():
#     train_csv_path='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/train.csv'
#     image_name_2_class_name=dict([(y,z) for x,y,z in read_csv(train_csv_path).to_records()])
#     class_name_2_image_names_all={}
#     for image_name in image_name_2_class_name.keys():
#         class_name=image_name_2_class_name.get(image_name)
#         if class_name not in class_name_2_image_names_all:
#             image_names=[]
#         else:
#             image_names=class_name_2_image_names_all.get(class_name)
#         if image_name not in image_names:
#             image_names.append(image_name)
#         class_name_2_image_names_all[class_name]=image_names
#     print len(class_name_2_image_names_all.keys())
#     
#     size_list=[]
#     count=0
#     for image_names in class_name_2_image_names_all.values():
#         len1=len(image_names)
#         size_list.append(len1)
#         if (len1 >5):
#             count=count+1
#     print count
    
    
