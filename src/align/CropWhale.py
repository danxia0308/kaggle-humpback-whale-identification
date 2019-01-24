from __future__ import division
import os
import PIL
from PIL import Image
from PIL.ImageDraw import Draw
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
import math
import shutil


model_base='/Users/chendanxia/sophie/kaggle/models'
train_path='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/test/'
crop_path='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/train_crop_test/'
model=load_model(os.path.join(model_base,'cropping.model'))
bad_file_list=[]
margin_scale=0.1

def move_bad_list_files():
    bad_list_file='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/bad_list.txt'
    with open(bad_list_file) as f:
        lines=f.readlines()
        for line in lines:
            index=line.rfind("\n")
            line=line[:index]
            print line
            file_name=os.path.basename(line)
            dir_name=os.path.basename(os.path.dirname(line))
            print dir_name,file_name
            dst_dir=os.path.join(crop_path,dir_name)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            dst_file_path=os.path.join(dst_dir,file_name)
            shutil.copyfile(line, dst_file_path)
def crop_image(image_path, cropped_image_path):

    img=image.load_img(image_path)
    img_arr = image.img_to_array(img)
    
    old_shape=img_arr.shape
    old_height=old_shape[0]
    old_width=old_shape[1]
    new_height=128
    new_width=128
#     print 'old_shape',old_shape
#     print 'old_height',old_height
#     print 'old_width',old_width
#     plt.imshow(img)
#     plt.show()
    
    # Get the cropped coordinates
    rimg=img.resize((128,128), PIL.Image.ANTIALIAS)
    rimg_L=rimg.convert('L')
    rimg_L_arr=image.img_to_array(rimg_L)
    bbox=model.predict(np.expand_dims(rimg_L_arr, axis=0))
    x0=bbox[0][0]
    y0=bbox[0][1]
    x1=bbox[0][2]
    y1=bbox[0][3]
    # process the edge
    if x0<0:
        x1=np.min([math.fabs(x0)+x1,new_height-1])
        x0=0
    if y0<0:
        y1=np.min([math.fabs(y0)+x1,new_width-1])
        y0=0
    if x1>=new_height:
        x0=np.max([0,x0-(x1-new_height+1)])
        x1=new_height-1
    if y1>=new_width:
        y0=np.max([0,y0-(y1-new_width+1)])
        y1=new_width-1
    #Covert to the coordinates before resize.
    x0_old=x0*old_height/new_height
    x1_old=x1*old_height/new_height
    y0_old=y0*old_width/new_width
    y1_old=y1*old_width/new_width
    
#     draw = Draw(rimg)
#     draw.rectangle(bbox[0], outline='red')
#     plt.imshow(rimg)
#     plt.show()
    
#     draw1=Draw(img)
#     draw1.rectangle([x0_old,y0_old,x1_old,y1_old],outline='green')
#     plt.imshow(img)
#     plt.show()
    
    print 'bbox',bbox
    print 'coord',[x0,y0,x1,y1]
    print 'trans_coor',[x0_old,y0_old,x1_old,y1_old]
    
    img_crop=rimg.crop([x0,y0,x1,y1])
    image.save_img(cropped_image_path,img_crop)
    
#     if (x0_old >= x1_old or y0_old >=y1_old):
#         print "bad crop box."
#         print cropped_image_path
#         bad_file_list.append(cropped_image_path)
#         image.save_img(cropped_image_path,img)
#     else:
#         coordinate_with_margin=do_margin([x0_old,y0_old,x1_old,y1_old],old_width,old_height)
#         coordinate_with_margin=[x0_old,y0_old,x1_old,y1_old]
#         img_crop_old=img.crop(coordinate_with_margin)
# #         plt.imshow(img_crop_old)
# #         plt.show()
#         image.save_img(cropped_image_path,img_crop_old)

def do_margin(arr,width,height):
    width_margin=width*margin_scale
    height_margin=height*margin_scale
    x0=arr[0]
    y0=arr[1]
    x1=arr[2]
    y1=arr[3]
    x0=np.max([0,x0-height_margin])
    y0=np.max([0,y0-width_margin])
    x1=np.min([height-1,x1+height_margin])
    y1=np.min([width-1,y1+width_margin])
#     return tuple([x0,y0,y1,x1]) #crop: left upper right lower
    return tuple([x0,y0,x1,y1])

def do_crop(src_file_path, crop_file_path):
    try:
        crop_image(src_file_path, crop_file_path)
    except Exception:
        print 'meet with exception'
        print src_file_path
        bad_file_list.append(src_file_path)
        shutil.copyfile(src_file_path, crop_file_path)
def main():
    if not os.path.exists(crop_path):
        os.makedirs(crop_path)
    for sub_dir_name in os.listdir(train_path):
        sub_dir_path=os.path.join(train_path,sub_dir_name)
        if not os.path.isdir(sub_dir_path):
            do_crop(sub_dir_path, os.path.join(crop_path,sub_dir_name))
            continue
        crop_dir_path=os.path.join(crop_path,sub_dir_name)
        if not os.path.exists(crop_dir_path):
            os.makedirs(crop_dir_path)
        for file_name in os.listdir(sub_dir_path):
            src_file_path=os.path.join(sub_dir_path,file_name)
            crop_file_path=os.path.join(crop_dir_path,file_name)
            do_crop(src_file_path, crop_file_path)
        
main()
# print bad_file_list
# print len(bad_file_list)
#  
# with open('/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/bad_list.txt','w') as f:
#     for bad_file in bad_file_list:
#         f.write(bad_file+"\n")

# crop_image('/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/train_new/w_3a1e62f/9086b3d6e.jpg', '/Users/chendanxia/sophie/1.jpg')

# move_bad_list_files()
        
        
        
