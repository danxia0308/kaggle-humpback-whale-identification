from __future__ import division
import os
import PIL
from PIL import Image,ImageTransform
from PIL.ImageDraw import Draw
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
import math
import shutil
from skimage import transform
from scipy import misc
from keras.preprocessing.image import img_to_array
from scipy.ndimage import affine_transform
from matplotlib.pyplot import subplot


model_base='/Users/chendanxia/sophie/kaggle/models'
train_path='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/test/'
crop_path='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/train_crop_test/'
model=load_model(os.path.join(model_base,'cropping.model'))
height_margin_scale_up=0.2#0.01
height_margin_scale_low=0.05#0.01
width_margin_scale=0.2

def crop_image(image_path, cropped_image_path):

    img=image.load_img(image_path)
    img_arr = image.img_to_array(img)
    
    old_shape=img_arr.shape
    old_height=old_shape[0]
    old_width=old_shape[1]
    new_height=128
    new_width=128
    
    # Get the cropped coordinates
    rimg=img.resize((128,128), PIL.Image.ANTIALIAS)
    rimg_L=rimg.convert('L')
    rimg_L_arr=image.img_to_array(rimg_L)
    bbox=model.predict(np.expand_dims(rimg_L_arr, axis=0))
    x0=bbox[0][0]  
    y0=bbox[0][1]
    x1=bbox[0][2]
    y1=bbox[0][3]
    
    arr=[x0,y0,x1,y1]
    if (x0 < 0):
        arr=[-x0,y0,x1-x0,y1]
    if (y0 < 0):
        arr=[x0,-y0,x1,y1-y0]
    
    # process the edge
    if x0<0:
        x1=np.min([math.fabs(x0)+x1,new_width-1])
        x0=0
    if y0<0:
        y1=np.min([math.fabs(y0)+x1,new_height-1])
        y0=0
    if x1>=new_width-1:
        x0=np.max([0,x0-(x1-new_width+1)])
        x1=new_width-1
    if y1>=new_height-1:
        y0=np.max([0,y0-(y1-new_height+1)])
        y1=new_height-1
    #Covert to the coordinates before resize.
    print 'old_width',old_width,'new_width',new_width,'old_width/new_width',old_width/new_width
    
    x0_old=x0*old_width/new_width
    x1_old=x1*old_width/new_width
    y0_old=y0*old_height/new_height
    y1_old=y1*old_height/new_height
    print 'x1',x1,'x1_old',x1_old
    
    print 'bbox',bbox
    print 'coord',[x0,y0,x1,y1]
    print 'trans_coor',[x0_old,y0_old,x1_old,y1_old]
    trans_arr=[x0_old,y0_old,x1_old,y1_old]
    tran_arr_magin=do_margin(trans_arr, old_width, old_height)
    print 'trans_arr',trans_arr
    print 'tran_arr_magin',tran_arr_magin
    
#     draw = Draw(rimg)
#     draw.rectangle(bbox[0], outline='red')
#     
#     draw.rectangle(arr,outline='green')
#     
#     arr=do_margin(arr,new_width,new_height);
#     print 'margin:',arr
#     draw.rectangle(arr,outline='blue')
    
#     plt.imshow(rimg)
#     plt.show()
    
#     draw1=Draw(img)
#     draw1.rectangle(trans_arr,outline='green')
#     draw1.rectangle(tran_arr_magin,outline='red')
#     
#     plt.imshow(img)
#     plt.show()
    
   
    
    img_crop=img.crop(tran_arr_magin)
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
    width_margin=width*width_margin_scale
    height_margin_up=height*height_margin_scale_up
    height_margin_low=height*height_margin_scale_low
    x0=arr[0]
    y0=arr[1]
    x1=arr[2]
    y1=arr[3]
    x0=np.max([0,x0-width_margin])
    y0=np.max([0,y0-height_margin_up])
    x1=np.min([width-1,x1+width_margin])
    y1=np.min([height-1,y1+height_margin_low])
#     return tuple([x0,y0,y1,x1]) #crop: left upper right lower
    return [x0,y0,x1,y1]

def do_crop(src_file_path, crop_file_path):
    try:
        crop_image(src_file_path, crop_file_path)
    except Exception:
        print 'meet with exception'
        print src_file_path
#         shutil.copyfile(src_file_path, crop_file_path)
        

def crop_bottom(image_path, cropped_image_path,ratio):
    img=image.load_img(image_path)
    img_arr = image.img_to_array(img)
    
    old_shape=img_arr.shape
    height=old_shape[0]
    width=old_shape[1]
    img_crop=img.crop([0,0,width-1,height*(1-ratio)])
    image.save_img(cropped_image_path,img_crop)

def crop_top(image_path, cropped_image_path,ratio):
    img=image.load_img(image_path)
    img_arr = image.img_to_array(img)
    
    old_shape=img_arr.shape
    height=old_shape[0]
    width=old_shape[1]
    img_crop=img.crop([0,height*ratio,width-1,height-1])
    image.save_img(cropped_image_path,img_crop)
    
def crop_left(image_path, cropped_image_path,ratio):
    img=image.load_img(image_path)
    img_arr = image.img_to_array(img)
    
    old_shape=img_arr.shape
    height=old_shape[0]
    width=old_shape[1]
    img_crop=img.crop([width*ratio,0,width-1,height-1])
    image.save_img(cropped_image_path,img_crop)

def crop_right(image_path, cropped_image_path,ratio):
    img=image.load_img(image_path)
    img_arr = image.img_to_array(img)
    
    old_shape=img_arr.shape
    height=old_shape[0]
    width=old_shape[1]
    img_crop=img.crop([0, 0,width*(1-ratio),height-1])
    image.save_img(cropped_image_path,img_crop)
    
        
# image_path='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/test/0aec113b3.jpg'
# dst_image_path='/Users/chendanxia/sophie/1.jpg'
# crop_image(image_path, dst_image_path)

def crop_by_model():
    src_dir='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/train/'
    dst_dir='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/train_temp/'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for file_name in os.listdir(src_dir):
        file_path=os.path.join(src_dir,file_name)
        dst_file_path=os.path.join(dst_dir,file_name)
        do_crop(file_path, dst_file_path)

def rotate(file_path,dst_file_path,angle):
    img1 = misc.imread(file_path)
    img2 =transform.rotate(img1, angle,resize=False)
    misc.imsave(dst_file_path,img2)
    
def crop_bottom_or_top():
    src_dir='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/analysis/'
    dst_dir='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/analysis2/'
    for file_name in os.listdir(src_dir):
        if file_name=='.DS_Store':
            continue
        file_path=os.path.join(src_dir,file_name)
        dst_file_path=os.path.join(dst_dir,file_name)
        
        crop_top(file_path, dst_file_path,0.3)
        crop_bottom(dst_file_path, dst_file_path,0.4)
        crop_left(dst_file_path, dst_file_path,0.2)
        crop_right(dst_file_path, dst_file_path,0.3)
        img1 = misc.imread(file_path)
#         rotate(file_path,dst_file_path,180)
        img2 = misc.imread(dst_file_path)
        
        plt.subplot(121)
        plt.imshow(img1)
        plt.subplot(122)
        plt.imshow(img2)
        plt.show()
        
def Xshear(angle, img):
    
    print img.shape
    angle_arc=np.deg2rad(math.fabs(angle))
    alpha=np.tan(angle_arc)
    
    H=(img.shape[0]+int(alpha*img.shape[1]))
    W=img.shape[1]
    shape=(H,W,img.shape[2])
    print shape
    shear_img=np.zeros(shape)
    print shear_img.shape
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
#             print int(i+alpha*j),j,'=',i,j
            if angle < 0:
                y_1=int(i+alpha*(W-j))
            else:
                y_1=int(i+alpha*j)
            shear_img[y_1,j]=img[i,j]
    
    shear_img=shear_img.astype('uint')
    return shear_img
    

def Yshear(angle,img):
    angle_arc=np.deg2rad(math.fabs(angle))
    alpha=np.tan(angle_arc)
    H=img.shape[0]
    W=img.shape[1]+int(alpha*img.shape[0])
    shape=(H,W,img.shape[2])
    shear_img=np.zeros(shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if angle < 0:
                x_1=int(j+alpha*i)
            else:
                x_1=int(j+alpha*(H-i))
            shear_img[i, x_1]=img[i,j]
    shear_img=shear_img.astype('uint')
    return shear_img

def TYshear(angle,img):
    angle_arc=np.deg2rad(math.fabs(angle))
    alpha=np.sin(angle_arc)
    H=img.shape[0]
    W=img.shape[1]+int(alpha*img.shape[0])
    shape=(H,W,img.shape[2])
    shear_img=np.zeros(shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if angle < 0:
                x_1=int(j+alpha*i)
            else:
                x_1=int(j+alpha*(H-i))
            shear_img[i, x_1]=img[i,j]
    shear_img=shear_img.astype('uint')
    return shear_img

def rotate_shear_zoom(img):
    rotate=np.random.uniform(-5, 5)
    shear=np.random.uniform(-5, 5)
    height_zoom=np.random.uniform(0.8, 1.0)
    width_zoom=np.random.uniform(0.8, 1.0)
    rotate_rad=np.deg2rad(rotate)
    shear_rad=np.deg2rad(shear)
    
    trans_rotate=np.array([[np.cos(rotate_rad),np.sin(rotate_rad),0],[-np.sin(rotate_rad),np.cos(rotate_rad),0],[0,0,1]])
    trans_shear=np.array([[1,np.sin(shear_rad),0],[0,np.cos(shear_rad),0],[0,0,1]])
    trans_zoom=np.array([[1.0/height_zoom,0,0],[0,1.0/width_zoom,0],[0,0,1]])
    
    trans=np.dot(np.dot(trans_rotate,trans_shear),trans_zoom)
    matrix=trans[:2,:2]
    offset=trans[:2,2]
    
    img1=affine_transform(img[:,:,0], matrix, offset,order=1, mode='constant', cval=np.average(img))
    img2=affine_transform(img[:,:,1], matrix, offset,order=1, mode='constant', cval=np.average(img))
    img3=affine_transform(img[:,:,2], matrix, offset,order=1, mode='constant', cval=np.average(img))
    img=np.stack([img1,img2,img3],axis=2)
    
    return img

def test1():
    path='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/clean_test/0a8e58929.jpg'
    img0=misc.imread(path)
    shape=img0.shape
    x0=34.6
    y0=120.85
    x1=469.547
    y1=323.15
    rotate=np.random.uniform(-5, 5)
    shear=np.random.uniform(-5, 5)
    height_zoom=np.random.uniform(0.8, 1.0)
    width_zoom=np.random.uniform(0.8, 1.0)
    shift_height=np.random.uniform(-0.05 * (y1 - y0), 0.05 * (y1 - y0))
    shift_width=np.random.uniform(-0.05 * (x1 - x0), 0.05 * (x1 - x0))
    
    img_shape=(384,384,1)
    trans_recenter=np.array([[1,0,-0.5*img_shape[0]],[0,1,-0.5*img_shape[1]],[0,0,1]])
    trans_scale=np.array([[(y1-y0)/img_shape[0],0,0],[0,(x1-x0)/img_shape[1],0],[0,0,1]])
    rotate_rad=np.deg2rad(rotate)
    shear_rad=np.deg2rad(shear)
    trans_rotate=np.array([[np.cos(rotate_rad),np.sin(rotate_rad),0],[-np.sin(rotate_rad),np.cos(rotate_rad),0],[0,0,1]])
    trans_shift=np.array([[1,0,shift_height],[0,1,shift_width],[0,0,1]])
    trans_shear=np.array([[1,np.sin(shear_rad),0],[0,np.cos(shear_rad),0],[0,0,1]])
    trans_zoom=np.array([[1.0/height_zoom,0,0],[0,1.0/width_zoom,0],[0,0,1]])
    trans_recenter2=np.array([[1,0,0.5*(y1+y0)],[0,1,0.5*(x1+x0)],[0,0,1]])
    trans0=np.dot(np.dot(trans_rotate,trans_shear),np.dot(trans_zoom,trans_shift))
    trans=np.dot(trans_scale,trans_recenter)
    trans=np.dot(trans0,trans)
    trans=np.dot(trans_recenter2,trans)
    print trans
    
    img=Image.open(path)#.convert('L')
    img=img_to_array(img)
    matrix=trans[:2,:2]
    offset=trans[:2,2]
    
    img1=affine_transform(img[:,:,0], matrix, offset, output_shape=img_shape[:-1],order=1, mode='constant', cval=np.average(img))
    img2=affine_transform(img[:,:,1], matrix, offset, output_shape=img_shape[:-1],order=1, mode='constant', cval=np.average(img))
    img3=affine_transform(img[:,:,2], matrix, offset, output_shape=img_shape[:-1],order=1, mode='constant', cval=np.average(img))

    img=np.stack([img1,img2,img3],axis=2)

    img=img.astype('uint8')
    plt.subplot(121)
    plt.imshow(img0)
    plt.subplot(122)
    plt.imshow(img)
    plt.show()

def test():
    path='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/clean_test/0a8e58929.jpg'
    img0=misc.imread(path)
    
    for i in range(1,10):
        sub=33*10+i
        plt.subplot(sub)
        img=rotate_shear_zoom(img0)
        img=img.astype('uint8')
        plt.imshow(img)
    
#     img=rotate_shear_zoom(img0)
#     img=img.astype('uint8')
#     plt.subplot(121)
#     plt.imshow(img0)
#     plt.subplot(122)
#     plt.imshow(img)
    plt.show()

test()

   
# crop_bottom_or_top()
























        
        
