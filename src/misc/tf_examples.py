import tensorflow as tf
from skimage import transform
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image,ImageEnhance
import numpy as np
from keras.preprocessing.image import img_to_array
from tensorflow.python.ops import math_ops
from scipy.ndimage import affine_transform


image_path='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/clean_test/0fcc458b4.jpg'
img=misc.imread(image_path)
# plt.imshow(img)
# plt.show()

def rotate(img):
    return transform.rotate(img, 10)
    
# with tf.Session() as sess:
#     op=tf.py_func(rotate,[img],tf.double)
#     tf.random_crop
#     img1=sess.run(op)
#     print img
#     plt.subplot('121')
#     plt.imshow(img)
#     plt.subplot('122')
#     plt.imshow(img1)
#     plt.show()
    
def adj_color(img_arr):
    image = Image.fromarray(img_arr.astype('uint8')).convert('RGB')
    color_factor=np.random.uniform(0,2)
    color_image=ImageEnhance.Color(image).enhance(color_factor)
    brightness_factor=np.random.uniform(0.6,1.4)
    brightness_image=ImageEnhance.Brightness(color_image).enhance(brightness_factor)
    contrast_factor=np.random.uniform(1.0,2.1)
    contrast_image=ImageEnhance.Contrast(brightness_image).enhance(contrast_factor)
    sharpness_factor=np.random.uniform(0,3.1)
    sharpness_image=ImageEnhance.Sharpness(contrast_image).enhance(sharpness_factor)
    return img_to_array(sharpness_image)

def ops():
    file_contents = tf.read_file(image_path)
    img = tf.image.decode_image(file_contents, channels=3)
    img = tf.cast(img, dtype=tf.uint8)
    img = tf.py_func(adj_color,[img],tf.uint8)
    
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

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

with tf.Session() as sess:
    file_contents= sess.run(tf.read_file(image_path))
    print type(file_contents)
    img = sess.run(tf.image.decode_image(file_contents, channels=3))
    print type(img)
    print img.shape
    img=tf.cast(img,tf.float32)
    op=tf.py_func(rotate_shear_zoom,[img],tf.float32)
    re = sess.run(op)
    
