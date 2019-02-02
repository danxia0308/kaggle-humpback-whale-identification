import tensorflow as tf
from skimage import transform
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image,ImageEnhance
import numpy as np
from keras.preprocessing.image import img_to_array
from tensorflow.python.ops import math_ops


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

with tf.Session() as sess:
    file_contents= sess.run(tf.read_file(image_path))
    print type(file_contents)
    img = sess.run(tf.image.decode_image(file_contents, channels=3))
    print type(img)
    print img.shape
    
    op=(tf.cast(img, tf.float32) - 127.5)/128.0
    re = sess.run(op)
    re2 =np.divide(np.subtract(img, 127.5),128.0)
    print re
    print re-re2
    
