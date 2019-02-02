from skimage import transform
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
from keras.preprocessing.image import img_to_array

image_path='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/clean_test/0fcc458b4.jpg'

def rotate():
    img1_path='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/train_backup/0ef8ede21.jpg'
    img1=misc.imread(img1_path)
    print img1.shape
    img2=transform.rotate(img1, 10, resize=False)
    print img2.shape
    img3=transform.rotate(img1,5, resize=False)
    
    print img3.shape
    plt.figure('rotate')
    plt.subplot(131)
    plt.title("origin")
    plt.imshow(img1)
    plt.subplot(132)
    plt.title("rotate 10 no resize")
    plt.imshow(img2)
    plt.subplot(133)
    plt.title('rotate -10 resize')
    plt.imshow(img3)
    plt.show()

def adj_color():
#     image=misc.imread(image_path)
    image=Image.open(image_path)
    print type(image)
    
    color_factor=np.random.uniform(0,2)
    color_image=ImageEnhance.Color(image).enhance(color_factor)
    brightness_factor=np.random.uniform(0.6,1.4)
    brightness_image=ImageEnhance.Brightness(color_image).enhance(brightness_factor)
    contrast_factor=np.random.uniform(1.0,2.1)
    contrast_image=ImageEnhance.Contrast(brightness_image).enhance(contrast_factor)
    sharpness_factor=np.random.uniform(0,3.1)
    sharpness_image=ImageEnhance.Sharpness(contrast_image).enhance(sharpness_factor)
    
    plt.figure('adjustment')
    plt.subplot(231)
    plt.title('origin')
    plt.imshow(image)
    plt.subplot(232)
    plt.title('color')
    plt.imshow(color_image)
    plt.subplot(233)
    plt.title('brightness')
    plt.imshow(brightness_image)
    plt.subplot(234)
    plt.title('contrast')
    plt.imshow(contrast_image)
    plt.subplot(235)
    plt.title('shapness')
    plt.imshow(sharpness_image)
    plt.show()
    img_arr=img_to_array(sharpness_image)
    img = Image.fromarray(img_arr.astype('uint8')).convert('RGB')
    plt.imshow(img)
    plt.show()
    misc.imsave('/Users/chendanxia/sophie/1.jpg', img_arr)
    
def adj_brightness():
    img = misc.imread(image_path)
    alpha=1.5
    print alpha
    
    beta=0
#     print img
    img1 = img.astype('float16')
    img2 = np.add(np.multiply(img1, alpha),beta)
    img2 = np.where(img2 >255,255,img2)
    img2 = np.where(img2 < 0, 0,img2)
#     print img2
    img2 = img2.astype('uint8')
    
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img2)
    plt.show()
    return img2

def adj_hue():
    img = misc.imread(image_path)
    alpha=1.2
    print alpha
    
#     print img
    img1 = img.astype('float16')
    img2 = np.copy(img1)
    img2[:,:,2] = img1[:,:,2]*alpha
    print img2
#     print img2
    img2 = img2.astype('uint8')
    
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img2)
    plt.show()
    return img2


# adj_color()
image_path='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/analysis/9fcd6e04d.jpg'
img=adj_hue()
misc.imsave('/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/analysis2/9fcd6e04d.jpg',img)
# a=np.array([[256,2],[3,4]])
# b=np.where(a> 255, 255, a)
# print b





























