from pandas import read_csv
import pandas as pd
from os.path import isfile
import os
from scipy import misc
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
from scipy.ndimage import affine_transform
import random
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform

base_dir='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/'
TRAIN_DF = base_dir+'train.csv'
SUB_Df = base_dir+'sample_submission.csv'
TRAIN = base_dir+'train_backup/'
TEST = base_dir+'test_backup/'
P2H = base_dir+'metadata/p2h.pickle'
P2SIZE = base_dir+'metadata/p2size.pickle'
BB_DF = base_dir+'metadata/bounding_boxes.csv'
train_image_name_file=base_dir+'/train_list.txt'
test_image_name_file=base_dir+'/test_list.txt'
tagged = dict([(p, w) for _, p, w in read_csv(TRAIN_DF).to_records()])
# submit = [p for _, p, _ in read_csv(SUB_Df).to_records()]
# join = list(tagged.keys()) + submit

p2bb = pd.read_csv(BB_DF).set_index("Image")

img_shape = (384, 384, 1)  # The image shape used by the model
anisotropy = 2.15  # The horizontal compression ratio
crop_margin = 0.05


def get_path(image_name):
    if isfile(os.path.join(TRAIN,image_name)):
        return os.path.join(TRAIN,image_name)
    if isfile(os.path.join(TEST,image_name)):
        return os.path.join(TEST,image_name)    

def read_crop_image(image_name):
    image_path=get_path(image_name)
    img1=image.load_img(image_path)
    img_arr = image.img_to_array(img1)
    size_x=img_arr.shape[0]
    size_y=img_arr.shape[1]
    
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
    print x0,y0,x1,y1
    print (x1-x0)/(y1-y0)
    
    trans = np.array([[1, 0, -0.5 * img_shape[0]], [0, 1, -0.5 * img_shape[1]], [0, 0, 1]])
    trans = np.dot(np.array([[(y1 - y0) / img_shape[0], 0, 0], [0, (x1 - x0) / img_shape[1], 0], [0, 0, 1]]), trans)
    
    trans = np.dot(build_transform(
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(0.8, 1.0),
            random.uniform(0.8, 1.0),
            random.uniform(-0.05 * (y1 - y0), 0.05 * (y1 - y0)),
            random.uniform(-0.05 * (x1 - x0), 0.05 * (x1 - x0))
        ), trans)
    
#     trans = np.dot(build_transform(
#             3.0,
#             3.0,
#             0.8,
#             0.8,
#             -8.7,
#             10.75
#         ), trans)
    
    
    trans = np.dot(np.array([[1, 0, 0.5 * (y1 + y0)], [0, 1, 0.5 * (x1 + x0)], [0, 0, 1]]), trans)
    print trans
    # Read the image, transform to black and white and comvert to numpy array
    img = img1.convert('L')
    img = img_to_array(img)

    # Apply affine transformation
    matrix = trans[:2, :2]
    offset = trans[:2, 2]
    img = img.reshape(img.shape[:-1])
    img = affine_transform(img, matrix, offset, output_shape=img_shape[:-1], order=1, mode='constant',
                           cval=np.average(img))
    img = img.reshape(img_shape)

#     img=img1.crop([x0,y0,x1,y1])
    return img

def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    """
    Build a transformation matrix with the specified characteristics.
    """
    rotation = np.deg2rad(rotation)
    shear = np.deg2rad(shear)
    rotation_matrix = np.array(
        [[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, height_shift], [0, 1, width_shift], [0, 0, 1]])
    shear_matrix = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    zoom_matrix = np.array([[1.0 / height_zoom, 0, 0], [0, 1.0 / width_zoom, 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
    return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))
def show_whale(imgs, per_row=2):
    n = len(imgs)
    rows = (n + per_row - 1) // per_row
    cols = min(per_row, n)
    fig, axes = plt.subplots(rows, cols, figsize=(24 // per_row * cols, 24 // per_row * rows))
    for ax in axes.flatten(): ax.axis('off')
    for i, (img, ax) in enumerate(zip(imgs, axes.flatten())): ax.imshow(img.convert('RGB'))
        

def read_raw_image(image_path):
    img = Image.open(image_path)
    return img

def read_crop_image2(image_name):
    image_path=get_path(image_name)
    img1=image.load_img(image_path)
    img3=img1
    img_arr = image.img_to_array(img1)
    size_x=img_arr.shape[0]
    size_y=img_arr.shape[1]

    # Determine the region of the original image we want to capture based on the bounding box.
    row = p2bb.loc[p]
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
    if True:
        trans = np.dot(build_transform(
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(0.8, 1.0),
            random.uniform(0.8, 1.0),
            random.uniform(-0.05 * (y1 - y0), 0.05 * (y1 - y0)),
            random.uniform(-0.05 * (x1 - x0), 0.05 * (x1 - x0))
        ), trans)
    trans = np.dot(np.array([[1, 0, 0.5 * (y1 + y0)], [0, 1, 0.5 * (x1 + x0)], [0, 0, 1]]), trans)

    # Read the image, transform to black and white and comvert to numpy array
    img = read_raw_image(image_path).convert('L')
    img1=img
    img = img_to_array(img)

    # Apply affine transformation
    matrix = trans[:2, :2]
    offset = trans[:2, 2]
    img = img.reshape(img.shape[:-1])
    img = affine_transform(img, matrix, offset, output_shape=img_shape[:-1], order=1, mode='constant',
                           cval=np.average(img))
    img = img.reshape(img_shape)
    img2=img
    
    show_whale([img3])
#     plt.imshow(img)
#     plt.show()

    return img

def process_train_test(src_dir, dst_dir, image_name_file, only_single_image):
    with open(image_name_file) as f:
        train_image_list=f.readline().split(' ')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for image_name in os.listdir(src_dir):
        if image_name not in train_image_list:
            continue
        class_name=tagged.get(image_name)
        if class_name == 'new_whale':
            continue
        class_path=os.path.join(dst_dir,class_name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)
        img=read_crop_image(image_name)
        image.save_img(os.path.join(class_path, image_name),img)
        if only_single_image:
            continue
        
        angles=np.linspace(-3,3,10)
        for i in range(10):
            angle=angles[i]
            imgi=transform.rotate(image.img_to_array(img), angle)
            misc.imsave(os.path.join(class_path, image_name+"."+str(i)+".jpg"),imgi)

def process_test(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for image_name in os.listdir(src_dir):
        class_name=tagged.get(image_name)
        if class_name == 'new_whale':
            continue
        img=read_crop_image(image_name)
        image.save_img(os.path.join(dst_dir, image_name),img)

def main():
    
#     process_train_test(base_dir+'siamese_agument_train/', TRAIN, train_image_name_file, False)
#     process_test(TRAIN, base_dir+'siamese_agument/')
#     process_test(TEST,base_dir+'siamese_submit_test')
    img=read_crop_image('0a8e58929.jpg')
#     img_arr=img_to_array(img)
    print img.shape
    img=img.astype('uint8')
    print img[:,:,0]
    plt.imshow(img[:,:,0])
    plt.show()

main()








