from pandas import read_csv
from skimage import transform
import os
from os.path import join
import shutil
from scipy import misc
import numpy as np

base_dir='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/'
train_csv = base_dir+'train.csv'
src_train_dir = base_dir+'clean_train/'
src_test_dir = base_dir+'clean_test/'
dst_train_dir = base_dir+'clean_train_agument/'
dst_test_dir = base_dir+'clean_test_agument/'
new_whale_name='new_whale'

i2c=dict([(i, c) for _, i, c in read_csv(train_csv).to_records()])
angles=np.linspace(-5,5,10)

for file_name in os.listdir(src_train_dir):
    if file_name == '.DS_Store':
        continue
    file_path=join(src_train_dir,file_name)
    class_name=i2c.get(file_name)
    if class_name == new_whale_name:
        continue
    dst_sub_dir=join(dst_train_dir,class_name)
    if not os.path.exists(dst_sub_dir):
        os.makedirs(dst_sub_dir)
    img=misc.imread(file_path)
    misc.imsave(join(dst_sub_dir,file_name),img)
    
    for i in range(len(angles)):
        ro_img=transform.rotate(img,angles[i])
        misc.imsave(join(dst_sub_dir,file_name+"."+str(i)+'.jpg'),ro_img)
        
        
        
        
        