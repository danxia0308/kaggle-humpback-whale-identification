import os
import shutil
from pandas import read_csv

src_dir='/home/nemo/kaggle/data/clean_train_agument_422/'
train_dir='/home/nemo/kaggle/data/clean_train_agument_422_train/'
test_dir='/home/nemo/kaggle/data/clean_train_agument_422_test/'
threshold=5
def split_1():
    for dir_name in os.listdir(src_dir):
        sub_dir_path=os.path.join(src_dir,dir_name)
        files=os.listdir(sub_dir_path)
        train_sub_dir=os.path.join(train_dir,dir_name)
        test_sub_dir=os.path.join(test_dir,dir_name)
        if not os.path.exists(train_sub_dir):
            os.makedirs(train_sub_dir)
        if len(files) >= threshold:
            if not os.path.exists(test_sub_dir):
                os.makedirs(test_sub_dir)
            test_file=files[0]
            shutil.copyfile(os.path.join(sub_dir_path,test_file),os.path.join(test_sub_dir,test_file))
            for i in range(1,len(files)):
                index=files[i].find('.jpg')
                if files[i][:(index+4)] == test_file:
                    shutil.copyfile(os.path.join(sub_dir_path,files[i]),os.path.join(test_sub_dir,files[i]))
                else:
                    shutil.copyfile(os.path.join(sub_dir_path,files[i]), os.path.join(train_sub_dir,files[i]))
        else:
            for i in range(0, len(files)):
                shutil.copyfile(os.path.join(sub_dir_path,files[i]), os.path.join(train_sub_dir,files[i]))

test_image_name_file='/home/nemo/kaggle/data/test_list.txt'
with open(test_image_name_file) as f:
    test_image_names=f.readline().split(' ')

def split_2():
    for dir_name in os.listdir(src_dir):
        sub_dir_path=os.path.join(src_dir,dir_name)
        files=os.listdir(sub_dir_path)
        train_sub_dir=os.path.join(train_dir,dir_name)
        test_sub_dir=os.path.join(test_dir,dir_name)
        if not os.path.exists(train_sub_dir):
            os.makedirs(train_sub_dir)
        for file_name in files:
            if file_name in test_image_names:
                if not os.path.exists(test_sub_dir):
                    os.makedirs(test_sub_dir)
                shutil.copyfile(os.path.join(sub_dir_path,file_name),os.path.join(test_sub_dir,file_name))
            else:
                shutil.copyfile(os.path.join(sub_dir_path,file_name), os.path.join(train_sub_dir,file_name))

split_2()