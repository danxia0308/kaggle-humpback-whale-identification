import os
from os.path import join

test_dir='/home/nemo/kaggle/data/clean_train_384_test/'
train_dir='/home/nemo/kaggle/data/clean_train_384_train/'
test_name_list_file='/home/nemo/kaggle/data/test_list.txt'
train_name_list_file='/home/nemo/kaggle/data/train_list.txt'

def get_image_name_list(dir_path):
    name_list=[]
    for sub_dir in os.listdir(dir_path):
        sub_dir_path=join(dir_path,sub_dir)
        for file_name in os.listdir(sub_dir_path):
            name_list.append(file_name)
    return name_list

with open(test_name_list_file) as f:
    f.write(get_image_name_list(test_dir))

with open(train_name_list_file) as f:
    f.write(get_image_name_list(train_dir))