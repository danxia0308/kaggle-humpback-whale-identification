import os
from os.path import join
import shutil

train_image_path='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/train/'
train_csv='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/train.csv'
train_base_dir="/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/train_new/"


def get_train_pair_dict():
    train_dict={}
    with open(train_csv) as train_csv_file:
        lines = train_csv_file.readlines()
        # Do not take the first line.
        lines=lines[1:]
    for line in lines:
        line=line[:-1]
        values = line.split(',')
        train_dict[values[0]]=values[1]
    return train_dict

def put_data_under_class():
    train_dict = get_train_pair_dict()
    for file_name in os.listdir(train_image_path):
        class_name = train_dict.get(file_name)
        if class_name:
            dir_path=join(train_base_dir,class_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            shutil.copyfile(join(train_image_path,file_name), join(dir_path,file_name))
        else:
            print file_name, "has no class."

put_data_under_class()