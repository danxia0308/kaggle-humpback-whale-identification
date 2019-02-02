import os
from os.path import join

image_dir='/home/nemo/kaggle/data/clean_train_422_train/'

for dir_name in os.listdir(image_dir):
    dir_path=join(image_dir,dir_name)
    for file_name in os.listdir(dir_path):
        names=file_name.split('.jpg')
        file_path=join(dir_path,file_name)
        if (len(names) == 3):
            command='rm -fr '+file_path
            print command
            os.system(command)
