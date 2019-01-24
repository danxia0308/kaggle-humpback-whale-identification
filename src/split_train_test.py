import os
import shutil
src_dir='/home/nemo/kaggle/data/clean_train_384/'
train_dir='/home/nemo/kaggle/data/clean_train_384_train/'
test_dir='/home/nemo/kaggle/data/clean_train_384_test/'
threshold=5

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
            shutil.copyfile(os.path.join(sub_dir_path,files[i]), os.path.join(train_sub_dir,files[i]))
    else:
        for i in range(0, len(files)):
            shutil.copyfile(os.path.join(sub_dir_path,files[i]), os.path.join(train_sub_dir,files[i]))

