import os

test_dir='/home/nemo/kaggle/data/clean_train_384_test/'
train_dir='/home/nemo/kaggle/data/clean_train_384_train/'
test_name_list_file='/home/nemo/kaggle/data/test_list.txt'
train_name_list_file='/home/nemo/kaggle/data/train_list.txt'

def get_image_name_list(dir_path):
    name_list=[]
    for sub_dir in os.listdir(dir_path):
        sub_dir_path=os.path.join(dir_path,sub_dir)
        for file_name in os.listdir(sub_dir_path):
            name_list.append(file_name)
    return name_list

def store_file(name_list_file, file_dir):
    with open(name_list_file,'w') as f:
        f.write(' '.join(get_image_name_list(file_dir)))

store_file(test_name_list_file, test_dir)
store_file(train_name_list_file, train_dir)