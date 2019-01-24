from os import path
import os
from scipy import misc
from Crypto.SelfTest.Cipher.test_CFB import file_name

def get_image_with_specified_shape():
    src_path='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/test/'
    dst_path='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/test_temp/'
    if not path.exists(dst_path):
        os.makedirs(dst_path)
    for file_name in os.listdir(src_path):
        if file_name == '.DS_Store':
            continue
        file_path=path.join(src_path,file_name)
        img=misc.imread(file_path)
        print img.shape
        if img.shape[1] == 1050 and img.shape[0] < 500:
            cmd='mv '+file_path+' '+dst_path
            print cmd
            os.system(cmd)

def delete_files():
    path1='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/test/'
    path2='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/test_temp1/'
    path_dst='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/test_temp2/'
    files2=os.listdir(path2)
    for file_name in os.listdir(path1):
        if file_name in files2:
            cmd='mv '+path1+file_name+" "+path_dst
            print cmd
            os.system(cmd)
            
def clean_files():
    path1='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/train/'
    path2='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/aa_train_good/'
    files2=os.listdir(path2)
    count=0
    for file_name in os.listdir(path1):
        if file_name in files2:
            cmd='rm -fr '+path1+file_name
            print cmd
            os.system(cmd)
            count=count+1
    print count

def copy_files():
    path1='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/train_backup/'
    path2='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/aa_train_good/'
    path_dst='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/train_temp1/'
    files2=os.listdir(path2)
    count=0
    for file_name in os.listdir(path1):
        if file_name not in files2:
            cmd='cp '+path1+file_name+" "+path_dst
            print cmd
            os.system(cmd)
            count=count+1
    print count
    

copy_files()