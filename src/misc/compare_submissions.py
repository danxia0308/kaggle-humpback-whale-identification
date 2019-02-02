import os
from pandas import read_csv
from os.path import join
import shutil

base_dir='/Users/chendanxia/sophie/'
s1=base_dir+'submission_0.673.csv'
s2=base_dir+'submission.csv'
test_dir='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/clean_test'
dst_dir='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/analysis'
new_whale_name='new_whale'


def compare():
    i2c1=dict([(i,c) for _, i, c in read_csv(s1).to_records()])
    i2c2=dict([(i,c) for _, i, c in read_csv(s2).to_records()])
    num=2
    diff_arr=[]
    for image_name in i2c1.keys():
        c1=i2c1.get(image_name).split(' ')
        c2=i2c2.get(image_name).split(' ')
        if c1 != c2:
            for v1 in c1[:num]:
                if v1 not in c2[:num]:
                    print image_name,'| ', c1,' | ', c2
                    diff_arr.append([image_name, c1, c2])
    
    print 'diff num:',len(diff_arr)

def get_class(class_names):
    cns=class_names.split(' ')
    for cn in cns:
        return cn
#         if cn != new_whale_name:
#             return cn
            

def result_analysis(submission_file):
    i2c=dict([(i,c) for _, i, c in read_csv(submission_file).to_records()])
    c2i={}
    for key in i2c.keys():
        class_name=get_class(i2c.get(key))
        if class_name not in c2i.keys():
            c2i[class_name]=[key]
        else:
            c2i[class_name].append(key)
    for class_name in c2i.keys():
        class_dir=join(dst_dir,class_name)
        if len(c2i.get(class_name)) < 5:
            continue
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        for file_name in c2i.get(class_name):
            shutil.copyfile(join(test_dir,file_name), join(join(class_dir,file_name)))
            
# result_analysis(s1)            
compare()
            
            
            
            
        