from __future__ import division
import os
from pandas import read_csv
from scipy import misc
import numpy as np
import pickle

base_dir='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/'
train_csv_path=base_dir+'train.csv'
clean_train_dir=base_dir+'clean_train'
clean_test_dir=base_dir+'clean_test'
origion_train_dir=base_dir+'train_backup'
origion_test_dir=base_dir+'test_backup'
images_size_pickle=base_dir+'image_sizes.pickle'

# my_pickle=base_dir+"my.pickle"
# my_pickle_no_new=base_dir+"my1.pickle"

i2c=dict([(y,z) for x,y,z in read_csv(train_csv_path).to_records()])

# ratio_dict={}
# ratios=[]
keys=np.linspace(1,6,10)

def getRatios(dir_path):
    ratio_dict={}
    for name in os.listdir(dir_path):
        if name=='.DS_Store':
            continue
#         if new:
#             class_name=i2c.get(name)
#             if class_name=='new_whale':
#                 continue
        image_path=os.path.join(dir_path,name)
        img=misc.imread(image_path)
        shape=img.shape
        height=shape[0]
        width=shape[1]
        ratio=width/height
        ratio_dict[name]=ratio
        print ratio
    return ratio_dict

def store_ratios(f1):
    clean_train_ratios=getRatios(clean_train_dir)
    clean_test_ratios=getRatios(clean_test_dir)
    origion_train_ratios=getRatios(origion_train_dir)
    origion_test_ratios=getRatios(origion_test_dir)
    pickle.dump([clean_train_ratios,clean_test_ratios,origion_train_ratios,origion_test_ratios], open(f1,'wb'))

def get_ratios(f1):
    ratios_list=pickle.load(open(f1,'rb'))
    clean_train_ratios=ratios_list[0]
    clean_test_ratios=ratios_list[1]
    origion_train_ratios=ratios_list[2]
    origion_test_ratios=ratios_list[3]
    return clean_train_ratios,clean_test_ratios,origion_train_ratios,origion_test_ratios

# def print_ratio_analysis(f):
#     ratios=pickle.load(open(f,'rb'))
#     for ratio in ratios:
#         adj_keys=np.abs(keys-ratio)
#         index=np.argmin(adj_keys)
#         key=keys[index]
#         if key not in ratio_dict:
#             ratio_dict[key]=[ratio]
#         else:
#             ratio_dict[key].append(ratio)
# #     print ratio_dict
#     
#     for key in sorted(ratio_dict.keys()):
#         print 'key=%.2f' %(key), len(ratio_dict.get(key))

# store_ratios(images_size_pickle)
clean_train_ratios,clean_test_ratios,origion_train_ratios,origion_test_ratios=get_ratios(images_size_pickle)
print len(clean_train_ratios)
print len(origion_train_ratios)
print len(clean_test_ratios)

# print_ratio_analysis(my_pickle_no_new)
# print '\n'
# keys=np.linspace(5.3,6.3,50)
# print_ratio_analysis(my_pickle_no_new)





