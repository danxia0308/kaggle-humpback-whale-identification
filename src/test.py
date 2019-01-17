#coding=utf-8
import os
import shutil
path="/Users/chendanxia/sophie/员工照片"
f='/Users/chendanxia/sophie/3.sh'
dest="/Users/chendanxia/sophie/ok/"
with open(f) as file:
    names=file.readlines()
    for name in names:
        name=name.replace("/Users/chendanxia/sophie/ok//","/")
        print name;
# with open(f) as file:
#     names=file.readlines()
#     names=[name[:-1] for name in names]
# print names;

# files=os.listdir(path)
# dict={}
# for file in files:
#     strs = file.split(".")
#     dict[strs[0]]=file
# print dict

# for name in names:
#     os.makedirs(dest+name)
#     p=os.path.join(path,name)
#     if not dict.get(p):
#         if not os.path.exists(os.path.join(dest,name)):
#             os.mkdir(os.path.join(dest,name))
#         print p, dict.get(p)
# #         shutil.copyfile(dict.get(p), os.path.join(dest,dict.get(p)))
#     else:
#         print name, "does not exist"