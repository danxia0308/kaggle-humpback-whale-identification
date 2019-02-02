import os
from scipy import misc
import tensorflow as tf
import re
from tensorflow.python.platform import gfile
import numpy as np
import argparse
import sys
from os.path import join
from pandas import read_csv

image_size = 384
base_dir='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/'
train_csv_path=base_dir+'train.csv'
image_name_2_class_name=dict([(y,z) for x,y,z in read_csv(train_csv_path).to_records()])
class_name_2_image_names_all={}
for image_name in image_name_2_class_name.keys():
    class_name=image_name_2_class_name.get(image_name)
    if class_name not in class_name_2_image_names_all:
        image_names=[]
    else:
        image_names=class_name_2_image_names_all.get(class_name)
    if image_name not in image_names:
        image_names.append(image_name)
    class_name_2_image_names_all[class_name]=image_names

def load_resize_image(image_paths):
    images=[]
    names=[]
    for image_path in image_paths:
        img=misc.imread(image_path)
        resized_img = misc.imresize(img,(image_size, image_size),interp='bilinear')
        resized_img = prewhiten(resized_img)
        images.append(resized_img)
        names.append(os.path.basename(image_path))
    return images, names

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def load_model(model, input_map=None):
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
      
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

def distances(emb, embs):
    sub = np.subtract(embs, emb)
    return np.sum(np.square(sub),1)
        
def main(args):
    
#     image_paths=[join(base_dir,'clean_train/ee46c9080.jpg'),join(base_dir,'analysis2/9fcd6e04d.jpg')]
    image_paths=[join(base_dir,'clean_train/3bd2b06cc.jpg'),join(base_dir,'analysis2/3bd2b06cc.jpg')]#,join(base_dir,'analysis2/2872f5a1a.jpg')]
    class_name='w_090c801'
    select_names = class_name_2_image_names_all.get(class_name)
    select_paths = [join(base_dir+'/clean_train/',name) for name in select_names]
    image_paths.extend(select_paths)
    images, names = load_resize_image(image_paths)
       
    with tf.Graph().as_default():
        with tf.Session() as sess:
            load_model(args.model) 
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            embs = sess.run(embeddings, feed_dict=feed_dict)

            for i, name in enumerate(names):
                print i, name
            for i, emb in enumerate(embs):
                print i, distances(emb, embs)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str,
                        help='model_dir', default='/Users/chendanxia/sophie/base_itnn_78_crop_95')
    parser.add_argument('--gpu_num', type=str,
                        help='select which gpu', default='0')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))



