#coding=utf-8
from __future__ import division
import os
from tensorflow.python.platform import gfile
import tensorflow as tf
import re
from tensorflow.python.ops import data_flow_ops
from scipy import misc
import numpy as np
from os.path import join
import csv
import argparse
import sys
import copy
# import pdb

# image_size_str='160,160'
# model='/home/nemo/models/kaggle/20190117-174225'
base_dir='/home/nemo/kaggle/'
all_path_dir=base_dir+'data/clean_train_160/'
test_path_dir=base_dir+'data/clean_train_160_test/'
test_dir=base_dir+'data/clean_test_160/'
use_flipped_images=False
use_fixed_image_standardization=False
# batch_size=100

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)
    
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
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
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

def get_control_flag(control, field):
    return tf.equal(tf.mod(tf.floor_div(control, field), 2), 1)

def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic')

RANDOM_ROTATE = 1
RANDOM_CROP = 2
RANDOM_FLIP = 4
FIXED_STANDARDIZATION = 8
FLIP = 16
def create_input_pipeline(input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder):
    images_and_labels_list = []
    for _ in range(nrof_preprocess_threads):
        filenames, label, control = input_queue.dequeue()
        images = []
        for filename in tf.unstack(filenames):
            file_contents = tf.read_file(filename)
            image = tf.image.decode_image(file_contents, 3)
            image = tf.cond(get_control_flag(control[0], RANDOM_ROTATE),
                            lambda:tf.py_func(random_rotate_image, [image], tf.uint8), 
                            lambda:tf.identity(image))
            image = tf.cond(get_control_flag(control[0], RANDOM_CROP), 
                            lambda:tf.random_crop(image, image_size + (3,)), 
                            lambda:tf.image.resize_image_with_crop_or_pad(image, image_size[0], image_size[1]))
            image = tf.cond(get_control_flag(control[0], RANDOM_FLIP),
                            lambda:tf.image.random_flip_left_right(image),
                            lambda:tf.identity(image))
            image = tf.cond(get_control_flag(control[0], FIXED_STANDARDIZATION),
                            lambda:(tf.cast(image, tf.float32) - 127.5)/128.0,
                            lambda:tf.image.per_image_standardization(image))
            image = tf.cond(get_control_flag(control[0], FLIP),
                            lambda:tf.image.flip_left_right(image),
                            lambda:tf.identity(image))
            #pylint: disable=no-member
            image.set_shape(image_size + (3,))
            images.append(image)
        images_and_labels_list.append([images, label])

    image_batch, label_batch = tf.train.batch_join(
        images_and_labels_list, batch_size=batch_size_placeholder, 
        shapes=[image_size + (3,), ()], enqueue_many=True,
        capacity=4 * nrof_preprocess_threads * 100,
        allow_smaller_final_batch=True)
    
    return image_batch, label_batch

def get_data_set(path_dir):
    classes = os.listdir(path_dir)
    classes.sort()
    class_num = len(classes)
    dataset=[]
    for i in range(class_num):
        class_name=classes[i]
        image_paths=[join(path_dir,class_name,img) for img in os.listdir(join(path_dir,class_name))]
        dataset.append(ImageClass(class_name, image_paths))
    return dataset

def get_class_dict(dataset):
    dict={}
    for i, data in enumerate(dataset):
        dict[i]=data.name
    return dict

def get_image_paths_and_labels_for_eval(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat.append(dataset[i].image_paths[0])
        labels_flat += [dataset[i].name]
    return image_paths_flat, labels_flat
    

def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    batch_size=100
    if args.submit:
        batch_size=40
    with tf.Graph().as_default():
        with tf.Session() as sess:
            image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
            labels_placeholder = tf.placeholder(tf.int64, shape=(None,1), name='labels')
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
            
            nrof_preprocess_threads = 4
            image_size1=args.image_size.split(',')
            image_size = (int(image_size1[0]), int(image_size1[1]))
            eval_input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                        dtypes=[tf.string, tf.int64, tf.int32],
                                        shapes=[(1,), (1,), (1,)],
                                        shared_name=None, name=None)
            enqueue_op = eval_input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='eval_enqueue_op')
            image_batch, label_batch = create_input_pipeline(eval_input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder)
             
            # Load the model
            input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
            load_model(args.model_dir, input_map=input_map)
        
            # Get output tensor
            logits = tf.get_default_graph().get_tensor_by_name("finallogits:0")
                    
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)
            
            image_labels=[]
            if args.submit:
                image_paths=[test_dir+name for name in os.listdir(test_dir)]
                image_paths.sort()
            else:
                dataset=get_data_set(test_path_dir)
                image_paths, image_labels=get_image_paths_and_labels_for_eval(dataset)

            evaluate(args, sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder,
        logits, label_batch, image_paths, image_labels, batch_size, use_flipped_images, use_fixed_image_standardization)
                
def evaluate(args, sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder,
        logits, label_batch, image_paths, image_labels, batch_size, use_flipped_images, use_fixed_image_standardization):

    nrof_embeddings = len(image_paths)
    nrof_flips = 2 if use_flipped_images else 1
    nrof_images = nrof_embeddings * nrof_flips
    labels_array = np.expand_dims(np.arange(0,nrof_images),1)
    image_paths_array = np.expand_dims(np.repeat(np.array(image_paths),nrof_flips),1)
    control_array = np.zeros_like(labels_array, np.int32)
    if use_fixed_image_standardization:
        control_array += np.ones_like(labels_array)*FIXED_STANDARDIZATION
    if use_flipped_images:
        control_array += (labels_array % 2)*FLIP

    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})
    
    logits_size = int(logits.get_shape()[1])
    nrof_batches = nrof_images // batch_size
#     if nrof_images % batch_size != 0:
#         nrof_batches=nrof_batches+1
    emb_array = np.zeros((nrof_images, logits_size))
    lab_array = np.zeros((nrof_images,))
#     pdb.set_trace()
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size}
        emb, lab = sess.run([logits, label_batch], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab, :] = emb

    logits1 = np.zeros((nrof_embeddings, logits_size*nrof_flips))
    if use_flipped_images:
        # Concatenate embeddings for flipped and non flipped version of the images
        logits1[:,:logits_size] = emb_array[0::2,:]
        logits1[:,logits_size:] = emb_array[1::2,:]
    else:
        logits1 = emb_array

#     assert np.array_equal(lab_array, np.arange(nrof_images))==True
    
    dataset=get_data_set(all_path_dir)
    class_dict=get_class_dict(dataset)
    count=0
    if args.submit:
        with open('/home/nemo/kaggle/submission.csv','w') as f:
            writer = csv.writer(f,dialect='excel')
            writer.writerow(['Image','Id'])
            for i in range(nrof_embeddings):
                logit=logits1[i]
                predict_class=np.argmax(logit)
                predict_class_name=class_dict.get(predict_class)
                image_name=os.path.basename(image_paths[i])
                
                print "class=%s,predict_class=%s, class_name=%s, predict_class_name=%s" %(i, predict_class, image_name,predict_class_name)
                logit_copy=copy.deepcopy(logit)
                logit_copy.sort()
                top_indexes=[np.argwhere(logit==value)[0][0] for value in logit_copy[::-1][:4]]
                top_classes=[class_dict.get(index) for index in top_indexes]
                print "predict_classes=%s    predict_class_name=%s" %(top_indexes, ' '.join(top_classes))
                
                writer.writerow([image_name,' '.join(top_classes)])
        return
    
    results=[]
    for i in range(nrof_embeddings):
        logit=logits1[i]
        predict_class=np.argmax(logit)
        predict_class_name=class_dict.get(predict_class)
        actual_class_name=image_labels[i]
        print "class=%s,predict_class=%s, class_name=%s = actual_class_name=%s, predict_class_name=%s" %(i, predict_class, os.path.basename(os.path.dirname(image_paths[i])),actual_class_name, predict_class_name)
        succeed=False
        if actual_class_name == predict_class_name:
            count=count+1
            succeed=True
        print succeed, logit[predict_class]
#         results.append([succeed,i,])
        
    print "count=%d, accuracy=%f" %(count, count/nrof_embeddings)
        
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', type=bool,
                        help='Submit kaggle', default=False)
    parser.add_argument('--model_dir', type=str,
                        help='model_dir', default='/home/nemo/models/kaggle/20190117-174225')
    parser.add_argument('--image_size', type=str,
                        help='image size', default='160,160')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    
    