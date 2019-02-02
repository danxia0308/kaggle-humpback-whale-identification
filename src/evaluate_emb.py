from __future__ import division
import os
from pandas import read_csv
from tensorflow.python.platform import gfile
import tensorflow as tf
import re
from tensorflow.python.ops import data_flow_ops
from scipy import misc
import numpy as np
import argparse
import sys
import copy
import time
import csv
import pdb
from os.path import join

train_image_name_file='/home/nemo/kaggle/data/train_list.txt'
test_image_name_file='/home/nemo/kaggle/data/test_list.txt'
submit_test_image_dir='/home/nemo/kaggle/data/clean_test_384/'
train_csv_path='/home/nemo/kaggle/data/train.csv'
image_dir='/home/nemo/kaggle/data/clean_train_384/'
embedding_size=512

compare_every_whale=False
ignore_new_whale=True
image_name_2_class_name=dict([(y,z) for x,y,z in read_csv(train_csv_path).to_records()])
new_whale_name='new_whale'
def get_image_name_list(file_path):
    with open(file_path) as f:
        whales = f.read().split(" ")
        if ignore_new_whale:
            clean_whale=[]
            for whale in whales:
                if image_name_2_class_name[whale]!=new_whale_name:
                    clean_whale.append(whale)
            return clean_whale
        else:
            return whales

train_image_name_list=get_image_name_list(train_image_name_file)
test_image_name_list=get_image_name_list(test_image_name_file)
train_image_num=len(train_image_name_list)
test_image_num=len(test_image_name_list)
train_image_path_list=[join(image_dir,image_name) for image_name in train_image_name_list]
test_image_path_list=[join(image_dir,image_name) for image_name in test_image_name_list]
submit_test_image_name_list=os.listdir(submit_test_image_dir)
submit_test_image_name_list.sort()
submit_test_image_path_list=[submit_test_image_dir+image_name for image_name in submit_test_image_name_list]
submit_test_image_num=len(submit_test_image_name_list)


class_name_2_image_names_all={}
for image_name in image_name_2_class_name.keys():
    class_name=image_name_2_class_name.get(image_name)
    if ignore_new_whale and class_name==new_whale_name:
        continue
    if class_name not in class_name_2_image_names_all:
        image_names=[]
    else:
        image_names=class_name_2_image_names_all.get(class_name)
    if image_name not in image_names:
        image_names.append(image_name)
    class_name_2_image_names_all[class_name]=image_names

class_name_2_image_names_train={}
for image_name in image_name_2_class_name.keys():
    class_name=image_name_2_class_name.get(image_name)
    if ignore_new_whale and class_name==new_whale_name:
        continue
    if image_name not in train_image_name_list:
        continue
    if class_name not in class_name_2_image_names_train:
        image_names=[]
    else:
        image_names=class_name_2_image_names_train.get(class_name)
    if image_name not in image_names:
        image_names.append(image_name)
    class_name_2_image_names_train[class_name]=image_names

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
#             images.append((tf.cast(image, tf.float32) - 127.5)/128.0)
            images.append(tf.image.per_image_standardization(image))
            
            
        images_and_labels_list.append([images, label])

    image_batch, label_batch = tf.train.batch_join(
        images_and_labels_list, batch_size=batch_size_placeholder, 
        shapes=[image_size + (3,), ()], enqueue_many=True,
        capacity=4 * nrof_preprocess_threads * 100,
        allow_smaller_final_batch=True)
    
    return image_batch, label_batch

def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
#     batch_size=90
#     if args.submit:
#         batch_size=20
    with tf.Graph().as_default():
        with tf.Session() as sess:
            image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
            labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
            
            nrof_preprocess_threads = 4
            image_size1=args.image_size.split(',')
            image_size = (int(image_size1[0]), int(image_size1[1]))
            eval_input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                        dtypes=[tf.string, tf.int32, tf.int32],
                                        shapes=[(1,), (1,), (1,)],
                                        shared_name=None, name=None)
            enqueue_op = eval_input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='eval_enqueue_op')
            image_batch, label_batch = create_input_pipeline(eval_input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder)
             
            # Load the model
            input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
            load_model(args.model_dir, input_map=input_map)
        
            # Get output tensor
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)
            
            if args.submit:
                all_image_paths=train_image_path_list+submit_test_image_path_list
            else:
                all_image_paths=train_image_path_list+test_image_path_list
                
            evaluate(args, sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder,
        embeddings, label_batch, all_image_paths, args.batch_size, args.use_flipped_images, False)

def distance(emb_arr1, emb_arr2):
#     pdb.set_trace()
    diff = np.subtract(emb_arr1, emb_arr2)
    dist = np.sum(np.square(diff))
    return dist

def distances(embs1, embs2):
    diffs = np.subtract(embs1,embs2)
    dists = np.sum(np.square(diffs), 1)
    return dists

def mean_emb_per_class(embeddings):
    class_v_mean_emb={}
    for class_name in class_name_2_image_names_train:
        image_names=class_name_2_image_names_train.get(class_name)
        embedding_size = embeddings.shape[1]
        embedding_sum=[0]*embedding_size
        for image_name in image_names:
            index=train_image_name_list.index(image_name)
            embedding=embeddings[index]
            embedding_sum=np.add(embedding_sum,embedding)
        embedding_mean=np.divide(embedding_sum,len(image_names))
        class_v_mean_emb[class_name]=embedding_mean
    return class_v_mean_emb

def test_v_trains_distances(train_embeddings, test_embedding):
    return distances(train_embeddings, test_embedding)
#     distances=[]
#     for train_embedding in train_embeddings:
#         dist=distance(train_embedding, test_embedding)
#         distances.append(dist)
#     return distances

def find_top_indexes_k(arr, k):
    arr_copy=copy.deepcopy(arr)
    arr_copy.sort()
    top_indexes=[np.argwhere(arr==value)[0][0] for value in arr_copy[:k]]
    return top_indexes
                
def evaluate(args, sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder,
        embeddings, label_batch, image_paths, batch_size, use_flipped_images, use_fixed_image_standardization):
    start_time=time.time()
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
    
    nrof_batches = nrof_images // batch_size
    embedding_size = int(embeddings.get_shape()[1])
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    
    print 'nrof_batches',nrof_batches
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size}
        emb, lab = sess.run([embeddings, label_batch], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab, :] = emb
    
    left_image_num=nrof_images % batch_size
    feed_dict = {phase_train_placeholder:False, batch_size_placeholder:left_image_num}
    emb, lab = sess.run([embeddings, label_batch], feed_dict=feed_dict)
    lab_array[lab] = lab
    emb_array[lab, :] = emb
    
    embeddings = np.zeros((nrof_embeddings, embedding_size*nrof_flips))
    if use_flipped_images:
        embeddings[:,:embedding_size] = emb_array[0::2,:]
        embeddings[:,embedding_size:] = emb_array[1::2,:]
    else:
        embeddings = emb_array
    find_retrive_emb_time=time.time()
    print 'retrive all embeddings cost time', find_retrive_emb_time-start_time
    
    if not compare_every_whale:
        train_mean_embeddings=mean_emb_per_class(embeddings[:train_image_num])
    else:
        train_mean_embeddings=[]
        
    one_result_count=0
    top_5_result_count=0
    
    true_thresholds=[]
    false_thresholds=[]
    false_classes=[]
    
    if args.submit:
        with open('/home/nemo/kaggle/submission.csv','w') as f:
            writer = csv.writer(f,dialect='excel')
            writer.writerow(['Image','Id'])
            for i in range(train_image_num, nrof_images):
                test_embedding=embeddings[i]
                if compare_every_whale:
                    dists=test_v_trains_distances(embeddings[:train_image_num], test_embedding)
                    dists_copy=copy.deepcopy(dists)
                    dists_copy.sort()
                    top_classes=[]
                    sorted_indexes=[np.argwhere(dists==value)[0][0] for value in dists_copy]
                    predict_class_bottom_5_dist=[]
                    for j in range(len(sorted_indexes)):
                        index=sorted_indexes[j]
                        image_name=os.path.basename(image_paths[index])
                        class_name=image_name_2_class_name.get(image_name)
                        
                        if new_whale_name not in top_classes and dists[index] > args.threshold:
                            top_classes.append(new_whale_name)
                        if class_name not in top_classes:
                            top_classes.append(class_name)
                            predict_class_bottom_5_dist.append(dists[index])
                        if len(top_classes) >=5:
                            break
                    test_image_name=os.path.basename(image_paths[i])
                    writer.writerow([test_image_name,' '.join(top_classes[:5])])
                
                else:
                    dists=test_v_trains_distances(train_mean_embeddings.values(), test_embedding)
                    predict_index=np.argmin(dists)
                    predict_class=train_mean_embeddings.keys()[predict_index]
                    predict_class_bottom_5_indexes=find_top_indexes_k(dists, 5)
                    predict_class_bottom_5=[train_mean_embeddings.keys()[index] for index in predict_class_bottom_5_indexes]
                    predict_class_bottom_5_dist=[dists[index] for index in predict_class_bottom_5_indexes]
                    top_classes=[]
                    for j in range(len(predict_class_bottom_5_dist)):
                        if predict_class_bottom_5_dist[j] > args.threshold and 'new_whale' not in top_classes:
                            top_classes.append('new_whale')
                        if predict_class_bottom_5[j] not in top_classes:
                            top_classes.append(predict_class_bottom_5[j])
                        
                    test_image_name=os.path.basename(image_paths[i])
                    writer.writerow([test_image_name,' '.join(top_classes[:5])])
            end_time=time.time()
            print 'finish calculation cost',end_time-find_retrive_emb_time
            print 'all cost',end_time-start_time
        return
 
    
    for i in range(train_image_num, nrof_images):
        test_embedding=embeddings[i]
        if compare_every_whale:
            dists=test_v_trains_distances(embeddings[:train_image_num], test_embedding)
            dists_copy=copy.deepcopy(dists)
            dists_copy.sort()
            top_classes=[]
            sorted_indexes=[np.argwhere(dists==value)[0][0] for value in dists_copy]
            predict_class_bottom_5_dist=[]
            for j in range(len(sorted_indexes)):
                index=sorted_indexes[j]
                image_name=os.path.basename(image_paths[index])
                class_name=image_name_2_class_name.get(image_name)
                
                if new_whale_name not in top_classes and dists[index] > args.threshold:
                    top_classes.append(new_whale_name)
                if class_name not in top_classes:
                    top_classes.append(class_name)
                    predict_class_bottom_5_dist.append(dists[index])
                if len(top_classes) >=5:
                    break
            test_image_name=os.path.basename(image_paths[i])
            actual_class_name=image_name_2_class_name.get(test_image_name)
            print "test_image_name=%s, actual_class_name=%s,top_classes=%s, predict_file_dists=%s" %(test_image_name,actual_class_name,top_classes,predict_class_bottom_5_dist)
            if top_classes[0]==actual_class_name:
                one_result_count=one_result_count+1
                true_thresholds.append(dists[sorted_indexes[0]])
            if actual_class_name in top_classes:
                top_5_result_count=top_5_result_count+1
            else:
                false_thresholds.append(dists[sorted_indexes[0]])    
        else:
            dists=test_v_trains_distances(train_mean_embeddings.values(), test_embedding)
            predict_index=np.argmin(dists)
            predict_class=train_mean_embeddings.keys()[predict_index]
            predict_class_bottom_5_indexes=find_top_indexes_k(dists, 5)
            predict_class_bottom_5=[train_mean_embeddings.keys()[index] for index in predict_class_bottom_5_indexes]
            predict_class_bottom_5_dist=[dists[index] for index in predict_class_bottom_5_indexes]
            
            top_classes=[]
            for j in range(len(predict_class_bottom_5_dist)):
                if predict_class_bottom_5_dist[j] > args.threshold and 'new_whale' not in top_classes:
                    top_classes.append('new_whale')
                if predict_class_bottom_5[j] not in top_classes:
                    top_classes.append(predict_class_bottom_5[j])
                            
            test_image_name=os.path.basename(image_paths[i])
            actual_class_name=image_name_2_class_name.get(test_image_name)
#             print "test_image_name=%s, actual_class_name=%s,predict_class_name=%s,predict_five_class_names=%s,top_classes=%s, predict_file_dists=%s" %(test_image_name,actual_class_name,predict_class,predict_class_bottom_5,top_classes,predict_class_bottom_5_dist)
            if predict_class==actual_class_name:
                one_result_count=one_result_count+1
                true_thresholds.append(dists[predict_index])
            if actual_class_name in top_classes:
                top_5_result_count=top_5_result_count+1
            else:
                false_thresholds.append(dists[predict_index])
                false_classes.append([actual_class_name,predict_class,dists[predict_index]])
                print "test_image_name=%s, actual_class_name=%s,predict_class_name=%s,predict_five_class_names=%s,top_classes=%s, predict_file_dists=%s" %(test_image_name,actual_class_name,predict_class,predict_class_bottom_5,top_classes,predict_class_bottom_5_dist)
            
    true_thresholds.sort()
    false_thresholds.sort()
    print 'true_thresholds',true_thresholds
    print '\n\n'
    print 'false_thresholds',false_thresholds
    print '\n\n'
    print 'false_classes',false_classes  
    print 'one_result_count=%d, top_5_result_count=%d' %(one_result_count,top_5_result_count)
    print 'one_result_accuray=%f, top_5_result_accuacy=%f' %(one_result_count/test_image_num, top_5_result_count/test_image_num)
    end_time=time.time()
    print 'finish calculation cost',end_time-find_retrive_emb_time
    print 'all cost',end_time-start_time
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', type=bool,
                        help='Submit kaggle', default=False)
    parser.add_argument('--use_flipped_images', type=bool,
                        help='use flipped images', default=False)
    parser.add_argument('--threshold', type=float,
                        help='threshold', default=0.9)
    parser.add_argument('--model_dir', type=str,
                        help='model_dir', default='/home/nemo/models/kaggle/20190128-193543/')
    parser.add_argument('--image_size', type=str,
                        help='image size', default='384,384')
    parser.add_argument('--gpu_num', type=str,
                        help='select which gpu', default='0')
    parser.add_argument('--batch_size', type=int, default=90)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))