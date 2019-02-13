# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
from datetime import datetime
import os
from os.path import join
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from scipy import misc
import numpy as np
import tensorflow.contrib.slim as slim
import math
import time
import sys
import argparse
from skimage import transform
import pdb
from PIL import Image, ImageEnhance
from keras.preprocessing.image import img_to_array
from pandas import read_csv
from scipy.ndimage import affine_transform
import util


# model_def='models.inception_resnet_v1'
logs_base_dir = '/home/nemo/logs/kaggle'
models_base_dir = '/home/nemo/models/kaggle'
train_csv_path='/home/nemo/kaggle/data/train.csv'
test_image_name_file='/home/nemo/kaggle/data/test_list.txt'

# batch_size=90
test_batch_size=62
epoch_size=1000
random_shear_zoom=True
random_rotate=True
random_crop=True
random_flip=True
keep_probability=0.6
embedding_size=512
weight_decay=5e-4
learning_rate_decay_epochs=100
learning_rate_schedule_file='data/learning_rate_schedule_classifier_vggface2.txt'
optimizer='ADAGRAD'
gpu_memory_fraction=0.9
max_nrof_epochs=600
prelogits_hist_max=10.0
use_fixed_image_standardization=True
use_flipped_images=False

ignore_new_whale=True

new_whale_name='new_whale'


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

test_image_name_list=get_image_name_list(test_image_name_file)

def get_class_name_2_image_names(forAll=False):
    class_name_2_image_names={}
    for image_name in image_name_2_class_name.keys():
        class_name=image_name_2_class_name.get(image_name)
        if class_name==new_whale_name:
            continue
        if image_name in test_image_name_list:
            if not forAll:
                continue
        if class_name not in class_name_2_image_names:
            image_names=[]
        else:
            image_names=class_name_2_image_names.get(class_name)
        if image_name not in image_names:
            image_names.append(image_name)
        class_name_2_image_names[class_name]=image_names
    return class_name_2_image_names

class_name_2_image_names_train=get_class_name_2_image_names(False)
class_name_2_image_names_all=get_class_name_2_image_names(True)





class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)

def get_data_set_train(path_dir, iggore_single_image, useAll=False):
    if useAll:
        class_name_2_image_names=class_name_2_image_names_all
    else:
        class_name_2_image_names=class_name_2_image_names_train
    classes = class_name_2_image_names.keys()
    classes.sort()
    class_num = len(classes)
    dataset=[]
    for i in range(class_num):
        class_name=classes[i]
        if ignore_new_whale and class_name==new_whale_name:
            continue
        image_paths=[join(path_dir,img) for img in class_name_2_image_names.get(class_name)]
        if iggore_single_image:
            if len(image_paths)==1:
                continue
        dataset.append(ImageClass(class_name, image_paths))
    return dataset

def get_data_set(path_dir, iggore_single_image):
    classes = os.listdir(path_dir)
    classes.sort()
    class_num = len(classes)
    dataset=[]
    for i in range(class_num):
        class_name=classes[i]
        if ignore_new_whale and class_name==new_whale_name:
            continue
        image_paths=[join(path_dir,class_name,img) for img in os.listdir(join(path_dir,class_name))]
        if iggore_single_image:
            if len(image_paths)==1:
                continue
        dataset.append(ImageClass(class_name, image_paths))
    return dataset

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat

def get_image_paths_and_labels_for_evaluate(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [dataset[i].name] * len(dataset[i].image_paths) 
    return image_paths_flat, labels_flat

def _add_loss_summaries(total_loss):
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
  
    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
  
    return loss_averages_op

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
  
def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer=='ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer=='ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer=='ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer=='RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer=='MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')
    
        grads = opt.compute_gradients(total_loss, update_gradient_vars)
        
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  
    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
   
    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
  
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
  
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
  
    return train_op

def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                if par[1]=='-':
                    lr = -1
                else:
                    lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate
RANDOM_ROTATE = 1
RANDOM_CROP = 2
RANDOM_FLIP = 4
FIXED_STANDARDIZATION = 8
FLIP = 16
               
def do_train(args, sess, epoch, learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, step, 
      loss, train_op, summary_op, summary_writer, reg_losses, learning_rate_schedule_file, cross_entropy_mean, accuracy, learning_rate):
    batch_number = 0
#     pdb.set_trace()
    lr = get_learning_rate_from_file(learning_rate_schedule_file, epoch)
        
    if lr<=0:
        return False 

    train_time = 0
    while batch_number < epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder:True, batch_size_placeholder:args.batch_size}
        tensor_list = [loss, train_op, step, reg_losses, cross_entropy_mean, learning_rate, accuracy]
        if batch_number % 100 == 0:
            loss_, _, step_, reg_losses_, cross_entropy_mean_, lr_, accuracy_, summary_str = sess.run(tensor_list + [summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step_)
        else:
            loss_, _, step_, reg_losses_, cross_entropy_mean_, lr_, accuracy_ = sess.run(tensor_list, feed_dict=feed_dict)
         
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tXent %2.3f\tRegLoss %2.3f\tAccuracy %2.3f\tLr %2.5f' %
              (epoch, batch_number+1, epoch_size, duration, loss_, cross_entropy_mean_, np.sum(reg_losses_), accuracy_, lr_))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, global_step=step_)
    return True

def get_class_dict(dataset):
    dict={}
    for i, data in enumerate(dataset):
        dict[i]=data.name
    return dict

def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, 
        logits, labels, image_paths, eval_labels, batch_size,  step, summary_writer, use_flipped_images, use_fixed_image_standardization, dataset):
   
    start_time = time.time()
    print('Runnning forward pass on validate images')
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
    
    embedding_size = int(logits.get_shape()[1])
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size}
        emb, lab = sess.run([logits, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab, :] = emb
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')
    embeddings = np.zeros((nrof_embeddings, embedding_size*nrof_flips))
    if use_flipped_images:
        embeddings[:,:embedding_size] = emb_array[0::2,:]
        embeddings[:,embedding_size:] = emb_array[1::2,:]
    else:
        embeddings = emb_array
    
    class_dict=get_class_dict(dataset)
    count=0
    for i in range(nrof_embeddings):
        logit=embeddings[i]
        predict_class=np.argmax(logit)
        predict_class_name=class_dict.get(predict_class)
        print ("predict_class=%s, class_name=%s, predict_class_name=%s" %(predict_class, eval_labels[i],predict_class_name))
        if eval_labels[i] == predict_class_name:
            count=count+1
    accuracy=count/nrof_embeddings
    print ("count=%d, accuracy=%f" %(count, accuracy))
    assert np.array_equal(lab_array, np.arange(nrof_images))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'

    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    
     
def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret  

def rotate_and_adjcolor(img):     
    angle=np.random.uniform(low=-10.0, high=10.0)
    ro_img = transform.rotate(img,angle)
    return ro_img
#     alpha=np.random.uniform(0.5,1.5)
#     beta=0
#     img2 = np.add(np.multiply(ro_img, alpha),beta)
#     img2 = np.where(img2 >255,255,img2)
#     img2 = np.where(img2 < 0, 0,img2)
#     return img2

def preprocess_function(image_path, label, args):
    file_contents = tf.read_file(image_path)
    img = tf.image.decode_image(file_contents, channels=3)
#     img = tf.image.random_flip_left_right(img)
    img=tf.cast(img,tf.float32)
    img = tf.py_func(util.read_cropped_image,[img,image_path],tf.float32)
    img.set_shape((args.image_size,args.image_size, 3))
    img = tf.image.per_image_standardization(img)
#     img = (tf.cast(img, tf.float32) - 127.5)/128.0
    
    return img, label

def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    # import network
    network = importlib.import_module(args.model_def)
    
    # create logs and models dir.
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = join(os.path.expanduser(logs_base_dir), subdir)
    if not os.path.isdir(log_dir): 
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(models_base_dir), subdir)
    if not os.path.isdir(model_dir): 
        os.makedirs(model_dir)

    train_set = get_data_set_train(args.data_dir,False,args.use_all)  
#     test_set = get_data_set(args.test_dir,False)
    class_num = len(train_set) 
    
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        print('Pre-trained model: %s' % pretrained_model)
    
    with tf.Graph().as_default():
#         pdb.set_trace()
        global_step = tf.Variable(0, trainable=False)
        image_list, label_list = get_image_paths_and_labels(train_set)
        images_path_tensor = tf.convert_to_tensor(image_list)
        labels_tensor = tf.convert_to_tensor(label_list, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((images_path_tensor, labels_tensor)).shuffle(len(image_list))
        dataset = dataset.map(lambda image_path, label: preprocess_function(image_path, label,args)) 
        dataset = dataset.batch(args.batch_size) 
        iterator = dataset.make_initializable_iterator() 
        dataset.repeat()
        images, labels = iterator.get_next()
        
        images = tf.identity(images, 'image_batch')
        images = tf.identity(images, 'input')
        labels = tf.identity(labels, 'label_batch')
        
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        
        print('Total number of classes: %d' % class_num)
        print('Total number of examples: %d' % len(image_list))
        print('Building training graph')
        
        # Build the inference graph
        prelogits, _ = network.inference(images, keep_probability, 
            phase_train=phase_train_placeholder, bottleneck_layer_size=embedding_size, 
            weight_decay=weight_decay)
        logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None, 
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                weights_regularizer=slim.l2_regularizer(weight_decay),
                scope='Logits', reuse=False)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        
        # Norm for the prelogits
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            learning_rate_decay_epochs*epoch_size, 1.0, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        
        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        
        eps = 1e-4
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits)+eps, ord=1.0, axis=1))
        
        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = train(total_loss, global_step, optimizer, 
            learning_rate, 0.9999, tf.global_variables(), True)
        
        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        with sess.as_default():
            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                saver.restore(sess, pretrained_model)
            # Training and validation loop
            print('Running training')
            sess.run(iterator.initializer)
            for epoch in range(1,max_nrof_epochs+1):
#                 step = sess.run(global_step, feed_dict=None)
                try:
                    cont = do_train(args, sess, epoch, learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step, 
                        total_loss, train_op, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file,
                        cross_entropy_mean, accuracy, learning_rate)
                    if not cont:
                        break
                except tf.errors.OutOfRangeError:
                    print ('Out of range.')
                    sess.run(iterator.initializer)
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, epoch)
                
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int,
                        help='Image Size', default=384)
    parser.add_argument('--data_dir', type=str,
                        help='data dir', default='/home/nemo/kaggle/data/train/')
    parser.add_argument('--gpu_num', type=str,
                        help='select which gpu', default='0')
    parser.add_argument('--batch_size', type=int,
                        help='batch size', default=90)
    parser.add_argument('--use_all', type=bool,
                        help='use all', default=False)
    parser.add_argument('--model_def', type=str,
                        help='model_def', default='models.inception_resnet_v1')
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--lr_file', type=str,
        help='Load a pretrained model before training starts.',default='data/learning_rate_schedule_classifier_vggface2.txt')

    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    
