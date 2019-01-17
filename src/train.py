from __future__ import absolute_import
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
import pdb


class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)

def get_data_set(path_dir):
    pdb.set_trace()
    classes = os.listdir(path_dir)
    classes.sort()
    class_num = len(classes)
    dataset=[]
    for i in range(class_num):
        class_name=classes[i]
        image_paths=[join(path_dir,class_name,img) for img in os.listdir(join(path_dir,class_name))]
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
        image_paths_flat += dataset[i].image_paths[0]
        labels_flat += [i] 
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
               
def do_train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder, 
      learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, step, 
      loss, train_op, summary_op, summary_writer, reg_losses, learning_rate_schedule_file, 
      stat, cross_entropy_mean, accuracy, 
      learning_rate, prelogits, random_rotate, random_crop, random_flip, prelogits_norm, prelogits_hist_max, use_fixed_image_standardization):
    batch_number = 0
    
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = get_learning_rate_from_file(learning_rate_schedule_file, epoch)
        
    if lr<=0:
        return False 

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]
    
    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch),1)
    image_paths_array = np.expand_dims(np.array(image_epoch),1)
    control_value = RANDOM_ROTATE * random_rotate + RANDOM_CROP * random_crop + RANDOM_FLIP * random_flip + FIXED_STANDARDIZATION * use_fixed_image_standardization
    control_array = np.ones_like(labels_array) * control_value
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})

    # Training loop
    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder:True, batch_size_placeholder:args.batch_size}
        tensor_list = [loss, train_op, step, reg_losses, prelogits, cross_entropy_mean, learning_rate, prelogits_norm, accuracy]
        if batch_number % 100 == 0:
            loss_, _, step_, reg_losses_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_, summary_str = sess.run(tensor_list + [summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step_)
        else:
            loss_, _, step_, reg_losses_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_ = sess.run(tensor_list, feed_dict=feed_dict)
         
        duration = time.time() - start_time
        stat['loss'][step_-1] = loss_
        stat['reg_loss'][step_-1] = np.sum(reg_losses_)
        stat['xent_loss'][step_-1] = cross_entropy_mean_
        stat['prelogits_norm'][step_-1] = prelogits_norm_
        stat['learning_rate'][epoch-1] = lr_
        stat['accuracy'][step_-1] = accuracy_
        stat['prelogits_hist'][epoch-1,:] += np.histogram(np.minimum(np.abs(prelogits_), prelogits_hist_max), bins=1000, range=(0.0, prelogits_hist_max))[0]
        
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tXent %2.3f\tRegLoss %2.3f\tAccuracy %2.3f\tLr %2.5f\tCl %2.3f' %
              (epoch, batch_number+1, args.epoch_size, duration, loss_, cross_entropy_mean_, np.sum(reg_losses_), accuracy_, lr_))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, global_step=step_)
    return True

# def evaluate(logits, labels, step, summary_writer, stat, epoch):
#     
#     print('Runnning forward pass on Validate images')
#     nrof_logits=int[logits.get_shape()[0]]
#     count=0
#     for i in range(nrof_logits):
#         emb=logits[i]
#         index = emb.index(max(emb))
#         if (index == labels[i]):
#             count=count+1
#     accuracy=count*100/nrof_logits
#             
#     print('Accuracy: %2.5f' % (accuracy))
#     
#     summary = tf.Summary()
#     #pylint: disable=maybe-no-member
#     summary.value.add(tag='lfw/accuracy', accuracy)
#     summary_writer.add_summary(summary, step)
#     
#     stat['lfw_accuracy'][epoch-1] = accuracy
            

def main():
    model_def='models.inception_resnet_v1'
    logs_base_dir = '/home/nemo/logs/kaggle'
    models_base_dir = '/home/nemo/models/kaggle'
    data_dir='/home/nemo/kaggle/data/train_160/'
    batch_size=90
    epoch_size=1000
    random_rotate=True
    random_crop=False
    image_size=160
    random_flip=True
    keep_probability=0.6
    embedding_size=512
    weight_decay=5e-4
    learning_rate_decay_epochs=100
    learning_rate_schedule_file='data/learning_rate_schedule_classifier_vggface2.txt'
    optimizer='ADAGRAD'
    gpu_memory_fraction=0.9
    max_nrof_epochs=300
    validate_every_n_epochs=5
    prelogits_hist_max=10.0
    use_fixed_image_standardization=True

    
    # import network
    network = importlib.import_module(model_def)
    
    # create logs and models dir.
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = join(os.path.expanduser(logs_base_dir), subdir)
    if not os.path.isdir(log_dir): 
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(models_base_dir), subdir)
    if not os.path.isdir(model_dir): 
        os.makedirs(model_dir)

    train_set = get_data_set(data_dir)  
    class_num = len(train_set) 
    
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        image_list, label_list = get_image_paths_and_labels(train_set)
        image_list_eval, label_list_eval = get_image_paths_and_labels(train_set)
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                             shuffle=True, seed=None, capacity=32)
        index_dequeue_op = index_queue.dequeue_many(batch_size*epoch_size, 'index_dequeue')
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None,1), name='labels')
        control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control')
        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                    dtypes=[tf.string, tf.int64],
                                    shapes=[(1,), (1,)],
                                    shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='enqueue_op')
        
        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)
                if random_rotate:
                    angle = np.random.uniform(low=-10.0, high=10.0)
                    image = misc.imrotate([image], angle, 'bicubic')
                if random_crop:
                    image = tf.random_crop(image, [image_size, image_size, 3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
                if random_flip:
                    image = tf.image.random_flip_left_right(image)
                image.set_shape((image_size, image_size, 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label])
    
        image_batch, label_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder, 
            shapes=[(image_size, image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * batch_size,
            allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')
        
        print('Total number of classes: %d' % class_num)
        print('Total number of examples: %d' % len(image_list))
        
        print('Building training graph')
        
        # Build the inference graph
        prelogits, _ = network.inference(image_batch, keep_probability, 
            phase_train=phase_train_placeholder, bottleneck_layer_size=embedding_size, 
            weight_decay=weight_decay)
        logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None, 
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                weights_regularizer=slim.l2_regularizer(weight_decay),
                scope='Logits', reuse=False)
        logits=tf.identity(logits,'logits')
        
        # Norm for the prelogits
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            learning_rate_decay_epochs*epoch_size, 1.0, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        
        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch, tf.int64)), tf.float32)
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
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            # Training and validation loop
            print('Running training')
            nrof_steps = max_nrof_epochs*epoch_size
            nrof_val_samples = int(math.ceil(max_nrof_epochs / validate_every_n_epochs))   # Validate every validate_every_n_epochs as well as in the last epoch
            stat = {
                'loss': np.zeros((nrof_steps,), np.float32),
                'center_loss': np.zeros((nrof_steps,), np.float32),
                'reg_loss': np.zeros((nrof_steps,), np.float32),
                'xent_loss': np.zeros((nrof_steps,), np.float32),
                'prelogits_norm': np.zeros((nrof_steps,), np.float32),
                'accuracy': np.zeros((nrof_steps,), np.float32),
                'val_loss': np.zeros((nrof_val_samples,), np.float32),
                'val_xent_loss': np.zeros((nrof_val_samples,), np.float32),
                'val_accuracy': np.zeros((nrof_val_samples,), np.float32),
                'lfw_accuracy': np.zeros((max_nrof_epochs,), np.float32),
                'lfw_valrate': np.zeros((max_nrof_epochs,), np.float32),
                'learning_rate': np.zeros((max_nrof_epochs,), np.float32),
                'time_train': np.zeros((max_nrof_epochs,), np.float32),
                'time_validate': np.zeros((max_nrof_epochs,), np.float32),
                'time_evaluate': np.zeros((max_nrof_epochs,), np.float32),
                'prelogits_hist': np.zeros((max_nrof_epochs, 1000), np.float32),
              }
            for epoch in range(1,max_nrof_epochs+1):
                step = sess.run(global_step, feed_dict=None)
                # Train for one epoch
                t = time.time()
                cont = do_train(sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
                    learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, global_step, 
                    total_loss, train_op, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file,
                    stat, cross_entropy_mean, accuracy, learning_rate,
                    prelogits, random_rotate, random_crop, random_flip, prelogits_norm, prelogits_hist_max, use_fixed_image_standardization)
                stat['time_train'][epoch-1] = time.time() - t
                
                if not cont:
                    break
                  

main()