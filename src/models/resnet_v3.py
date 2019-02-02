# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def get_network_layer(num_layers): 
    filter_list = [64, 64, 128, 256, 512]
    num_stages = 4
    if num_layers == 18:
        units = [2, 2, 2, 2]
    elif num_layers == 34:
        units = [3, 4, 6, 3]
    elif num_layers == 49:
        units = [3, 4, 14, 3]
    elif num_layers == 50:
        units = [3, 4, 14, 3]
    elif num_layers == 74:
        units = [3, 6, 24, 3]
    elif num_layers == 90:
        units = [3, 8, 30, 3]
    elif num_layers == 100:
        units = [3, 13, 30, 3]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))
    return units, filter_list

def residual_unit_v3(data, num_filter, stride, dim_match, name):
    with tf.variable_scope(name, reuse=None):
        net = slim.batch_norm(data)
        net = slim.conv2d(net, num_filter, 3, stride=1, scope='%s_conv0'%name)
        print(net)
        net = slim.conv2d(net, num_filter, 3, stride=stride, scope='%s_conv1'%name, activation_fn=None)
        print(net)
        if dim_match:
            shortcut = data
        else:
            shortcut = slim.conv2d(data, num_filter, 1, stride=stride, scope='%s__conv1sc'%name, activation_fn=None)
            print(shortcut)

        net = net + shortcut
        return net

def inference(images, keep_probability, phase_train=True, num_layers=50,
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.9,
        # epsilon to prevent 0s in variance.
        'epsilon': 2e-5,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return resnet_v3(images, num_layers, is_training=phase_train, 
              dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size, reuse=reuse)


def resnet_v3(inputs, num_layers, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=True, 
                        scope='ResnetV3'):
    """Creates the Resnet V3 model.
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      num_layers: network layer number
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
    end_points = {}
    print("num_layers %d"%num_layers)
    units, filter_list = get_network_layer(num_layers)
    num_stages = len(units)
    print(units)
    print(inputs)
  
    with tf.variable_scope(scope, 'ResnetV3', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            #here it needs to set the default args for batch_norm because we call it directly.
            with slim.arg_scope([slim.batch_norm], decay=0.9, epsilon=2e-5, updates_collections=None, variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES]): 
                with slim.arg_scope([slim.conv2d, slim.max_pool2d], stride=1, padding='SAME'): 
                    net = slim.conv2d(inputs, 32, 3, stride=1, scope='scale0_conv0') 
                    print(net) 
                    end_points['scale0_conv0'] = net 
                    
                    for i in range(num_stages): 
                        name = name='stage%d_unit%d' % (i + 1, 1) 
                        net = residual_unit_v3(net, filter_list[i+1], (2, 2), False, name=name) 
                        end_points[name] = net 
                        for j in range(units[i]-1): 
                            name = 'stage%d_unit%d' % (i+1, j+2) 
                            net = residual_unit_v3(net, filter_list[i+1], (1,1), True, name=name) 
                            end_points[name] = net

                    with tf.variable_scope('Logits'):
                        net = slim.batch_norm(net)
                        print(net)
                        net = slim.flatten(net)
                        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout')
                        end_points['PreLogitsFlatten'] = net
                
                    net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, 
                        scope='Bottleneck', reuse=reuse)
                    print(net)
 
    return net, end_points
