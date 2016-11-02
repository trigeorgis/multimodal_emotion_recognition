import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

from config import Config

import pdb
import datetime
import numpy as np
import os
import time

MOVING_AVERAGE_DECAY = 0.9999
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
RNN_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]

tf.app.flags.DEFINE_integer('input_size', 224, "input image size")


activation = tf.nn.relu


def rnn(cnn,same,different,hidden_units,num_units_out_fc1):

  cnn = tf.reshape(cnn,[different,same,num_units_out_fc1])

  lstm = tf.nn.rnn_cell.LSTMCell(
            hidden_units,
            use_peepholes=True,
            cell_clip=100,
            state_is_tuple=True)

  stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * 2, state_is_tuple=True)

  # We have to specify the dimensionality of the Tensor so we can allocate
  # weights for the fully connected layers.
  outputs, states = tf.nn.dynamic_rnn(stacked_lstm, cnn, dtype=tf.float32)

  output = tf.reshape(outputs, (different * same, hidden_units))

  weights_initializer = tf.truncated_normal_initializer(
        stddev=RNN_WEIGHT_STDDEV)

  weights = _get_variable('weights',
                            shape=[hidden_units, 8],
                            initializer=weights_initializer,
                            weight_decay=RNN_WEIGHT_STDDEV,trainable = True)
  biases = _get_variable('biases',
                           shape=[8],
                           initializer=tf.zeros_initializer,trainable = True)
  output = tf.nn.xw_plus_b(output, weights, biases)

  return output



def inference(x, is_training,num_units_out_fc1=1500,
              num_classes=8,
              num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
              use_bias=False, # defaults to using batch norm
              bottleneck=True):
    
    x = tf.reshape(x,[-1,224,224,3])
    c = Config()
    c['bottleneck'] = bottleneck
    c['is_training'] = tf.convert_to_tensor(is_training,
                                            dtype='bool',
                                            name='is_training')
    c['ksize'] = 3
    c['stride'] = 1
    c['use_bias'] = use_bias
    c['fc_units_out'] = num_classes
    c['num_blocks'] = num_blocks
    c['stack_stride'] = 2

    with tf.variable_scope('scale1'):
        c['conv_filters_out'] = 64
        c['ksize'] = 7
        c['stride'] = 2
        x = conv(x, c)
        x = bn(x, c)
        x = activation(x)

    with tf.variable_scope('scale2'):
        x = _max_pool(x, ksize=3, stride=2)
        c['num_blocks'] = num_blocks[0]
        c['stack_stride'] = 1
        c['block_filters_internal'] = 64
        x = stack(x, c)

    with tf.variable_scope('scale3'):
        c['num_blocks'] = num_blocks[1]
        c['block_filters_internal'] = 128
        assert c['stack_stride'] == 2
        x = stack(x, c)

    with tf.variable_scope('scale4'):
        c['num_blocks'] = num_blocks[2]
        c['block_filters_internal'] = 256
        x = stack(x, c)

    with tf.variable_scope('scale5'):
        c['num_blocks'] = num_blocks[3]
        c['block_filters_internal'] = 512
        x = stack(x, c)

    # post-net
    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

    
    with tf.variable_scope('fc1'):
      x = fc1(x, c,is_training,num_units_out_fc1)

   
    return x


def stack(x, c):
    for n in range(c['num_blocks']):
        s = c['stack_stride'] if n == 0 else 1
        c['block_stride'] = s
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, c)
    return x


def block(x, c):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed. 
    # That is the case when bottleneck=False but when bottleneck is 
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    m = 4 if c['bottleneck'] else 1
    filters_out = m * c['block_filters_internal']

    shortcut = x  # branch 1

    c['conv_filters_out'] = c['block_filters_internal']

    if c['bottleneck']:
        with tf.variable_scope('a'):
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('b'):
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('c'):
            c['conv_filters_out'] = filters_out
            c['ksize'] = 1
            assert c['stride'] == 1
            x = conv(x, c)
            x = bn(x, c)
    else:
        with tf.variable_scope('A'):
            c['stride'] = c['block_stride']
            assert c['ksize'] == 3
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('B'):
            c['conv_filters_out'] = filters_out
            assert c['ksize'] == 3
            assert c['stride'] == 1
            x = conv(x, c)
            x = bn(x, c)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or c['block_stride'] != 1:
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            c['conv_filters_out'] = filters_out
            shortcut = conv(shortcut, c)
            shortcut = bn(shortcut, c)

    return activation(x + shortcut)


def bn(x, c):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if c['use_bias']:
        bias = _get_variable('bias', params_shape,
                             initializer=tf.zeros_initializer)
        return x + bias


    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer)
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer)

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer,
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        c['is_training'], lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #x.set_shape(inputs.get_shape()) ??

    return x


def fc1(x,c,is_training,num_units_out_fc1):
    
    num_units_in = x.get_shape()[1] 

    
    weights_initializer = tf.truncated_normal_initializer(
        stddev=FC_WEIGHT_STDDEV)

    weights = _get_variable('weights_fc1',
                            shape=[num_units_in, num_units_out_fc1],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV,trainable = is_training)
    biases = _get_variable('biases_fc1',
                           shape=[num_units_out_fc1],
                           initializer=tf.zeros_initializer,trainable = is_training)

    x = tf.nn.xw_plus_b(x, weights, biases)

    return x



def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=False):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


def conv(x, c):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')



def loss(predictions, labels):
  """Calculates the loss (mean squared error) from the predictions and the labels.

  Args:
    predictions: prediction tensor, float.
    labels: Labels tensor, float.

  Returns:
    loss: Loss (mse) tensor of type float.
  """
  loss = tf.reduce_mean(tf.square(predictions - labels))
  return loss


def pearson_loss(predictions, labels):
        """Calculates the pearson correlation from the predictions and the labels.

        Args:
          predictions: prediction tensor, float.
          labels: Labels tensor, float.

        Returns:
          loss: Loss (mse) tensor of type float.
        """
        xy = tf.mul(predictions,labels)
        x2 = tf.square(predictions)
        y2 = tf.square(labels)
                        
        nom1 = tf.reduce_mean(xy)
        E_x_one_value = tf.reduce_mean(predictions)

        E_y_one_value = tf.reduce_mean(labels)
            

        nom2 = tf.mul(E_x_one_value,E_y_one_value)

        nom = tf.sub(nom1,nom2)

        
        E_x2_one_value = tf.reduce_mean(x2)
        E_y2_one_value = tf.reduce_mean(y2)

        
        dnom1 = tf.sub( E_x2_one_value, tf.square(E_x_one_value))
        dnom2 = tf.sub( E_y2_one_value, tf.square(E_y_one_value))

                                        
        dnom = tf.mul(tf.sqrt(dnom1),tf.sqrt(dnom2))
        
                    
                    
        #calculate pearson one value
        #square_root_1_one_value = tf.sqrt(tf.sub(E_x2_one_value,tf.square(E_x_one_value)))
        #square_root_2_one_value = tf.sqrt(tf.sub(E_y2_one_value,tf.square(E_y_one_value)))
                            
                            
        #nominator_one_value = tf.sub(E_xy_one_value,E_x_times_E_y_one_value)
        #denominator_one_value = tf.mul(square_root_1_one_value,square_root_2_one_value)
                                    
        pearson_one_value = tf.div(nom,dnom)

                
        return pearson_one_value

def concordance_loss(predictions, labels):
        """Calculates the pearson correlation from the predictions and the labels.

        Args:
          predictions: prediction tensor, float.
          labels: Labels tensor, float.

        Returns:
          loss: Loss (mse) tensor of type float.
        """
        E_x = tf.reduce_mean(predictions)
        E_y = tf.reduce_mean(labels)

        S_x = tf.reduce_mean(tf.square(tf.sub(predictions,E_x))) 
        S_y = tf.reduce_mean(tf.square(tf.sub(labels,E_y))) 

        S_xy = tf.reduce_mean(tf.mul(tf.sub(predictions,E_x),tf.sub(labels,E_y))) 
        
        add = tf.add(tf.square(S_x),tf.square(S_y))
        add2 = tf.add(add,tf.square(tf.sub(E_x,E_y)))

        res = tf.div( 2*S_xy , add2 )                
        
                
        return 1 - res


def concordance_loss_trig(predictions, labels):
    
    pred_mean, pred_var = tf.nn.moments(predictions, (0,))
    gt_mean, gt_var = tf.nn.moments(labels, (0,))

    mean_cent_prod = tf.reduce_mean((predictions - pred_mean) * (labels - gt_mean))

    return (2 * mean_cent_prod) / (pred_var + gt_var + tf.square(pred_mean - gt_mean))
