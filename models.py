import tensorflow as tf
import numpy as np
import losses
import numpy as np
import csv
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.slim.nets import resnet_v1

slim = tf.contrib.slim

def prepare_video_model(net, inputs, hidden_units=128):
  batch_size, seq_length,_,_,_ = inputs[0].get_shape().as_list()
  return tf.reshape(slim.layers.linear(net, 2), (batch_size, seq_length, 2))

def prepare_audio_model(net, inputs, hidden_units=128):

  batch_size, seq_length, num_features = inputs[1].get_shape().as_list()

  net = tf.reshape(net, (batch_size, seq_length, num_features // 2 * 4))
  net = insert_LSTM(net, batch_size, seq_length, hidden_units)

  return tf.reshape(slim.layers.linear(net, 2), (batch_size, seq_length, 2))

def prepare_multimodal_model(net, inputs, hidden_units=128):
  batch_size, seq_length,_ = inputs[1].get_shape().as_list()
  
  net = insert_LSTM(net, batch_size, seq_length, hidden_units)

  return tf.reshape(slim.layers.linear(net, 2), (batch_size, seq_length, 2))

def get_prepared_model(name):
  name_to_fun = {'audio': prepare_audio_model, 'video': prepare_video_model, 'both': prepare_multimodal_model}

  if name in name_to_fun:
    features = name_to_fun[name]
  else:
    ValueError('Requested name [{}] not a valid model'.format(name))
  
  return features

def insert_LSTM(net, batch_size, seq_length, hidden_units=128):

  # Put LSTM on top of concatenation
  lstm = tf.nn.rnn_cell.LSTMCell(hidden_units,
                                use_peepholes=True,
                                cell_clip=100,
                                state_is_tuple=True)

  # We have to specify the dimensionality of the Tensor so we can allocate
  # weights for the fully connected layers.
  outputs,_ = tf.nn.dynamic_rnn(lstm, net, dtype=tf.float32)
  
  net = tf.reshape(outputs, (batch_size * seq_length, hidden_units))
  net = slim.dropout(net)

  return net

def video_model(inputs):

  batch_size, seq_length, height, width, channels = inputs[0].get_shape().as_list()
  video_input = tf.reshape(inputs[0], [batch_size * seq_length, 1, height, width, channels])[:,0, :, :]

  video_input = tf.to_float(video_input, name='ToFloat')
  video_network, _ = resnet_v1.resnet_v1_50(video_input, None)

  return video_network

def audio_model(inputs, conv_filters=40):
  
  batch_size, seq_length, num_features = inputs[1].get_shape().as_list()
  audio_input = tf.reshape(inputs[1], [batch_size * seq_length, 1, num_features, 1])

  # Create Network.
  with slim.arg_scope([slim.layers.conv2d], padding='SAME'):

    net = slim.dropout(audio_input)
    net = slim.layers.conv2d(net, conv_filters, (1, 20))
        
    # Subsampling of the signal to 8KhZ.
    net = tf.nn.max_pool(
            net,
            ksize=[1, 1, 2, 1],
            strides=[1, 1, 2, 1],
            padding='SAME',
            name='pool1')
    
    # Original model had 400 output filters for the second conv layer
    # but this trains much faster and achieves comparable accuracy.
    net = slim.layers.conv2d(net, conv_filters, (1, 40))
    
    net = tf.reshape(net, (batch_size * seq_length, num_features // 2, conv_filters, 1))
    
    # Pooling over the feature maps.
    net = tf.nn.max_pool(
        net,
        ksize=[1, 1, 10, 1],
        strides=[1, 1, 10, 1],
        padding='SAME',
        name='pool2')
  
  return net

def get_model(name):
  name_to_fun = {'audio': audio_model, 'video': video_model, 'both': combined_model}

  if name in name_to_fun:
    features = name_to_fun[name]
  else:
    ValueError('Requested name [{}] not a valid model'.format(name))
  
  return features

def combined_model(inputs):
  
  batch_size, seq_length, num_features = inputs[1].get_shape().as_list()
  _, _, height, width, channels = inputs[0].get_shape().as_list()

  audio_features = get_model('audio')(inputs)
  visual_features = get_model('video')(inputs)

  audio_features = tf.reshape(audio_features, (batch_size, seq_length, num_features // 2 * 4))
  visual_features = tf.reshape(visual_features, (batch_size, seq_length, int(visual_features.get_shape()[3])))

  return tf.concat(2, [audio_features, visual_features], name='concat')
