import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

def audio_model(inputs, batch_size, seq_length, num_features, conv_filters=40):

    ##################### Create Network - start from input and insert more layers sequentially -> return it
    with slim.arg_scope([slim.layers.conv2d], padding='SAME'):

        net = slim.dropout(inputs)
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
        