import tensorflow as tf
import numpy as np

slim = tf.contrib.slim


def audio_model(inputs, hidden_units=128, conv_filters=40):
    batch_size, seq_length, num_features = inputs.get_shape().as_list()
    
    with slim.arg_scope([slim.layers.conv2d], padding='SAME'):
        inputs = tf.reshape(inputs, [batch_size * seq_length, 1, num_features, 1])
        inputs.set_shape([batch_size * seq_length,  1, num_features, 1])
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

        lstm = tf.nn.rnn_cell.LSTMCell(
            hidden_units,
            use_peepholes=True,
            cell_clip=100,
            state_is_tuple=True)

        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * 2, state_is_tuple=True)

        # We have to specify the dimensionality of the Tensor so we can allocate
        # weights for the fully connected layers.
        net = tf.reshape(net, (batch_size, seq_length, num_features // 2 * 4))
        outputs, states = tf.nn.dynamic_rnn(stacked_lstm, net, dtype=tf.float32)

        net = tf.reshape(outputs, (batch_size * seq_length, hidden_units))
        net = slim.dropout(net)
        return tf.reshape(slim.layers.linear(net, 2), (batch_size, seq_length, 2))

#(16, 50, 150, 640)
def audio_model_detailed(inputs):
    num_features = 640
    num_batch = 50
    grad_clip = 100
    num_hidden = 128
    check_frequency = 50
    seq_length = 150
    filter_size = 20
    num_filters = 40

    dropout = tf.nn.dropout(inputs, 0.5)

    l_reshape = tf.reshape(dropout, [-1, 1, num_features * seq_length])

    with tf.name_scope('conv1d_1') as scope:
        # for 1d convolution make size: [-1,1,num_features * seq_length,1] to fit : [batch, in_height, in_width, in_channels]
        ninputs = tf.expand_dims(l_reshape, 3)
        # kernel should fit : [filter_height, filter_width, in_channels, out_channels] and filter_height is 1 because tensorflow has 2D conv (doesnt support 1D conv)
        kernel = tf.Variable(
            tf.truncated_normal(
                [1, filter_size, 1, num_filters],
                dtype=tf.float32,
                stddev=1e-1),
            name='weights')
        # h mipws : tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)   
        # no bias
        l_conv = tf.nn.conv2d(ninputs, kernel, [1, 1, 1, 1], padding='SAME')
        l_conv = tf.nn.relu(l_conv, name=scope)

    # pool1
    l_pool = tf.nn.max_pool(
        l_conv,
        ksize=[1, 1, 2, 1],
        strides=[1, 1, 2, 1],
        padding='SAME',
        name='pool1')

    l_pool = tf.reshape(l_pool, [num_batch, -1, num_filters])
    l_pool = tf.transpose(l_pool, [0, 2, 1])

    # conv1d_2
    with tf.name_scope('conv2_1') as scope:
        ninputs = tf.expand_dims(l_pool, 3)
        kernel2 = tf.Variable(
            tf.truncated_normal(
                [1, 400, 1, num_filters], dtype=tf.float32, stddev=1e-1),
            name='weights')

        l_conv = tf.nn.conv2d(ninputs, kernel2, [1, 1, 1, 1], padding='SAME')
        l_conv = tf.nn.relu(l_conv, name=scope)

    # pool2
    l_pool = tf.nn.max_pool(
        l_conv,
        ksize=[1, 1, 1, num_filters // 2],
        strides=[1, 1, 1, num_filters // 2],
        padding='SAME',
        name='pool2')

    l_pool = tf.transpose(l_pool, [0, 1, 3, 2])
    l_pool = tf.reshape(l_pool, [num_batch, -1, num_features * seq_length // 2])

    # Current data input shape: (batch_size, seq_length, num_features)
    l_reshape = tf.reshape(l_pool, [-1, seq_length, num_features])
    dropout = tf.nn.dropout(l_reshape, 0.5)
    # Required shape: 'n_steps' tensors list of shape (batch_size, num_features)

    # Permuting batch_size and seq_length
    l_reshape = tf.transpose(dropout, [1, 0, 2])
    # Reshaping to (seq_length*batch_size, num_features)
    l_reshape = tf.reshape(l_reshape, [-1, num_features])
    # Split to get a list of 'seq_length' tensors of shape (batch_size, num_features)
    l_reshape = tf.split(0, seq_length, l_reshape)

    lstm = tf.nn.rnn_cell.LSTMCell(
        num_hidden,
        use_peepholes=True,
        cell_clip=grad_clip,
        state_is_tuple=True)
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * 2, state_is_tuple=True)

    outputs, states = tf.nn.rnn(stacked_lstm, l_reshape, dtype=tf.float32)
    l_reshape = tf.reshape(outputs, [-1, num_hidden])
    dropout = tf.nn.dropout(l_reshape, 0.5)

    with tf.name_scope('fc') as scope:
        fcw = tf.Variable(
            tf.truncated_normal(
                [num_hidden, 2], dtype=tf.float32, stddev=1e-1),
            name='weights')
        #no bias
        fcl = tf.matmul(dropout, fcw)

    l_out = tf.reshape(fcl, [-1, seq_length, 2])
