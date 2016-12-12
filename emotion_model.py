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
