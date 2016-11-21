import tensorflow as tf
import numpy as np
import audio_model as am
import losses
from tensorflow.python.platform import tf_logging as logging

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001,
                          '''Initial learning rate.''')
tf.app.flags.DEFINE_string('train_dir', './recola/',
                           '''Directory where to write event logs '''
                           '''and checkpoint.''')
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           '''If specified, restore this pretrained model '''
                           '''before beginning any training.''')

def train_audio_model(net, batch_size, seq_length, num_features, ground_truth, hidden_units=128):

    prediction = prepare_training_model(net, batch_size, seq_length, num_features, hidden_units)

    ##################### Get losses
    total_loss = losses.get_losses(prediction,ground_truth)
    tf.scalar_summary('losses/total loss', total_loss)
        
    optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

    ##################### Do training
    print(' -Start training audio model\n')
    with tf.Session() as sess:
            
        if FLAGS.pretrained_model_checkpoint_path:
            variables_to_restore = slim.get_variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, FLAGS.pretrained_model_checkpoint_path)
        
        train_op = slim.learning.create_train_op(total_loss,
                                                 optimizer,
                                                 summarize_gradients=True)
        
        logging.set_verbosity(1)
        slim.learning.train(train_op,
                            FLAGS.train_dir,
                            save_summaries_secs=60,
                            save_interval_secs=600)
    
    return net

def prepare_training_model(net, batch_size, seq_length, num_features, hidden_units=128):
    
    lstm = tf.nn.rnn_cell.LSTMCell(
        hidden_units,
        use_peepholes=True,
        cell_clip=100,
        state_is_tuple=True)
    
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm], state_is_tuple=True)
    
    # We have to specify the dimensionality of the Tensor so we can allocate
    # weights for the fully connected layers.
    net = tf.reshape(net, (batch_size, seq_length, num_features // 2 * 4))
    
    outputs, states = tf.nn.dynamic_rnn(stacked_lstm, net, dtype=tf.float32)
    
    net = tf.reshape(outputs, (batch_size * seq_length, hidden_units))
    net = slim.dropout(net)    

    return tf.reshape(slim.layers.linear(net, 2), (batch_size, seq_length, 2))

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
        