import tensorflow as tf
import numpy as np
import audio_model as am

slim = tf.contrib.slim

def train_audio_model(net, batch_size, seq_length, num_features, hidden_units=128):

    prediction = get_train_model(net, batch_size, seq_length, num_features, hidden_units)

    ##################### Give data to model -> get predictions ( + ground truth -> get loss )
    print('Find prediction and get losses\n')
    for i, name in enumerate(['arousal', 'valence']):
        pred_single = tf.reshape(prediction[:, :, i], (-1,))
        gt_single = tf.reshape(ground_truth[:, :, i], (-1,))

        loss = losses.concordance_cc(pred_single, gt_single)
        tf.scalar_summary('losses/{} loss'.format(name), loss)

        mse = tf.reduce_mean(tf.square(pred_single - gt_single))
        tf.scalar_summary('losses/rmse {} loss'.format(name), mse)

        slim.losses.add_loss(loss / 2.)
        
    total_loss = slim.losses.get_total_loss()
        
    tf.scalar_summary('losses/total loss', total_loss)
        
    optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

    ##################### Do training
    print('Start Training\n')
    with tf.Session(graph=g) as sess:
            
        if FLAGS.pretrained_model_checkpoint_path:
            variables_to_restore = slim.get_variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, FLAGS.pretrained_model_checkpoint_path)
        
        train_op = slim.learning.create_train_op(total_loss,
                                                 optimizer,
                                                 summarize_gradients=True)
        print('-- start real training')
        logging.set_verbosity(1)
        slim.learning.train(train_op,
                            FLAGS.train_dir,
                            save_summaries_secs=60,
                            save_interval_secs=600)

def get_train_model(net, batch_size, seq_length, num_features, hidden_units=128):
    
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