import tensorflow as tf
import data_provider
import losses
import models
from pathlib import Path
from tensorflow.python.platform import tf_logging as logging

slim = tf.contrib.slim

# Create FLAGS
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001,
                         '''Initial learning rate.''')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0,
                         '''Epochs after which learning rate decays.''')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.97,
                         '''Learning rate decay factor.''')
tf.app.flags.DEFINE_integer('batch_size', 2, '''The batch size to use.''')
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                           '''How many preprocess threads to use.''')
tf.app.flags.DEFINE_string('train_dir', 'ckpt/train',
                          '''Directory where to write event logs '''
                          '''and checkpoint.''')
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                          '''If specified, restore this pretrained model '''
                          '''before beginning any training.''')
tf.app.flags.DEFINE_string(
   'pretrained_resnet_checkpoint_path', '',
   '''If specified, restore this pretrained resnet '''
   '''before beginning any training.'''
   '''This restores only the weights of the resnet model''')
tf.app.flags.DEFINE_integer('max_steps', 100000,
                           '''Number of batches to run.''')
tf.app.flags.DEFINE_string('train_device', '/gpu:0',
                          '''Device to train with.''')
tf.app.flags.DEFINE_string('model', 'video',
                          '''Which model is going to be used: audio,video, or both ''')


def train(data_folder = Path('../')):

  g = tf.Graph()
  with g.as_default():
    
    # Load dataset.
    data, ground_truth = data_provider.get_split(data_folder, 'train', FLAGS.batch_size)
    
    # Define model graph.
    with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                           is_training=True):

      model = models.get_model(FLAGS.model)(data)
      prediction = models.get_prepared_model(FLAGS.model)(model, data)

      total_loss = losses.get_losses(prediction, ground_truth)
      tf.scalar_summary('losses/total loss', total_loss)

      optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

    # Define session
    with tf.Session(graph=g) as sess:

      # Load if exists pretrained model.
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


if __name__ == '__main__':
  train()
