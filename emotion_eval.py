from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import data_provider
import models
import losses

from menpo.visualize import print_progress
from pathlib import Path

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                          '''If specified, restore this pretrained model '''
                          '''before beginning any training.''')

def evaluate(data_folder = Path('../')):

  g = tf.Graph()
  with g.as_default():
    
    # Load dataset.
    frames, audio, ground_truth = data_provider.get_split(data_folder, 'valid', FLAGS.batch_size)
    
    # Define model graph.
    with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                           is_training=False):
      prediction = models.get_model(FLAGS.model)(data)

    with tf.Session(graph=g) as sess:
      # Restore pretrained model variables from checkpoint
      variables_to_restore = slim.get_variables_to_restore()
      restorer = tf.train.Saver(variables_to_restore)
      restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)

      tf.train.start_queue_runners(sess=sess)

      accuracy = losses.concordance_cc2(prediction, ground_truth)
      
      return sess.run(accuracy)

      '''
      predictions = []
      gts = []

      for i in print_progress(range(5)):
        p, gt = sess.run([prediction, ground_truth])
        predictions.append(p)
        gts.append(gt)
      '''

def main(_):
    evaluate()

if __name__ == '__main__':
    tf.app.run()
