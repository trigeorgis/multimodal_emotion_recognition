from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import data_provider
import models
import losses
import math
import metrics

from menpo.visualize import print_progress
from pathlib import Path
from tensorflow.python.platform import tf_logging as logging

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                          '''If specified, restore this pretrained model '''
                          '''before beginning any training.''')
tf.app.flags.DEFINE_integer('batch_size', 15, '''The batch size to use.''')
tf.app.flags.DEFINE_string('model', 'audio','''Which model is going to be used: audio,video, or both ''')
tf.app.flags.DEFINE_string('dataset_dir', '/vol/atlas/homes/pt511/db/RECOLA/tf_records', 'The tfrecords directory.')
tf.app.flags.DEFINE_string('checkpoint_dir', '/vol/atlas/homes/pt511/ckpt/multimodal_emotion_recognition/audio_8Feb', 'The tfrecords directory.')
tf.app.flags.DEFINE_string('log_dir', '/vol/atlas/homes/pt511/ckpt/multimodal_emotion_recognition/audio_8Feb', 'The tfrecords directory.')
tf.app.flags.DEFINE_integer('num_examples', 101435, 'The number of examples in the test set')
tf.app.flags.DEFINE_string('eval_interval_secs', 300, 'The number of examples in the test set')

def evaluate(data_folder):

  g = tf.Graph()
  with g.as_default():

    # Load dataset.
    frames, audio, ground_truth = data_provider.get_split(data_folder, 'valid', FLAGS.batch_size)

    # Define model graph.
    with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
          is_training=False):
      prediction = models.audio_model(audio)

    # Computing MSE and Concordance values, and adding them to summary
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
          'eval/mse_arousal':slim.metrics.streaming_mean_squared_error(prediction[:,:,0], ground_truth[:,:,0]),
          'eval/mse_valence':slim.metrics.streaming_mean_squared_error(prediction[:,:,1], ground_truth[:,:,1]),
      })

    summary_ops = []
    conc_total = 0
    mse_total = 0
    for i, name in enumerate(['arousal', 'valence']):
      with tf.name_scope(name) as scope:
        concordance_cc2, values, updates = metrics.concordance_cc2(
                        tf.reshape(prediction[:,:,i], [-1]),
                        tf.reshape(ground_truth[:,:,i], [-1]))

        for n, v in updates.items():
          names_to_updates[n+'/'+name] = v

      op = tf.summary.scalar('eval/concordance_'+name, concordance_cc2)
      op = tf.Print(op, [concordance_cc2], 'eval/concordance_'+name)
      summary_ops.append(op)

      mse_eval = 'eval/mse_' + name
      op = tf.summary.scalar(mse_eval, names_to_values[mse_eval])
      op = tf.Print(op, [names_to_values[mse_eval]], mse_eval)
      summary_ops.append(op)

      mse_total += names_to_values[mse_eval]
      conc_total += concordance_cc2

    conc_total = conc_total / 2
    mse_total = mse_total / 2

    op = tf.summary.scalar('eval/concordance_total', conc_total)
    op = tf.Print(op, [conc_total], 'eval/concordance_total')
    summary_ops.append(op)

    op = tf.summary.scalar('eval/mse_total', mse_total)
    op = tf.Print(op, [mse_total], 'eval/mse_total')
    summary_ops.append(op)

    num_examples = FLAGS.num_examples
    num_batches = int(num_examples / (FLAGS.batch_size*150.0))
    logging.set_verbosity(1)

    # Setup the global step.
    eval_interval_secs = FLAGS.eval_interval_secs # How often to run the evaluation.
    slim.evaluation.evaluation_loop(
        '',
        FLAGS.checkpoint_dir,
        FLAGS.log_dir,
        num_evals=num_batches,
        eval_op=list(names_to_updates.values()),
        summary_op=tf.summary.merge(summary_ops),
        eval_interval_secs=eval_interval_secs)

def main(_):
    evaluate(FLAGS.dataset_dir)

if __name__ == '__main__':
    tf.app.run()
