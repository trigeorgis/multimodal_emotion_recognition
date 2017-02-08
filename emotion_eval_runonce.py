from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import data_provider
import models
import math
import numpy as np
import time
import os

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
tf.app.flags.DEFINE_string('checkpoint_dir', '/vol/atlas/homes/pt511/ckpt/multimodal_emotion_recognition/audio_7Feb', 'The tfrecords directory.')
tf.app.flags.DEFINE_string('log_dir', '/vol/atlas/homes/pt511/ckpt/multimodal_emotion_recognition/audio_scaled_1Feb', 'The tfrecords directory.')
tf.app.flags.DEFINE_integer('num_examples', 101435, 'The number of examples in the test set') #101435,
tf.app.flags.DEFINE_string('eval_interval_secs', 300, 'The number of examples in the test set')

def evaluate(data_folder):
  """Evaluates the model once. Prints in terminal the Accuracy and the UAR of the audio model.

  Args:
     data_folder: The folder that contains the test data.
  """

  g = tf.Graph()
  with g.as_default():

    # Load dataset.
    frames, audio, ground_truth = data_provider.get_split(data_folder, 'valid', FLAGS.batch_size)

    # Define model graph.
    with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
          is_training=False):
     prediction = models.get_model(FLAGS.model)(frames, audio)

    coord = tf.train.Coordinator()
    variables_to_restore = slim.get_variables_to_restore()

    num_batches = math.ceil(FLAGS.num_examples / (float(FLAGS.batch_size*100.0) )) 

    evaluated_predictions = []
    evaluated_labels = []

    saver = tf.train.Saver(variables_to_restore)
    model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    print('Loading model from {}'.format(model_path))

    with tf.Session() as sess:
        saver.restore(sess, model_path)
        tf.train.start_queue_runners(sess=sess)

        try:
            for _ in print_progress(range(num_batches), prefix="Batch"):
                pr, l = sess.run([prediction, ground_truth])
                evaluated_predictions.append(pr)
                evaluated_labels.append(l)

                if coord.should_stop():
                    break
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)

        predictions = np.reshape(evaluated_predictions, (-1, 2))
        labels = np.reshape(evaluated_labels, (-1, 2))
        conc_arousal = concordance_cc2(predictions[:,0], labels[:,0])
        conc_valence = concordance_cc2(predictions[:,1], labels[:,1])

        print('Concordance on valence : {}'.format(conc_valence))
        print('Concordance on arousal : {}'.format(conc_arousal))
        print('Concordance on total : {}'.format((conc_arousal+conc_valence)/2))
        r1 = predictions[:,0]
        r2 = labels[:,0]
        mean_cent_prod = ((r1 - r1.mean()) * (r2 - r2.mean())).mean()
        print('mean_cent_prod : {}'.format(mean_cent_prod))
        mse_arousal = sum((predictions[:,0] - labels[:,0])**2)/len(labels[:,0])
        print('MSE Arousal : {}'.format(mse_arousal))
        mse_valence = sum((predictions[:,1] - labels[:,1])**2)/len(labels[:,1])
        print('MSE valence : {}'.format(mse_valence))

        return conc_valence, conc_arousal, (conc_arousal+conc_valence)/2, mse_arousal, mse_valence

def concordance_cc2(r1, r2):
    mean_cent_prod = ((r1 - r1.mean()) * (r2 - r2.mean())).mean()
    return (2 * mean_cent_prod) / (r1.var() + r2.var() + (r1.mean() - r2.mean()) ** 2)

def main():
    evaluate(FLAGS.dataset_dir)

if __name__ == '__main__':
    main()
