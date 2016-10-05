from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from pathlib import Path
slim = tf.contrib.slim


def get_split(split_name, batch_size=15, seq_length=150, debugging=False):
    """Returns a data split of the RECOLA dataset.
    
    Args:
        split_name: A train/test/valid split name.
    Returns:
        The raw audio examples and the corresponding arousal/valence
        labels.
    """
    dataset_dir = Path('/vol/atlas/homes/gt108/db/RECOLA_CNN/tf_records')
    paths = [str(x) for x in (dataset_dir / split_name).glob('*.tfrecords')]
    
    path = paths[0]
    filename_queue = tf.train.string_input_producer([path], shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'raw_audio': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
            'subject_id': tf.FixedLenFeature([], tf.int64)
        }
    )

    raw_audio = tf.decode_raw(features['raw_audio'], tf.float32)
    label = tf.decode_raw(features['label'], tf.float32)
    subject_id = features['subject_id']

    raw_audio.set_shape([640])
    label.set_shape([2])

    # Number of threads should always be one, in order to load samples
    # sequentially.
    audio_samples, labels, subject_ids = tf.train.batch(
        [raw_audio, label, subject_id], seq_length, num_threads=1, capacity=10000)
    

    # Assert is an expensive op so we only want to use it when it's a must.
    if debugging:
        # Asserts that a sequence contains samples from a single subject.
        assert_op = tf.Assert(
            tf.equal(subject_ids[0], subject_ids[10]),
            [subject_ids])

        with tf.control_dependencies([assert_op]):
            audio_samples = tf.identity(audio_samples)

    audio_samples = tf.expand_dims(audio_samples, 0)
    labels = tf.expand_dims(labels, 0)
    
    audio_samples, labels = tf.train.batch(
        [audio_samples, labels], batch_size,num_threads=1,capacity=10000 )

    return audio_samples[:, 0, :, :], labels[:, 0, :, :]
