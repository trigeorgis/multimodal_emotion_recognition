from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import glob
from pathlib import Path


slim = tf.contrib.slim

def get_split(dataset_dir=Path('../'), split_name='train', batch_size=32, seq_length=1, debugging=False):
    """Returns a data split of the RECOLA dataset.
    
    Args:
        split_name: A train/test/valid split name.
    Returns:
        The raw audio examples and the corresponding arousal/valence
        labels.
    """
    #dataset_dir = Path('./')
    paths = [str(x) for x in (dataset_dir).glob('*.tfrecords')]
    
    ## Vazei ston grafo ta instances 
    if split_name == 'train':
        filename_queue = tf.train.string_input_producer(paths, shuffle=True)
    else:
        filename_queue = tf.train.string_input_producer(paths, shuffle=False)
    
    ## Diavazei ta instances
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) #### Returns the next record (key, value pair) produced by a reader.
    features = tf.parse_single_example( #### Parses a single Example proto.
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'raw_audio': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
            'subject_id': tf.FixedLenFeature([], tf.int64),
            'frame': tf.FixedLenFeature([], tf.string),
        }
    )
    
    raw_audio = tf.decode_raw(features['raw_audio'], tf.float32)
    frame = tf.image.decode_jpeg(features['frame'])

    label = tf.decode_raw(features['label'], tf.float32)
    subject_id = features['subject_id']
   
    # 640 samples at 16KhZ corresponds to 40ms which is the frequency of
    # annotations.
    raw_audio.set_shape([640])
    label.set_shape([2])
    frame.set_shape([720, 1280, 3])

    # Number of threads should always be one, in order to load samples
    # sequentially.
    frames, audio_samples, labels, subject_ids = tf.train.batch( #### Creates batches of tensors in tensors.
        [frame, raw_audio, label, subject_id], seq_length, num_threads=1, capacity=1000)

    # Assert is an expensive op so we only want to use it when it's a must.
    if debugging:
        # Asserts that a sequence contains samples from a single subject.
        assert_op = tf.Assert(
            tf.reduce_all(tf.equal(subject_ids[0], subject_ids)),
            [subject_ids])
        
        with tf.control_dependencies([assert_op]):
            audio_samples = tf.identity(audio_samples)

        # TODO: Assert that the samples are in sequential order.

    audio_samples = tf.expand_dims(audio_samples, 0)
    labels = tf.expand_dims(labels, 0)
    frames = tf.expand_dims(frames, 0)

    if split_name == 'train':
        frames, audio_samples, labels = tf.train.shuffle_batch(
            [frames, audio_samples, labels], batch_size, 1000, 50, num_threads=1)
    else:
        frames, audio_samples, labels = tf.train.batch(
            [frames, audio_samples, labels], batch_size, num_threads=1, capacity=1000)

    return list([frames[:, 0, :, :], audio_samples[:, 0, :, :]]), labels[:, 0, :, :]
