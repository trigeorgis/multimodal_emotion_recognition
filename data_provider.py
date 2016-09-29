from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from models.slim.datasets import dataset_utils
import pdb 

slim = tf.contrib.slim

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'a frame from RECOLA database',
    'audio': 'the corresponding audio',
    'label': 'the label for this frame',
}

_SPLITS_TO_SIZES = {'train': 16, 'test': 15}

def get_split(split_name, dataset_dir, file_pattern='%s/*.tfrecords'):
  """Gets a dataset tuple with instructions for reading MNIST.
  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.
  Returns:
    A `Dataset` namedtuple.
  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """

  if split_name not in 'train':
    raise ValueError('split name %s was not recognized.' % split_name)


  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  reader = tf.TFRecordReader

  keys_to_features = {
      #'frame': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'raw_audio': tf.FixedLenFeature([150,640], tf.float32, default_value=tf.zeros([150,640], dtype=tf.float32)),
      'label': tf.FixedLenFeature(
          [2], tf.float32, default_value=tf.zeros([2], dtype=tf.float32)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(shape=[28, 28], channels=1),
      'audio': slim.tfexample_decoder.Tensor('raw_audio'),
      'label': slim.tfexample_decoder.Tensor('label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)
  

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      labels_to_names=labels_to_names)


def get_data(portion='train', batch_size=32):
    #root = '/vol/atlas/homes/gt108/db/RECOLA_CNN/wav_data/{}*.wav'.format(portion)
    #filename = tf.matching_files(root)
    #contents = tf.read_file(filename)
    #sampled_audio = tf.audio.decode_audio(
    #        contents, file_format='wav', samples_per_second=16000, channel_count=1)


    #tf.train.batch(tensors, batch_size, num_threads=1, capacity=batch_size*16)

    dataset = get_split('train','/vol/atlas/homes/gt108/db/RECOLA_CNN/tf_records/' )
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset,shuffle=True)
    [image, label, audio] = data_provider.get(['image', 'label','audio'])
    images,labels,audios = tf.train.batch([image, label, audio], batch_size, num_threads=1, capacity=batch_size*16)
    pdb.set_trace()

    return images,labels,audios


def main(_):
    get_data()


if __name__ == '__main__':
    tf.app.run()

'''
def get_data(portion='train', batch_size=32):
    root = '/vol/atlas/homes/gt108/db/RECOLA_CNN/wav_data/{}*.wav'.format(portion)
    filename = tf.matching_files(root)
    contents = tf.read_file(filename)
    sampled_audio = tf.audio.decode_audio(
            contents, file_format='wav', samples_per_second=16000, channel_count=1)


    tf.train.batch(tensors, batch_size, num_threads=1, capacity=batch_size*16)
'''
