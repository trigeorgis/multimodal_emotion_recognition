from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from pathlib import Path

from scipy.io import wavfile
import numpy as np
import csv
import glob
from io import BytesIO
from pathlib import Path
from moviepy.editor import VideoFileClip
from menpo.visualize import progress_bar_str, print_progress
from menpo.image import Image
import menpo

slim = tf.contrib.slim

def get_split(split_name, batch_size, seq_length=1, debugging=False):
    """Returns a data split of the RECOLA dataset.
    
    Args:
        split_name: A train/test/valid split name.
    Returns:
        The raw audio examples and the corresponding arousal/valence
        labels.
    """
    dataset_dir = Path('./')
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
    label = tf.decode_raw(features['label'], tf.float32)
    frame = tf.image.decode_jpeg(features['frame'])
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
 
    return frames[:, 0, :, :], audio_samples[:, 0, :, :], labels[:, 0, :, :]


def shuffle(data):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    shuffle_data = data[indices,] 
    return shuffle_data


def load_wav_files(dataset_dir):
    paths = glob.glob(dataset_dir + "*.wav")
    print(paths)
    data_shape = 4*480
    #data = np.empty((0,data_shape), float)
    data_dict = dict()
    for wav_path in paths:
        sample_rate,wav_data = wavfile.read(wav_path)
        subject_id = os.path.basename(wav_path)[1:3]
        print(os.path.basename(wav_path))
        wav_data = wav_data.reshape([wav_data.shape[0]/data_shape, data_shape])
        #data = np.vstack([data,wav_data])
        data_dict[str(subject_id)] = wav_data

    
    return data_dict

def load_labels(dataset_dir):
    paths = glob.glob(dataset_dir + "*.csv")

    #label_data = np.empty((0, 7), float)
    label_dict = dict()
    for labels in paths:
        data = []
        f = False
        subject_id = os.path.basename(labels)[:-4]
        with open(labels, 'rb') as csvfile:
          spamreader = csv.reader(csvfile, delimiter=';')
          for row in spamreader:
              if not f:
                f = True
                continue
              data.append([float(i) for i in ', '.join(row).split(',')])
        data = np.mean(np.asarray(data)[1:, 1:], axis=1)
        label_dict[str(subject_id)] = data
        #label_data = np.vstack([np.asarray(label_data), np.asarray(data)])

    return label_dict

def get_raw_data(wavs_dir, labels_dir, batch_size, seq_length=150):
    wavs = load_wav_files(wavs_dir)
    labels = load_labels(labels_dir)

    # if string is in list string. e.g. any("P17" in s for s in labels.keys())

    data = shuffle(data)
    audio_samples = tf.expand_dims(data, 0)
    
    audio_samples, labels, subject_ids = tf.train.batch( #### Creates batches of tensors in tensors.
         [audio_samples, labels], seq_length, num_threads=1, capacity=1000) # capacity: An integer. The maximum number of elements in the queue.

    audio_samples = tf.expand_dims(audio_samples, 0)
    labels = tf.expand_dims(labels, 0)
    
    if split_name == 'train':
        audio_samples, labels = tf.train.shuffle_batch(
            [audio_samples, labels], batch_size, 1000, 50, num_threads=1)
    else:
        audio_samples, labels = tf.train.batch(
            [audio_samples, labels], batch_size, num_threads=1, capacity=1000)

    return audio_samples[:, 0, :, :], labels[:, 0, :, :]

def get_samples(subject_id):
    arousal_label_path = './recola/emotional_behaviors/P16_arousal.csv'#.format(subject_id)
    valence_label_path = './recola/emotional_behaviors/P16_valence.csv'#.format(subject_id)
    
    clip = VideoFileClip(str("./recola/P{}.mp4".format(subject_id)))
    
    print('Clip loaded')

    subsampled_audio = clip.audio.set_fps(16000)
    video_frames = []
    audio_frames = []
                    # 7501
    for i in range(1, 10):
        print('index : {}'.format(i))
        time = (4 * i) / 100.
        print('-- time : {}'.format(time))
        video = clip.get_frame(time)
        print(type(video))
        print(video.shape)
        audio = np.array(list(subsampled_audio.subclip(time - 0.04, time).iter_frames())).mean(1)[:640]
        
        video_frames.append(video)
        audio_frames.append(audio.astype(np.float32))
    
    arousal = np.loadtxt(str(arousal_label_path), delimiter=',')[:, 1][1:]
    valence = np.loadtxt(str(valence_label_path), delimiter=',')[:, 1][1:]
    
    return video_frames, audio_frames, np.dstack([arousal, valence])[0].astype(np.float32)

def get_jpg_string(im):
    # Gets the serialized jpg from a menpo `Image`.
    fp = BytesIO()
    menpo.io.export_image(im, fp, extension='jpg')
    fp.seek(0)
    return fp.read()

def _int_feauture(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feauture(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_sample(writer, subject_id):
    
    for i, (video, audio, label) in enumerate(zip(*get_samples('{}'.format(subject_id)))):
        frame = Image.init_from_channels_at_back(video)
        
        example = tf.train.Example(features=tf.train.Features(feature={
                    'sample_id': _int_feauture(i),
                    'subject_id': _int_feauture(subject_id),
                    'label': _bytes_feauture(label.tobytes()),
                    'raw_audio': _bytes_feauture(audio.tobytes()),
                    'frame': _bytes_feauture(get_jpg_string(frame))
                }))

        writer.write(example.SerializeToString())
        del video, audio, label
