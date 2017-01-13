import menpo
import tensorflow as tf
import numpy as np

from io import BytesIO
from pathlib import Path
from moviepy.editor import VideoFileClip
from menpo.visualize import progress_bar_str, print_progress
from menpo.image import Image

root_dir = Path('/vol/atlas/homes/gt108/db/RECOLA_CNN')

portion_to_id = dict(
    train=[15, 16 ,17 ,18 ,21 ,23 ,25 ,37 ,39 ,41 ,46 ,50 ,51 ,55 ,56, 60],
    valid=[14, 19, 24, 26, 28, 30, 34, 40, 42, 43, 44, 45, 52, 64, 65],
    test=[13, 20, 22, 32, 38, 47, 48, 49, 53, 54, 57, 58, 59, 62, 63]
)

def crop_face(img, boundary=50, group=None, shape=(256, 256)):
    pc = img.landmarks[group].lms
    nan_points = np.any(np.isnan(pc.points).reshape(-1, 2), 1)

    pc = PointCloud(pc.points[~nan_points, :])
    min_indices, max_indices = pc.bounds(boundary=boundary)
    h = max_indices[0] - min_indices[0]
    w = max_indices[1] - min_indices[1]
    pad = abs(w - h)

    try:
        index = 1 - int(w > h)
        min_indices[index] -= int(pad / 2.)
        max_indices[index] += int(pad / 2.) + int(pad) % 2

        img = img.crop(min_indices, max_indices, constrain_to_boundary=True)
    except Exception as e:
        print("Exception in crop_face", e)

    img = img.resize(shape)
    return img

landmarks_directory = Path('/vol/atlas/homes/grigoris/videos_external/panag/landmarks')

def get_samples(subject_id):
    arousal_label_path = root_dir / 'Ratings_affective_behaviour_CCC_centred/arousal/{}.csv'.format(subject_id)
    valence_label_path = root_dir / 'Ratings_affective_behaviour_CCC_centred/valence/{}.csv'.format(subject_id)
    
    clip = VideoFileClip(str(root_dir / "Video_recordings_MP4/{}.mp4".format(subject_id)))
    
    subsampled_audio = clip.audio.set_fps(16000)
    video_frames = []
    audio_frames = []
    
    for i in range(1, 7501):
        time = 0.04 * i
        
        video = clip.get_frame(time)
        audio = np.array(list(subsampled_audio.subclip(time - 0.04, time).iter_frames())).mean(1)
        
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
    subject_name = 'P{}'.format(subject_id)
    
    for i, (video, audio, label) in enumerate(zip(*get_samples(subject_name))):
        
        frame = Image.init_from_channels_at_back(video)
        lms_path = landmarks_directory / subject_name / "{}.pts".format(i)
        
        try:
            lms =  mio.import_landmark_file(lms_path)
        except:
            pass

        frame.landmarks['PTS'] = lms
        frame = crop_face(frame)
        
        example = tf.train.Example(features=tf.train.Features(feature={
                    'sample_id': _int_feauture(i),
                    'subject_id': _int_feauture(subject_id),
                    'label': _bytes_feauture(label.tobytes()),
                    'raw_audio': _bytes_feauture(audio.tobytes()),
                    'frame': _bytes_feauture(get_jpg_string(frame))
                }))

        writer.write(example.SerializeToString())
        del video, audio, label

def main(directory):
  for portion in portion_to_id.keys():
    print(portion)
    
    for subj_id in print_progress(portion_to_id[portion]):
      writer = tf.python_io.TFRecordWriter(
          (directory / 'tf_records' / portion / '{}.tfrecords'.format(subj_id)
          ).as_posix())
      serialize_sample(writer, subj_id)

if __name__ == "__main__":
  main(Path('/vol/atlas/homes/pt511/db/RECOLA'))
