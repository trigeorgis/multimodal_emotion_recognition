import tensorflow as tf
import data_provider
import video_funcs
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1

slim = tf.contrib.slim

def get_trained_video_model(data_folder='./'):

	g = tf.Graph()
	with g.as_default():
		##################### Load dataset.
		print(' -Load dataset')
		frames, _, ground_truth = data_provider.get_split(data_folder,2)
		
		##################### Define video network
		video_batch_size, video_seq_length, height, width, channels = frames.get_shape().as_list()
		video_inputs = tf.reshape(frames, [video_batch_size * video_seq_length, 1, height, width, channels])[:,0, :, :]
		video_inputs = tf.to_float(video_inputs, name='ToFloat')
		
		print(' -Define Video model')
		with tf.variable_scope('video_net'):
			with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training=True)):
				video_network, _ = resnet_v1.resnet_v1_50(video_inputs, None)
				video_network = video_funcs.train_video_model(video_network, video_batch_size, video_seq_length, ground_truth)

	return video_network


if __name__ == '__main__':
	print('\nStart Training Video Model\n---------------------------')
	get_trained_video_model() 
