import audio_funcs
import tensorflow as tf
import data_provider

slim = tf.contrib.slim

def get_trained_audio_model(data_folder='./'):

	g = tf.Graph()
	with g.as_default():
		
		##################### Load dataset.
		print(' -Load dataset')
		_, audio, ground_truth = data_provider.get_split(data_folder,2)

		##################### Define audio network
		batch_size, seq_length, num_features = audio.get_shape().as_list()
		audio_inputs = tf.reshape(audio, [batch_size * seq_length, 1, num_features, 1])
		audio_inputs.set_shape([batch_size * seq_length,  1, num_features, 1])

		print(' -Define Audio model')
		with tf.variable_scope('audio_net'):
			with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
								is_training=True):
				audio_network = audio_funcs.audio_model(audio_inputs, batch_size, seq_length, num_features)
				audio_network = audio_funcs.train_audio_model(audio_network, batch_size, seq_length, num_features, ground_truth)

	return audio_network

if __name__ == '__main__':
	print('\nStart Training Audio Model\n---------------------------')
	get_trained_audio_model() 
