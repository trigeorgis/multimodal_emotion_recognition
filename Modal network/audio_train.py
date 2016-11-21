import audio_funcs

def audio_train(data_folder='./'):

	g = tf.Graph()
	with g.as_default():
		
		##################### Load dataset.
		print('\n-Load dataset\n')
		frames, audio, ground_truth = data_provider.get_split(data_folder,2)

		##################### Define audio network
		batch_size, seq_length, num_features = audio.get_shape().as_list()
		audio_inputs = tf.reshape(audio, [batch_size * seq_length, 1, num_features, 1])
		audio_inputs.set_shape([batch_size * seq_length,  1, num_features, 1])

		print('-Define Audio model\n')
		with tf.variable_scope('audio_net'):
			with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
								is_training=True):
				audio_network = audio_funcs.audio_model(audio_inputs, batch_size, seq_length, num_features)
				audio_network = audio_funcs.train_audio_model(audio_network, batch_size, seq_length, num_features, ground_truth)

	return audio_network