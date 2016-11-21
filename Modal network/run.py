import tensorflow as tf
from scipy import misc
import numpy as np
from tensorflow.contrib import learn
import data_provider
import audio_funcs
import losses
from tensorflow.contrib.slim.nets import resnet_v1

from tensorflow.python.platform import tf_logging as logging

slim = tf.contrib.slim
is_training = True

face = misc.imread('/home/pt511/Desktop/audiovisual_emotion_req-master/recola/database/1.jpg')
inputs = np.expand_dims(face, axis=0)
hidden_units = 20
data_folder = './'

def train():
	print('\nModal Network Training\n------------------')
	
	#print('\nTrain Video Model\n------------------')
	#video_network = video.get_trained_video_model()
	#print video_network

	#print('\nTrain Audio Model\n------------------')
	#audio_network = audio.audio_train()
	#print audio_network

	g = tf.Graph()
	with g.as_default():
		print(' -Start Training Model')

		with tf.Session() as sess:

			print(' -Load Database')
			frames, audio, ground_truth = data_provider.get_split(data_folder,2)
			batch_size, seq_length, height, width, channels = frames.get_shape().as_list()
			_, _, num_features = audio.get_shape().as_list()

			##################### Define video inputs + network
			print(' -Get Video Model')
			video_inputs = tf.reshape(frames, [batch_size * seq_length, 1, height, width, channels])[:,0, :, :]
			video_inputs = tf.to_float(video_inputs, name='ToFloat')

			video_network, _ = resnet_v1.resnet_v1_50(video_inputs, None)
			saver.restore(sess, "/test.ckpt")
			print 'Model successfully loaded'
			
			##################### Define audio inputs + network
			print(' -Get Audio Model')
			audio_inputs = tf.reshape(audio, [batch_size * seq_length, 1, num_features, 1])
			audio_inputs.set_shape([batch_size * seq_length,  1, num_features, 1])

			audio_network = audio_funcs.audio_model(audio_inputs, batch_size, seq_length, num_features)

			##################### Concatenate audio and video networks
			audio_network = tf.reshape(audio_network, (batch_size, seq_length, num_features // 2 * 4))
			video_network = tf.reshape(video_network, (batch_size, seq_length, int(video_network.get_shape()[3])))
			
			print(' -Concatenate Video+Audio Models and put LSTM on top')
			concat_network = tf.concat(2, [audio_network, video_network], name='concat')
			
			##################### Put LSTM on top of concatenation
			lstm = tf.nn.rnn_cell.LSTMCell(hidden_units,
										use_peepholes=True,
										cell_clip=100,
										state_is_tuple=True)

			# We have to specify the dimensionality of the Tensor so we can allocate
			# weights for the fully connected layers.
			outputs, states = tf.nn.dynamic_rnn(lstm, concat_network, dtype=tf.float32)
			
			net = tf.reshape(outputs, (batch_size * seq_length, hidden_units))
			net = slim.dropout(net)
			
			prediction = tf.reshape(slim.layers.linear(net, 2), (batch_size, seq_length, 2))
			

			total_loss = losses.get_losses(prediction,ground_truth)
			tf.scalar_summary('losses/total loss', total_loss)

			optimizer = tf.train.AdamOptimizer(0.01)


			train_op = slim.learning.create_train_op(total_loss,
								optimizer,
								summarize_gradients=True)
			
			logging.set_verbosity(tf.logging.INFO)
			merged = tf.merge_all_summaries()
			train_writer = tf.train.SummaryWriter('/losses', sess.graph)
			slim.learning.train(train_op,'./',summary_writer=train_writer,number_of_steps=2)

		'''
		with tf.Session(graph=g) as sess:
			writer = tf.train.SummaryWriter("./test", sess.graph)
			writer.close()
		'''

if __name__ == '__main__':
    train()

