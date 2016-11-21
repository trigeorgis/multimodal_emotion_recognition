import tensorflow as tf
import numpy as np
import losses
from tensorflow.python.platform import tf_logging as logging

slim = tf.contrib.slim

def get_video_model(net, batch_size, seq_length):
	return tf.reshape(slim.layers.linear(net, 2), (batch_size, seq_length, 2))

def train_video_model(net, batch_size, seq_length, ground_truth):

	print(' -Start training model\n')
	prediction = get_video_model(net, batch_size, seq_length)

	total_loss = losses.get_losses(prediction,ground_truth)
	tf.scalar_summary('losses/total loss', total_loss)

	optimizer = tf.train.AdamOptimizer(0.01)
	with tf.Session() as sess:
		train_op = slim.learning.create_train_op(total_loss,
							optimizer,
							summarize_gradients=True)
		
		logging.set_verbosity(tf.logging.INFO)
		merged = tf.merge_all_summaries()
		train_writer = tf.train.SummaryWriter('/losses', sess.graph)
		slim.learning.train(train_op,'./',summary_writer=train_writer,number_of_steps=2) # for testing: ,number_of_steps=2
		
	return net
